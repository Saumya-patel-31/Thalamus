"""TypeSafe Jev, over stdlib urllib.

Deliberately not using the official SDK: this whole project installs with zero
dependencies, and the API is a single POST. If you would rather use the SDK
(`pip install typesafe-sdk`), the request shape below maps one-to-one.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from . import Verdict

ENDPOINT = os.environ.get("TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone")
PRICE_PER_MTOK_INPUT = 0.042  # output is free, which is why we ask twice


class JevBackend:
    name = "jev"

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        timeout: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is not set. Run with --backend mock to drive "
                "the engine from a scripted transcript instead."
            )

    # Blocking call, run off the loop.
    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "thalamus/0.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode()), resp.status

    async def evaluate(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        oracle: dict[str, Any] | None = None,
    ) -> Verdict:
        del oracle  # never leaves the process
        payload = {"model": self.model, "state": state, "questions": questions}
        started = time.perf_counter()
        loop = asyncio.get_running_loop()

        last_err: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data, _ = await loop.run_in_executor(None, self._post, payload)
                usage = data.get("usage") or {}
                return Verdict(
                    answers=data.get("answers") or {},
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=data.get("model", self.model),
                )
            except urllib.error.HTTPError as exc:
                last_err = f"HTTP {exc.code}"
                # 429 rate limit / 529 overloaded are the documented retryables.
                if exc.code in (429, 529) and attempt < self.max_retries:
                    backoff = (2**attempt) * 0.25 + random.uniform(0, 0.1)
                    await asyncio.sleep(backoff)
                    continue
                if exc.code == 401:
                    last_err = "HTTP 401 - check TYPESAFE_API_KEY"
                elif exc.code == 422:
                    detail = exc.read()[:200].decode(errors="replace")
                    last_err = f"HTTP 422 - malformed questions: {detail}"
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    await asyncio.sleep((2**attempt) * 0.25)
                    continue
                break

        return Verdict(
            answers={},
            latency_ms=(time.perf_counter() - started) * 1000,
            model=self.model,
            error=last_err,
        )
