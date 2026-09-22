"""Thalamus — a semantic interrupt controller.

The model decides *when*; your code decides *what*.
"""

__version__ = "0.1.0"

from .channels import BANK, Channel, Tier
from .dsp import Polarity, Trace, TriggerSpec, attractors, required_tau, suggest_band
from .engine import Engine, Frame
from .rules import Card, Rule, Signals
from .state import RollingState, Utterance

__all__ = [
    "BANK", "Channel", "Tier", "Polarity", "Trace", "TriggerSpec",
    "attractors", "required_tau", "suggest_band", "Engine", "Frame",
    "Card", "Rule", "Signals", "RollingState", "Utterance", "__version__",
]
