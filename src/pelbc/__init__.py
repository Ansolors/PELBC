"""PELBC offline encounter-level inference prototype."""

from .predictor import EncounterPredictor
from .version import __version__

__all__ = ["EncounterPredictor", "__version__"]
