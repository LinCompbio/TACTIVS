"""TACTIVS reference-only transductive readout."""

from .cache import EmbeddingCache, TargetData, load_projection
from .core import ScoreResult, score_library
from .reference_pool import ReferencePool

__all__ = [
    "EmbeddingCache",
    "ReferencePool",
    "ScoreResult",
    "TargetData",
    "load_projection",
    "score_library",
]
__version__ = "0.1.0"
