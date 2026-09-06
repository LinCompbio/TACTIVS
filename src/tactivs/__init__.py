"""TACTIVS reference-only transductive readout."""

from .cache import EmbeddingCache, TargetData, load_projection
from .core import ScoreResult, score_library
from .reference_pool import ReferencePool
from .whitening import fit_whitener

__all__ = [
    "EmbeddingCache",
    "ReferencePool",
    "ScoreResult",
    "TargetData",
    "load_projection",
    "score_library",
    "fit_whitener",
]
__version__ = "0.1.0"
