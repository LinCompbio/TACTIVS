"""TACTIVS reference-only transductive readout."""

from .bundle import BUNDLE_SCHEMA, load_bundle_specs
from .cache import EmbeddingCache, TargetData, load_projection
from .core import ScoreResult, score_library
from .reference_pool import ReferencePool

__all__ = [
    "BUNDLE_SCHEMA",
    "EmbeddingCache",
    "ReferencePool",
    "ScoreResult",
    "TargetData",
    "load_bundle_specs",
    "load_projection",
    "score_library",
]
__version__ = "0.1.0"
