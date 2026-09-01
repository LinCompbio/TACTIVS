"""Stable fingerprints for reproducible selection and benchmark runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = "tactivs-reference-only-v1"
REPRODUCIBILITY_SEED = 20260720


def configure_determinism(seed: int = REPRODUCIBILITY_SEED) -> None:
    """Configure process-level RNGs and deterministic Torch kernels."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_set_sha256(paths: list[Path]) -> str:
    """Hash file names and contents in a stable, path-location-independent order."""
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths), key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def cache_sha256(cache: Path) -> str:
    cache = Path(cache)
    if (cache / "index.npz").exists():
        paths = [cache / "index.npz"]
        paths.extend(sorted((cache / "embeddings").glob("embeddings_*.npy")))
    else:
        paths = [cache / "target_manifest.npz", cache / "target_parent_ids.npz"]
        paths.extend(sorted(cache.glob("embeddings_*.npy")))
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cache fingerprint inputs missing: {missing}")
    return artifact_set_sha256(paths)


def source_sha256(root: Path) -> str:
    root = Path(root)
    paths = sorted((root / "src" / "tactivs").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*.py")))
    return artifact_set_sha256(paths)


def software_versions(*, include_optuna: bool = False) -> dict[str, str]:
    packages = ["numpy", "pandas", "rdkit", "torch"]
    if include_optuna:
        packages.append("optuna")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }
