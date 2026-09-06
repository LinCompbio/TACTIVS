"""Fit a covariance-only whitener from an EPT cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .cache import EmbeddingCache


def fit_whitener(
    cache_path: Path,
    output: Path,
    shrinkage: float = 0.01,
    eps: float = 1e-5,
    molecule_batch_size: int = 4096,
) -> dict[str, int]:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if molecule_batch_size < 1:
        raise ValueError("molecule_batch_size must be positive")

    cache = EmbeddingCache(cache_path)
    dimension = cache.store.dim
    molecule_count = sum(
        len(cache.meta[target_id]["parent_counts"]) for target_id in cache.target_ids
    )
    if molecule_count <= dimension:
        raise ValueError(
            f"whitener fitting needs more than {dimension} molecules; "
            f"the cache contains {molecule_count}"
        )
    covariance_sum = torch.zeros((dimension, dimension), dtype=torch.float64)

    for target_id in cache.target_ids:
        meta = cache.meta[target_id]
        starts = meta["parent_starts"]
        counts = meta["parent_counts"]
        count = len(counts)
        if count < 2:
            raise ValueError(f"target {target_id} needs at least two molecules")

        total = torch.zeros(dimension, dtype=torch.float64)
        second_moment = torch.zeros((dimension, dimension), dtype=torch.float64)
        for begin in range(0, count, molecule_batch_size):
            batch_counts = counts[begin : begin + molecule_batch_size]
            raw = cache.store.load_ranges(
                starts[begin : begin + molecule_batch_size],
                batch_counts,
                torch.device("cpu"),
            )
            centers = _molecule_means(raw, batch_counts).to(torch.float64)
            total += centers.sum(0)
            second_moment += centers.T @ centers

        mean = total / count
        covariance = second_moment / count - torch.outer(mean, mean)
        covariance_sum += covariance

    covariance = (covariance_sum / len(cache.target_ids)).numpy()
    covariance = (covariance + covariance.T) / 2.0
    average_variance = float(np.trace(covariance) / dimension)
    if average_variance <= 0.0:
        raise ValueError("cache embeddings have no variance")
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * average_variance * np.eye(dimension)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_scales = 1.0 / np.sqrt(np.maximum(eigenvalues, 0.0) + eps)
    projection = (eigenvectors * inverse_scales) @ eigenvectors.T

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        projection=projection.astype(np.float32),
        shrinkage=np.asarray(shrinkage),
        eps=np.asarray(eps),
    )
    return {
        "targets": len(cache.target_ids),
        "molecules": molecule_count,
        "dimension": dimension,
    }


def _molecule_means(raw: torch.Tensor, counts: np.ndarray) -> torch.Tensor:
    count_tensor = torch.as_tensor(counts, dtype=raw.dtype)
    codes = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64), count_tensor.to(torch.int64)
    )
    centers = torch.zeros((len(counts), raw.shape[1]), dtype=raw.dtype)
    centers.index_add_(0, codes, raw)
    return centers / count_tensor[:, None]
