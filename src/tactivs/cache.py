"""Read benchmark and user EPT embedding caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TargetData:
    target_id: str
    embeddings: torch.Tensor
    parent_ids: np.ndarray
    parent_starts: np.ndarray
    parent_counts: np.ndarray
    parent_labels: np.ndarray | None = None
    raw_embeddings: torch.Tensor | None = None
    projection: torch.Tensor | None = None


def load_projection(path: Path, device: torch.device) -> torch.Tensor:
    """Load the frozen covariance-only whitening projection."""
    with np.load(path, allow_pickle=False) as data:
        if "projection" not in data:
            raise KeyError(f"{path} does not contain a projection array")
        if "mean" in data:
            raise ValueError(
                "TACTIVS expects a covariance-only whitener without a fixed mean"
            )
        projection = np.asarray(data["projection"], dtype=np.float32)
    return torch.as_tensor(projection, device=device)


def molecule_means(raw: torch.Tensor, parent_counts: np.ndarray) -> torch.Tensor:
    """Average conformer embeddings independently for each parent molecule."""
    counts = torch.as_tensor(parent_counts, dtype=raw.dtype, device=raw.device)
    codes = torch.repeat_interleave(
        torch.arange(len(parent_counts), dtype=torch.int64, device=raw.device),
        counts.to(torch.int64),
    )
    centers = torch.zeros(
        (len(parent_counts), raw.shape[1]), dtype=raw.dtype, device=raw.device
    )
    centers.index_add_(0, codes, raw)
    centers /= counts[:, None]
    return centers


def molecule_balanced_mean(
    raw: torch.Tensor, parent_counts: np.ndarray
) -> torch.Tensor:
    """Return the mean after assigning equal weight to every molecule."""
    return molecule_means(raw, parent_counts).mean(0)


def external_whiten(
    raw: torch.Tensor, projection: torch.Tensor, parent_counts: np.ndarray
) -> torch.Tensor:
    """Apply frozen covariance whitening with an unlabeled target-local mean."""
    target_mean = molecule_balanced_mean(raw, parent_counts)
    return F.normalize((raw - target_mean) @ projection, dim=1)


class EmbeddingStore:
    def __init__(self, cache: Path):
        paths = sorted(cache.glob("embeddings_*.npy"))
        if not paths:
            raise FileNotFoundError(f"no embeddings_*.npy shards in {cache}")
        self.arrays = [np.load(path, mmap_mode="r") for path in paths]
        dimensions = {array.shape[1] for array in self.arrays}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent embedding dimensions: {dimensions}")
        self.dim = dimensions.pop()
        self.starts = np.cumsum([0] + [array.shape[0] for array in self.arrays[:-1]])
        self.ends = np.cumsum([array.shape[0] for array in self.arrays])

    def load_ranges(
        self, starts: np.ndarray, counts: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Gather molecule ranges from a shared, possibly non-contiguous store."""
        starts = np.asarray(starts, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        if starts.shape != counts.shape or np.any(counts < 1):
            raise ValueError("invalid embedding ranges")
        if np.any(starts < 0) or np.any(starts + counts > self.ends[-1]):
            raise ValueError("embedding range lies outside the shard store")

        output = torch.empty(
            (int(counts.sum()), self.dim), dtype=torch.float32, device=device
        )
        cursor = 0
        for begin in range(0, len(starts), 20000):
            block_starts = starts[begin : begin + 20000]
            block_counts = counts[begin : begin + 20000]
            width = int(block_counts.max())
            offsets = np.arange(width, dtype=np.int64)[None, :]
            valid = offsets < block_counts[:, None]
            rows = (block_starts[:, None] + offsets)[valid]
            shard_codes = np.searchsorted(self.ends, rows, side="right")
            block = np.empty((len(rows), self.dim), dtype=np.float32)
            for shard in np.unique(shard_codes):
                mask = shard_codes == shard
                local_rows = rows[mask] - int(self.starts[int(shard)])
                block[mask] = self.arrays[int(shard)][local_rows]
            output[cursor : cursor + len(rows)] = torch.from_numpy(block).to(device)
            cursor += len(rows)
        return output


class EmbeddingCache:
    """Load targets without touching labels unless ``include_labels=True``."""

    def __init__(self, cache: Path):
        self.cache = Path(cache)
        self.index_path = self.cache / "index.npz"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"embedding index not found: {self.index_path}")
        self.store = EmbeddingStore(self.cache / "embeddings")
        with np.load(self.index_path, allow_pickle=False) as data:
            target_ids = list(map(str, data["target_ids"]))
            offsets = np.asarray(data["parent_offsets"], dtype=np.int64)
            parent_ids = np.asarray(data["parent_ids"])
            starts = np.asarray(data["parent_starts"], dtype=np.int64)
            counts = np.asarray(data["parent_counts"], dtype=np.int64)
            has_labels = "parent_labels" in data
        if len(offsets) != len(target_ids) + 1 or offsets[0] != 0:
            raise ValueError("invalid target offsets in embedding index")
        if offsets[-1] != len(parent_ids):
            raise ValueError("embedding index offsets do not cover parent IDs")
        if not (len(parent_ids) == len(starts) == len(counts)):
            raise ValueError("embedding index parent arrays have inconsistent lengths")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("embedding index contains duplicate target IDs")

        self.meta = {}
        self.parent_ids = {}
        for index, target_id in enumerate(target_ids):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            self.meta[target_id] = {
                "index": index,
                "parent_begin": begin,
                "parent_end": end,
                "parent_starts": starts[begin:end].copy(),
                "parent_counts": counts[begin:end].copy(),
            }
            self.parent_ids[target_id] = parent_ids[begin:end].copy()
        self.has_labels = has_labels
        self.parent_count = len(parent_ids)
        self._parent_labels = None

    @property
    def target_ids(self) -> list[str]:
        return sorted(self.meta)

    def get_parent_labels(self, target_id: str) -> np.ndarray:
        """Read complete parent labels for benchmark episode construction."""
        if not self.has_labels:
            raise ValueError(f"target {target_id} does not contain activity labels")
        meta = self.meta[target_id]
        if self._parent_labels is None:
            with np.load(self.index_path, allow_pickle=False) as data:
                labels = np.asarray(data["parent_labels"], dtype=np.int8)
            if len(labels) != self.parent_count:
                raise ValueError("activity labels do not match parent IDs")
            if not set(map(int, np.unique(labels))) <= {0, 1}:
                raise ValueError("activity labels must be binary")
            self._parent_labels = labels
        return self._parent_labels[meta["parent_begin"] : meta["parent_end"]].copy()

    def get(
        self,
        target_id: str,
        device: torch.device,
        projection: torch.Tensor | None = None,
        include_labels: bool = False,
        retain_raw: bool = False,
    ) -> TargetData:
        meta = self.meta[target_id]
        counts = meta["parent_counts"].copy()
        raw = self.store.load_ranges(meta["parent_starts"], counts, device)
        parent_starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
        embeddings = (
            external_whiten(raw, projection, counts)
            if projection is not None
            else F.normalize(raw, dim=1)
        )
        labels = None
        if include_labels:
            labels = self.get_parent_labels(target_id)
        return TargetData(
            target_id=target_id,
            embeddings=embeddings,
            parent_ids=self.parent_ids[target_id].copy(),
            parent_starts=parent_starts,
            parent_counts=counts,
            parent_labels=labels,
            raw_embeddings=raw if retain_raw else None,
            projection=projection if retain_raw else None,
        )

    def get_subset(
        self,
        target_id: str,
        selected_parent_ids: set[str],
        device: torch.device,
        projection: torch.Tensor | None = None,
        include_labels: bool = False,
        retain_raw: bool = False,
    ) -> TargetData:
        """Load selected parents directly from a shared embedding store."""
        selected = set(map(str, selected_parent_ids))
        parent_ids = self.parent_ids[target_id].astype(str)
        mask = np.asarray([parent in selected for parent in parent_ids])
        missing = selected - set(parent_ids[mask])
        if missing:
            raise KeyError(f"parent IDs absent from {target_id}: {sorted(missing)}")
        if not mask.any():
            raise ValueError("selected parent set is empty")
        meta = self.meta[target_id]
        counts = meta["parent_counts"][mask].copy()
        raw = self.store.load_ranges(meta["parent_starts"][mask], counts, device)
        starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
        embeddings = (
            external_whiten(raw, projection, counts)
            if projection is not None
            else F.normalize(raw, dim=1)
        )
        labels = self.get_parent_labels(target_id)[mask] if include_labels else None
        return TargetData(
            target_id=target_id,
            embeddings=embeddings,
            parent_ids=self.parent_ids[target_id][mask].copy(),
            parent_starts=starts,
            parent_counts=counts,
            parent_labels=labels,
            raw_embeddings=raw if retain_raw else None,
            projection=projection if retain_raw else None,
        )


def subset_target(target: TargetData, parent_mask: np.ndarray) -> TargetData:
    """Build an episode-local target and reapply centering before whitening."""
    parent_mask = np.asarray(parent_mask, dtype=bool)
    if parent_mask.shape != (len(target.parent_ids),):
        raise ValueError("parent mask has the wrong shape")
    if not parent_mask.any():
        raise ValueError("episode target is empty")

    row_mask = np.repeat(parent_mask, target.parent_counts)
    row_mask_t = torch.as_tensor(row_mask, device=target.embeddings.device)
    counts = target.parent_counts[parent_mask].copy()
    starts = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
    raw = (
        target.raw_embeddings[row_mask_t] if target.raw_embeddings is not None else None
    )
    if raw is not None:
        embeddings = (
            external_whiten(raw, target.projection, counts)
            if target.projection is not None
            else F.normalize(raw, dim=1)
        )
    else:
        embeddings = target.embeddings[row_mask_t]
    labels = (
        target.parent_labels[parent_mask].copy()
        if target.parent_labels is not None
        else None
    )
    return TargetData(
        target_id=target.target_id,
        embeddings=embeddings,
        parent_ids=target.parent_ids[parent_mask].copy(),
        parent_starts=starts,
        parent_counts=counts,
        parent_labels=labels,
        raw_embeddings=raw,
        projection=target.projection,
    )
