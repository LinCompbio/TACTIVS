"""Readers for the sharded EPT embedding cache used by the experiments."""

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


def molecule_balanced_mean(
    raw: torch.Tensor, parent_counts: np.ndarray
) -> torch.Tensor:
    """Return the mean after assigning equal weight to every molecule."""
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
    return centers.mean(0)


def external_whiten(
    raw: torch.Tensor, projection: torch.Tensor, parent_counts: np.ndarray
) -> torch.Tensor:
    """Apply frozen covariance whitening with an unlabeled target-local mean."""
    target_mean = molecule_balanced_mean(raw, parent_counts)
    return F.normalize((raw - target_mean) @ projection, dim=1)


class EmbeddingStore:
    def __init__(self, cache: Path):
        embedding_root = (
            cache / "embeddings" if (cache / "embeddings").is_dir() else cache
        )
        paths = sorted(embedding_root.glob("embeddings_*.npy"))
        if not paths:
            raise FileNotFoundError(f"no embeddings_*.npy shards in {cache}")
        self.arrays = [np.load(path, mmap_mode="r") for path in paths]
        dimensions = {array.shape[1] for array in self.arrays}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent embedding dimensions: {dimensions}")
        self.dim = dimensions.pop()
        self.starts = np.cumsum([0] + [array.shape[0] for array in self.arrays[:-1]])
        self.ends = np.cumsum([array.shape[0] for array in self.arrays])

    def load_indexed(
        self, starts: np.ndarray, counts: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Load arbitrary contiguous molecule segments while preserving their order."""
        starts = np.asarray(starts, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.int64)
        total = int(counts.sum())
        output = torch.empty((total, self.dim), dtype=torch.float32, device=device)
        destinations = np.arange(total, dtype=np.int64)
        molecule_offsets = np.repeat(np.cumsum(counts) - counts, counts)
        global_rows = np.repeat(starts, counts) + destinations - molecule_offsets
        shard_codes = np.searchsorted(self.ends, global_rows, side="right")
        if len(shard_codes) and int(shard_codes.max()) >= len(self.arrays):
            raise IndexError("indexed embedding row exceeds the available shards")
        for shard in np.unique(shard_codes):
            mask = shard_codes == shard
            rows = global_rows[mask] - int(self.starts[shard])
            block = np.asarray(self.arrays[int(shard)][rows], dtype=np.float32)
            output[torch.from_numpy(destinations[mask]).to(device)] = torch.from_numpy(
                block
            ).to(device)
        return output

    def load(self, start: int, end: int, device: torch.device) -> torch.Tensor:
        output = torch.empty(
            (end - start, self.dim), dtype=torch.float32, device=device
        )
        cursor = start
        while cursor < end:
            shard = int(np.searchsorted(self.ends, cursor, side="right"))
            shard_start = int(self.starts[shard])
            take_end = min(end, int(self.ends[shard]), cursor + 20000)
            block = np.array(
                self.arrays[shard][cursor - shard_start : take_end - shard_start],
                copy=True,
            )
            output[cursor - start : take_end - start] = torch.from_numpy(block).to(
                device=device, dtype=torch.float32
            )
            cursor = take_end
        return output


class EmbeddingCache:
    """Load targets without touching labels unless ``include_labels=True``."""

    def __init__(self, cache: Path):
        self.cache = Path(cache)
        self.store = EmbeddingStore(self.cache)
        bundle_index = self.cache / "index.npz"
        if bundle_index.exists():
            self._load_bundle_index(bundle_index)
            return
        self.manifest_path = self.cache / "target_manifest.npz"
        parent_path = self.cache / "target_parent_ids.npz"
        if not self.manifest_path.exists() or not parent_path.exists():
            raise FileNotFoundError(
                "cache requires target_manifest.npz and target_parent_ids.npz"
            )

        with np.load(self.manifest_path, allow_pickle=False) as data:
            target_ids = list(map(str, data["target_ids"]))
            offsets = data["parent_offsets"].astype(np.int64)
            starts = data["parent_starts"].astype(np.int64)
            counts = data["parent_counts"].astype(np.int64)
            target_starts = data["target_starts"].astype(np.int64)
            target_ends = data["target_ends"].astype(np.int64)
        self.meta = {}
        for index, target_id in enumerate(target_ids):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            row_start = int(target_starts[index])
            self.meta[target_id] = {
                "index": index,
                "parent_begin": begin,
                "parent_end": end,
                "row_start": row_start,
                "row_end": int(target_ends[index]),
                "parent_starts": starts[begin:end] - row_start,
                "parent_counts": counts[begin:end],
            }

        with np.load(parent_path, allow_pickle=False) as data:
            parent_target_ids = list(map(str, data["target_ids"]))
            parent_offsets = data["parent_offsets"].astype(np.int64)
            parent_ids = data["parent_ids"]
        if parent_target_ids != target_ids:
            raise ValueError("target IDs differ between cache manifests")
        self.parent_ids = {
            target_id: parent_ids[
                int(parent_offsets[i]) : int(parent_offsets[i + 1])
            ].copy()
            for i, target_id in enumerate(target_ids)
        }
        self._parent_labels: np.ndarray | None = None
        self._indexed = False

    def _load_bundle_index(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "target_ids",
                "parent_offsets",
                "parent_ids",
                "parent_starts",
                "parent_counts",
                "parent_labels",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"bundle index missing arrays: {sorted(missing)}")
            target_ids = list(map(str, data["target_ids"]))
            offsets = np.asarray(data["parent_offsets"], dtype=np.int64)
            parent_ids = np.asarray(data["parent_ids"]).astype(str)
            starts = np.asarray(data["parent_starts"], dtype=np.int64)
            counts = np.asarray(data["parent_counts"], dtype=np.int64)
            labels = np.asarray(data["parent_labels"], dtype=np.int8)
        if len(offsets) != len(target_ids) + 1 or offsets[0] != 0:
            raise ValueError("bundle parent offsets do not match target IDs")
        if offsets[-1] != len(parent_ids):
            raise ValueError("bundle parent offsets do not cover every parent")
        if not (len(parent_ids) == len(starts) == len(counts) == len(labels)):
            raise ValueError("bundle parent arrays have inconsistent lengths")
        if np.any(counts < 1) or np.any(starts < 0):
            raise ValueError("bundle embedding starts and counts must be positive")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("bundle target IDs must be unique")
        self.manifest_path = path
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
        self._parent_labels = labels
        self._indexed = True

    @property
    def target_ids(self) -> list[str]:
        return sorted(self.meta)

    def get_parent_labels(self, target_id: str) -> np.ndarray:
        """Read complete parent labels for benchmark episode construction."""
        meta = self.meta[target_id]
        if self._parent_labels is None:
            with np.load(self.manifest_path, allow_pickle=False) as data:
                self._parent_labels = data["parent_labels"].astype(np.int8)
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
        raw = (
            self.store.load_indexed(
                meta["parent_starts"], meta["parent_counts"], device
            )
            if self._indexed
            else self.store.load(meta["row_start"], meta["row_end"], device)
        )
        counts = meta["parent_counts"].copy()
        embeddings = (
            external_whiten(raw, projection, counts)
            if projection is not None
            else F.normalize(raw, dim=1)
        )
        labels = None
        if include_labels:
            labels = self.get_parent_labels(target_id)
        local_starts = (
            np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int64)
            if self._indexed
            else meta["parent_starts"].copy()
        )
        return TargetData(
            target_id=target_id,
            embeddings=embeddings,
            parent_ids=self.parent_ids[target_id].copy(),
            parent_starts=local_starts,
            parent_counts=counts,
            parent_labels=labels,
            raw_embeddings=raw if retain_raw else None,
            projection=projection if retain_raw else None,
        )

    def get_raw(self, target_id: str, device: torch.device) -> torch.Tensor:
        """Load unnormalized conformer embeddings for an individual target."""
        meta = self.meta[target_id]
        if self._indexed:
            return self.store.load_indexed(
                meta["parent_starts"], meta["parent_counts"], device
            )
        return self.store.load(meta["row_start"], meta["row_end"], device)


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
