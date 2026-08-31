#!/usr/bin/env python3
"""Build the compact target manifests consumed by TACTIVS from metadata.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    metadata = args.cache / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(metadata)

    target_ids: list[str] = []
    target_starts: list[int] = []
    target_ends: list[int] = []
    parent_offsets = [0]
    parent_starts: list[int] = []
    parent_counts: list[int] = []
    parent_labels: list[int] = []
    parent_ids: list[str] = []

    current_target: str | None = None
    current_parent: str | None = None
    current_target_start = 0
    current_parent_start = 0
    current_label = -1
    expected_index = 0

    def finish_parent(end: int) -> None:
        parent_starts.append(current_parent_start)
        parent_counts.append(end - current_parent_start)
        parent_labels.append(current_label)
        parent_ids.append(str(current_parent))

    def finish_target(end: int) -> None:
        target_ids.append(str(current_target))
        target_starts.append(current_target_start)
        target_ends.append(end)
        parent_offsets.append(len(parent_starts))

    with metadata.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"embedding_index", "target_id", "parent_molecule_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata missing columns: {sorted(missing)}")
        for row in reader:
            index = int(row["embedding_index"])
            if index != expected_index:
                raise ValueError(
                    f"non-contiguous embedding_index: expected {expected_index}, got {index}"
                )
            expected_index += 1
            target = row["target_id"]
            parent = row["parent_molecule_id"]
            label = int(row.get("label", -1) or -1)

            if current_target is None:
                current_target = target
                current_parent = parent
                current_target_start = current_parent_start = index
                current_label = label
            elif target != current_target:
                finish_parent(index)
                finish_target(index)
                current_target = target
                current_parent = parent
                current_target_start = current_parent_start = index
                current_label = label
            elif parent != current_parent:
                finish_parent(index)
                current_parent = parent
                current_parent_start = index
                current_label = label
            elif label != current_label:
                raise ValueError(f"label changes within {target}/{parent}")

    if current_target is None:
        raise ValueError(f"empty metadata: {metadata}")
    finish_parent(expected_index)
    finish_target(expected_index)

    stat = metadata.stat()
    common = {
        "metadata_size": np.asarray(stat.st_size, dtype=np.int64),
        "metadata_mtime_ns": np.asarray(stat.st_mtime_ns, dtype=np.int64),
    }
    np.savez_compressed(
        args.cache / "target_manifest.npz",
        target_ids=np.asarray(target_ids),
        target_starts=np.asarray(target_starts, dtype=np.int64),
        target_ends=np.asarray(target_ends, dtype=np.int64),
        parent_offsets=np.asarray(parent_offsets, dtype=np.int64),
        parent_starts=np.asarray(parent_starts, dtype=np.int64),
        parent_counts=np.asarray(parent_counts, dtype=np.int64),
        parent_labels=np.asarray(parent_labels, dtype=np.int8),
        **common,
    )
    np.savez_compressed(
        args.cache / "target_parent_ids.npz",
        target_ids=np.asarray(target_ids),
        parent_offsets=np.asarray(parent_offsets, dtype=np.int64),
        parent_ids=np.asarray(parent_ids),
        **common,
    )
    print(
        f"prepared {len(target_ids)} targets, {len(parent_ids)} molecules, "
        f"{expected_index} conformer rows"
    )


if __name__ == "__main__":
    main()
