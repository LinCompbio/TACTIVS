#!/usr/bin/env python3
"""Build one Zenodo-ready bundle for LIT-PCBA, RandomDecoy, and TrueDecoy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactivs.bundle import BUNDLE_SCHEMA
from tactivs.provenance import cache_sha256, sha256


def link_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def link_embeddings(source: Path, destination: Path, mode: str) -> list[Path]:
    shards = sorted(source.glob("embeddings_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no embedding shards in {source}")
    output = []
    for shard in shards:
        target = destination / shard.name
        link_file(shard, target, mode)
        output.append(target)
    return output


def write_active_manifest(
    source: Path, destination: Path, allowed: dict[str, set[str]]
) -> int:
    rows = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "parent_molecule_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"active manifest missing columns: {sorted(missing)}")
        for row in reader:
            target_id = row["target_id"]
            parent_id = row["parent_molecule_id"]
            if parent_id in allowed.get(target_id, set()):
                rows.append(
                    {
                        "target_id": target_id,
                        "parent_molecule_id": parent_id,
                        "canonical_smiles": row["canonical_smiles"],
                    }
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(required))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def legacy_index(record: dict, destination: Path, mode: str) -> dict:
    source = Path(record["source"]).resolve()
    with np.load(source / "target_manifest.npz", allow_pickle=False) as data:
        target_ids = np.asarray(data["target_ids"]).astype(str)
        offsets = np.asarray(data["parent_offsets"], dtype=np.int64)
        starts = np.asarray(data["parent_starts"], dtype=np.int64)
        counts = np.asarray(data["parent_counts"], dtype=np.int64)
        labels = np.asarray(data["parent_labels"], dtype=np.int8)
    with np.load(source / "target_parent_ids.npz", allow_pickle=False) as data:
        parent_ids = np.asarray(data["parent_ids"]).astype(str)
    if not (len(parent_ids) == len(starts) == len(counts) == len(labels)):
        raise ValueError(f"inconsistent legacy cache arrays in {source}")
    np.savez_compressed(
        destination / "index.npz",
        target_ids=target_ids,
        parent_offsets=offsets,
        parent_ids=parent_ids,
        parent_starts=starts,
        parent_counts=counts,
        parent_labels=labels,
    )
    allowed = {}
    for index, target_id in enumerate(target_ids):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        allowed[target_id] = set(parent_ids[begin:end][labels[begin:end] == 1])
    positives = write_active_manifest(
        Path(record["active_manifest"]), destination / "active_manifest.csv", allowed
    )
    link_embeddings(source, destination / "embeddings", mode)
    link_file(Path(record["whitener"]), destination / "whitener.npz", mode)
    return {
        "targets": len(target_ids),
        "parents": len(parent_ids),
        "positives": positives,
        "conformers": int(counts.sum()),
    }


def shared_bank_index(record: dict, destination: Path, mode: str) -> dict:
    source = Path(record["source"]).resolve()
    embedding_root = source / "embeddings"
    shards = sorted(embedding_root.glob("embeddings_*.npy"))
    shard_sizes = np.asarray(
        [np.load(path, mmap_mode="r").shape[0] for path in shards], dtype=np.int64
    )
    shard_offsets = np.concatenate(([0], np.cumsum(shard_sizes[:-1])))
    with (source / "molecules.csv").open() as handle:
        molecule_count = sum(1 for _ in handle) - 1
    starts = np.full(molecule_count, -1, dtype=np.int64)
    counts = np.zeros(molecule_count, dtype=np.int64)
    for path in sorted(embedding_root.glob("index_*.npz")):
        shard = int(path.stem.split("_")[-1])
        with np.load(path, allow_pickle=False) as data:
            molecules = np.asarray(data["molecule_indices"], dtype=np.int64)
            if np.any(starts[molecules] >= 0):
                raise ValueError("shared-bank molecule occurs in multiple shards")
            starts[molecules] = shard_offsets[shard] + np.asarray(
                data["starts"], dtype=np.int64
            )
            counts[molecules] = np.asarray(data["counts"], dtype=np.int64)

    memberships: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    with (source / "target_memberships.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            molecule = int(row["molecule_index"])
            if starts[molecule] >= 0:
                memberships[row["target_id"]].append(
                    (molecule, row["parent_molecule_id"], int(row["label"]))
                )

    target_ids = np.asarray(sorted(memberships))
    offsets = [0]
    output_ids = []
    output_starts = []
    output_counts = []
    output_labels = []
    allowed: dict[str, set[str]] = {}
    for target_id in target_ids:
        rows = sorted(memberships[str(target_id)], key=lambda row: starts[row[0]])
        target_actives = set()
        for molecule, parent_id, label in rows:
            output_ids.append(parent_id)
            output_starts.append(starts[molecule])
            output_counts.append(counts[molecule])
            output_labels.append(label)
            if label == 1:
                target_actives.add(parent_id)
        allowed[str(target_id)] = target_actives
        offsets.append(len(output_ids))
    parent_counts = np.asarray(output_counts, dtype=np.int64)
    np.savez_compressed(
        destination / "index.npz",
        target_ids=target_ids,
        parent_offsets=np.asarray(offsets, dtype=np.int64),
        parent_ids=np.asarray(output_ids),
        parent_starts=np.asarray(output_starts, dtype=np.int64),
        parent_counts=parent_counts,
        parent_labels=np.asarray(output_labels, dtype=np.int8),
    )
    positives = write_active_manifest(
        Path(record["active_manifest"]), destination / "active_manifest.csv", allowed
    )
    link_embeddings(embedding_root, destination / "embeddings", mode)
    link_file(Path(record["whitener"]), destination / "whitener.npz", mode)
    return {
        "targets": len(target_ids),
        "parents": len(output_ids),
        "positives": positives,
        "conformers": int(parent_counts.sum()),
        "unique_molecules": int((starts >= 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    expected_names = {"litpcba", "randomdecoy", "truedecoy"}
    records = recipe.get("datasets", [])
    if {record.get("name") for record in records} != expected_names:
        raise ValueError(f"bundle recipe must contain exactly {sorted(expected_names)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty bundle: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": BUNDLE_SCHEMA, "datasets": []}
    for record in records:
        name = record["name"]
        destination = args.output / "datasets" / name
        destination.mkdir(parents=True, exist_ok=True)
        if record["layout"] == "legacy_cache":
            summary = legacy_index(record, destination, args.link_mode)
        elif record["layout"] == "shared_bank":
            summary = shared_bank_index(record, destination, args.link_mode)
        else:
            raise ValueError(f"unknown source layout for {name}: {record['layout']}")
        manifest["datasets"].append(
            {
                "name": name,
                "path": f"datasets/{name}",
                "protocols": record.get("protocols", ["fixed_random"]),
                **summary,
                "cache_sha256": cache_sha256(destination),
                "index_sha256": sha256(destination / "index.npz"),
                "whitener_sha256": sha256(destination / "whitener.npz"),
                "active_manifest_sha256": sha256(destination / "active_manifest.csv"),
            }
        )
        print(f"prepared {name}: {summary}", flush=True)
    (args.output / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.output / 'bundle.json'}")


if __name__ == "__main__":
    main()
