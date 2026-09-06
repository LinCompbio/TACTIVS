"""Build EPT embedding caches for molecular libraries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reference_pool import ConformerEncoder, MoleculeRecord, generate_conformers


def build_embedding_cache(
    records: list[MoleculeRecord],
    encoder: ConformerEncoder,
    output: Path,
    target_id: str,
    conformers_per_molecule: int = 10,
    seed: int = 0,
    molecule_batch_size: int = 256,
) -> dict[str, int]:
    if not records:
        raise ValueError("molecule input is empty")
    if molecule_batch_size < 1:
        raise ValueError("molecule_batch_size must be positive")

    output = Path(output)
    embedding_dir = output / "embeddings"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    embedding_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    row_count = 0
    shard_count = 0
    for begin in range(0, len(records), molecule_batch_size):
        batch = records[begin : begin + molecule_batch_size]
        conformers, batch_counts = generate_conformers(
            batch, conformers_per_molecule, seed
        )
        embeddings = encoder.encode(conformers).numpy().astype(np.float32, copy=False)
        if len(embeddings) != int(batch_counts.sum()):
            raise ValueError("encoder output does not match the generated conformers")
        np.save(embedding_dir / f"embeddings_{shard_count:06d}.npy", embeddings)
        counts.extend(batch_counts.tolist())
        row_count += len(embeddings)
        shard_count += 1

    parent_counts = np.asarray(counts, dtype=np.int64)
    parent_starts = np.concatenate(([0], np.cumsum(parent_counts[:-1]))).astype(
        np.int64
    )
    np.savez_compressed(
        output / "index.npz",
        target_ids=np.asarray([target_id]),
        parent_offsets=np.asarray([0, len(records)], dtype=np.int64),
        parent_ids=np.asarray([record.parent_id for record in records]),
        parent_starts=parent_starts,
        parent_counts=parent_counts,
    )
    return {
        "molecules": len(records),
        "conformers": row_count,
        "shards": shard_count,
    }
