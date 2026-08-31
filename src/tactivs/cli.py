"""Command-line entry points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .cache import EmbeddingCache, load_projection
from .core import score_library
from .provenance import configure_determinism
from .reference_pool import (
    EPTConformerEncoder,
    ReferencePool,
    build_uploaded_pool,
    read_uploaded_molecules,
)


def infer_main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank one unlabeled molecular library from K positive references."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--whitener", type=Path, required=True)
    parser.add_argument("--target", required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--sample-labeled",
        type=int,
        metavar="K",
        help="Sample K references from cached positive labels.",
    )
    reference.add_argument(
        "--reference-pool",
        type=Path,
        help="Use an encoded external reference-pool artifact.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_determinism()
    device = torch.device(args.device)
    params = json.loads(args.theta.read_text())
    cache = EmbeddingCache(args.cache)
    projection = load_projection(args.whitener, device)
    if args.sample_labeled is not None:
        target = cache.get(
            args.target,
            device,
            projection=projection,
            include_labels=True,
        )
        pool = ReferencePool.sample_labeled(target, args.sample_labeled, args.seed)
    else:
        target = cache.get(
            args.target,
            device,
            projection=projection,
            include_labels=False,
            retain_raw=True,
        )
        pool = ReferencePool.load(args.reference_pool, device)
    target, references = pool.materialize(target, projection)
    result = score_library(target, references, params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parent_molecule_id", "score", "similarity", "direct", "graph"])
        for row in zip(
            result.parent_ids,
            result.scores,
            result.similarity,
            result.direct,
            result.graph,
        ):
            writer.writerow(row)
    print(
        json.dumps(
            {
                "target": args.target,
                "reference_mode": pool.source,
                "references": len(references),
                "ranked_molecules": len(result.parent_ids),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def build_reference_pool_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, encode, and cache an external TACTIVS reference pool."
    )
    parser.add_argument("--molecules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ept-root", type=Path, required=True)
    parser.add_argument(
        "--ept-ranking-root",
        type=Path,
        required=True,
        help="Path to the upstream EPT ranking source tree containing models/ and data/.",
    )
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_determinism(args.seed)
    device = torch.device(args.device)
    records = read_uploaded_molecules(args.molecules)
    encoder = EPTConformerEncoder(
        args.ept_root,
        args.ept_ranking_root,
        args.encoder_checkpoint,
        device,
        batch_size=args.batch_size,
    )
    pool = build_uploaded_pool(
        records,
        encoder,
        conformers_per_molecule=args.conformers,
        seed=args.seed,
    )
    pool.save(args.output)
    print(
        json.dumps(
            {
                "reference_mode": pool.source,
                "reference_molecules": len(pool.parent_ids),
                "reference_conformers": int(pool.parent_counts.sum()),
                "output": str(args.output),
            },
            indent=2,
        )
    )
