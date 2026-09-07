"""Command-line entry points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from .cache import EmbeddingCache, load_projection
from .core import score_library
from .encoding import build_embedding_cache
from .provenance import configure_determinism
from .reference_pool import (
    EPTConformerEncoder,
    ReferencePool,
    build_reference_pool,
    read_molecules,
)
from .whitening import fit_whitener


def infer_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python tactivs.py infer",
        description="Rank one unlabeled molecular library from K positive references.",
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
    reference.add_argument(
        "--reference-ids",
        nargs="+",
        metavar="ID",
        help="Use known active molecule IDs already present in the cache.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

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
    elif args.reference_pool is not None:
        target = cache.get(
            args.target,
            device,
            projection=projection,
            include_labels=False,
            retain_raw=True,
        )
        pool = ReferencePool.load(args.reference_pool, device)
    else:
        target = cache.get(
            args.target,
            device,
            projection=projection,
            include_labels=False,
        )
        pool = ReferencePool.from_cached_ids(
            args.reference_ids, source="cached_known_actives"
        )
    target, references = pool.materialize(target, projection)
    result = score_library(target, references, params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["parent_molecule_id", "score", "similarity", "direct", "graph"]
        )
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
                "references": references,
                "library_molecules": len(target.parent_ids),
                "ranked_molecules": len(result.parent_ids),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def build_reference_pool_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python tactivs.py build-reference-pool",
        description="Generate, encode, and cache an external TACTIVS reference pool.",
    )
    parser.add_argument("--molecules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    _add_ept_arguments(parser)
    args = parser.parse_args(argv)

    configure_determinism(args.seed)
    device = torch.device(args.device)
    records = read_molecules(args.molecules)
    encoder = _load_ept_encoder(args, device)
    pool = build_reference_pool(
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


def build_cache_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python tactivs.py build-cache",
        description="Generate a TACTIVS EPT cache for a molecular library.",
    )
    parser.add_argument("--molecules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="library")
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--molecule-batch-size", type=int, default=256)
    _add_ept_arguments(parser)
    args = parser.parse_args(argv)

    configure_determinism(args.seed)
    device = torch.device(args.device)
    records = read_molecules(args.molecules)
    encoder = _load_ept_encoder(args, device)
    info = build_embedding_cache(
        records,
        encoder,
        args.output,
        target_id=args.target,
        conformers_per_molecule=args.conformers,
        seed=args.seed,
        molecule_batch_size=args.molecule_batch_size,
    )
    print(json.dumps({"cache": str(args.output), **info}, indent=2))


def fit_whitener_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python tactivs.py fit-whitener",
        description="Fit a covariance-only whitener from an EPT cache.",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shrinkage", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--molecule-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)

    info = fit_whitener(
        args.cache,
        args.output,
        shrinkage=args.shrinkage,
        eps=args.eps,
        molecule_batch_size=args.molecule_batch_size,
    )
    print(json.dumps({"whitener": str(args.output), **info}, indent=2))


def _add_ept_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )


def _load_ept_encoder(args, device: torch.device) -> EPTConformerEncoder:
    return EPTConformerEncoder(
        args.encoder_checkpoint,
        device,
        batch_size=args.batch_size,
    )


def _benchmark_main(argv: Sequence[str] | None = None) -> None:
    from .benchmark import main as benchmark_main

    benchmark_main(argv)


COMMANDS: dict[str, tuple[str, Callable[[Sequence[str] | None], None]]] = {
    "infer": (
        "rank a molecular library from known active references",
        infer_main,
    ),
    "build-cache": (
        "encode a molecular library into an EPT cache",
        build_cache_main,
    ),
    "build-reference-pool": (
        "encode known active molecules into a reference pool",
        build_reference_pool_main,
    ),
    "fit-whitener": (
        "fit a covariance-only whitening projection",
        fit_whitener_main,
    ),
    "benchmark": (
        "reproduce a published benchmark episode",
        _benchmark_main,
    ),
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the source-only command-line interface."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python tactivs.py",
        description="TACTIVS molecular-library ranking tools.",
        epilog="\n".join(
            f"  {name:<22} {description}" for name, (description, _) in COMMANDS.items()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="command to run",
    )
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command][1](arguments[1:])
