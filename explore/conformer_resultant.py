#!/usr/bin/env python3
"""Explore conformer directional agreement discarded by graph normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tactivs.cache import EmbeddingCache, load_projection


def resultant_lengths(target) -> np.ndarray:
    """Compute ||mean conformer embedding|| before graph-node normalization."""
    counts = torch.as_tensor(
        target.parent_counts,
        dtype=target.embeddings.dtype,
        device=target.embeddings.device,
    )
    codes = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64, device=counts.device),
        counts.to(torch.int64),
    )
    sums = torch.zeros(
        (len(counts), target.embeddings.shape[1]),
        dtype=target.embeddings.dtype,
        device=target.embeddings.device,
    )
    sums.index_add_(0, codes, target.embeddings)
    return (sums / counts[:, None]).norm(dim=1).cpu().numpy()


def molecule_rows(spec: dict, device: torch.device, max_targets: int | None):
    cache = EmbeddingCache(Path(spec["cache"]))
    projection = load_projection(Path(spec["whitener"]), device)
    target_ids = cache.target_ids[:max_targets]
    for index, target_id in enumerate(target_ids, 1):
        target = cache.get(
            target_id, device, projection=projection, include_labels=True
        )
        resultant = resultant_lengths(target)
        counts = target.parent_counts.astype(np.int64)
        pairwise_cosine = np.full(len(counts), np.nan, dtype=np.float64)
        multi = counts > 1
        pairwise_cosine[multi] = (
            counts[multi] * resultant[multi] ** 2 - 1.0
        ) / (counts[multi] - 1.0)
        yield pd.DataFrame(
            {
                "dataset": spec["name"],
                "target_id": target_id,
                "parent_molecule_id": target.parent_ids.astype(str),
                "label": target.parent_labels.astype(np.int8),
                "conformer_count": counts,
                "resultant_length": resultant,
                "mean_pairwise_cosine": pairwise_cosine,
            }
        )
        if index % 20 == 0 or index == len(target_ids):
            print(f"[{spec['name']} {index}/{len(target_ids)}]", flush=True)


def describe_group(frame: pd.DataFrame) -> pd.Series:
    values = frame["resultant_length"]
    return pd.Series(
        {
            "molecules": len(frame),
            "targets": frame["target_id"].nunique(),
            "mean": values.mean(),
            "std": values.std(),
            "q05": values.quantile(0.05),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "q95": values.quantile(0.95),
            "fraction_below_0.5": (values < 0.5).mean(),
            "fraction_below_0.75": (values < 0.75).mean(),
        }
    )


def write_summaries(molecules: pd.DataFrame, output_dir: Path) -> None:
    annotated = molecules.assign(
        conformer_group=np.where(
            molecules["conformer_count"].eq(1), "single", "multiple"
        ),
        label_group=np.where(molecules["label"].eq(1), "active", "inactive"),
    )
    summaries = []
    grouping_sets = (
        ("dataset",),
        ("dataset", "conformer_group"),
        ("dataset", "conformer_group", "label_group"),
    )
    for columns in grouping_sets:
        summary = annotated.groupby(list(columns), as_index=False).apply(
            describe_group, include_groups=False
        )
        summary.insert(1, "stratification", "+".join(columns[1:]) or "all")
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        output_dir / "summary.csv", index=False
    )
    annotated.groupby(
        ["dataset", "conformer_count"], as_index=False
    ).apply(describe_group, include_groups=False).to_csv(
        output_dir / "by_conformer_count.csv", index=False
    )


def spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank().corr(right.rank()))


def score_associations(
    score_files: list[Path], molecules: pd.DataFrame
) -> pd.DataFrame:
    lookup = molecules.set_index(
        ["dataset", "target_id", "parent_molecule_id"]
    )["resultant_length"]
    rows = []
    required = [
        "dataset",
        "config",
        "target_id",
        "protocol",
        "seed",
        "K",
        "parent_molecule_id",
        "role",
        "score",
        "direct",
        "graph",
    ]
    for path in score_files:
        scores = pd.read_csv(path, usecols=required)
        scores = scores.loc[
            scores["protocol"].eq("fixed_random") & scores["role"].eq("candidate")
        ].copy()
        keys = pd.MultiIndex.from_frame(
            scores[["dataset", "target_id", "parent_molecule_id"]].astype(str)
        )
        scores["resultant_length"] = lookup.reindex(keys).to_numpy()
        if scores["resultant_length"].isna().any():
            raise ValueError(f"{path}: score IDs do not match resultant-length rows")
        for key, episode in scores.groupby(
            ["dataset", "config", "target_id", "seed", "K"], sort=False
        ):
            rows.append(
                {
                    "source": str(path),
                    "dataset": key[0],
                    "config": key[1],
                    "target_id": key[2],
                    "seed": key[3],
                    "K": key[4],
                    "candidates": len(episode),
                    "rho_R_graph": spearman(
                        episode["resultant_length"], episode["graph"]
                    ),
                    "rho_R_direct": spearman(
                        episode["resultant_length"], episode["direct"]
                    ),
                    "rho_R_final": spearman(
                        episode["resultant_length"], episode["score"]
                    ),
                    "rho_R_graph_minus_direct": spearman(
                        episode["resultant_length"],
                        episode["graph"].rank(pct=True)
                        - episode["direct"].rank(pct=True),
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--score-files", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_targets is not None and args.max_targets < 1:
        parser.error("--max-targets must be positive")

    specs = json.loads(args.data_config.read_text())["datasets"]
    if args.datasets:
        requested = set(args.datasets)
        available = {spec["name"] for spec in specs}
        if missing := requested - available:
            parser.error(f"unknown datasets: {', '.join(sorted(missing))}")
        specs = [spec for spec in specs if spec["name"] in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frames = [
        frame
        for spec in specs
        for frame in molecule_rows(spec, device, args.max_targets)
    ]
    molecules = pd.concat(frames, ignore_index=True)
    molecules.to_csv(
        args.output_dir / "molecule_resultants.csv.gz", index=False, compression="gzip"
    )
    write_summaries(molecules, args.output_dir)
    if args.score_files:
        associations = score_associations(args.score_files, molecules)
        associations.to_csv(args.output_dir / "score_associations.csv", index=False)
    print(pd.read_csv(args.output_dir / "summary.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
