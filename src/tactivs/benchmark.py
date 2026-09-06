"""Reproduce published TACTIVS benchmark episodes from the released data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import EmbeddingCache, load_projection, subset_target
from .core import score_library
from .metrics import screening_metrics
from .provenance import configure_determinism
from .reference_pool import ReferencePool

BENCHMARK_SPLITS = {
    "truedecoy": {"molecule-random", "series-disjoint"},
    "randomdecoy": {"molecule-random", "series-disjoint"},
    "litpcba": {"molecule-random", "ave"},
}
METRICS = (
    "auc_roc",
    "auc_pr",
    "ef_0.5%",
    "ef_1%",
    "ef_5%",
    "bedroc_20",
    "bedroc_80_5",
    "bedroc_85",
    "bayes_ef_0.5%",
    "bayes_ef_1%",
    "bayes_ef_5%",
)
def read_episode_rows(path: Path, seed: int, ks: set[int]) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "target_id",
            "seed",
            "K",
            "support_parent_ids",
            "candidate_count",
            "positive_count",
            "negative_count",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["seed"]) != seed or int(row["K"]) not in ks:
                continue
            references = list(map(str, json.loads(row["support_parent_ids"])))
            k = int(row["K"])
            if len(references) != k or len(set(references)) != k:
                raise ValueError(
                    f"invalid references for {row['target_id']}, seed={seed}, K={k}"
                )
            rows.append(
                {
                    "target_id": row["target_id"],
                    "seed": seed,
                    "K": k,
                    "references": references,
                    "expected_candidate_count": int(row["candidate_count"]),
                    "expected_positive_count": int(row["positive_count"]),
                    "expected_negative_count": int(row["negative_count"]),
                    "expected_excluded_count": int(
                        row.get("series_excluded_count", 0)
                    ),
                }
            )
    if not rows:
        raise ValueError(f"no episodes for seed={seed} and K={sorted(ks)} in {path}")
    return rows


def read_series(path: Path) -> dict[str, dict[str, str]]:
    output = defaultdict(dict)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "parent_molecule_id", "series_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            output[row["target_id"]][row["parent_molecule_id"]] = row["series_id"]
    return dict(output)


def read_smi_ids(path: Path) -> set[str]:
    ids = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected SMILES and molecule ID")
            ids.add(fields[-1])
    return ids


def episode_masks(
    parent_ids: np.ndarray,
    references: list[str],
    split: str,
    target_id: str,
    split_root: Path,
    series: dict[str, dict[str, str]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    parents = np.asarray(parent_ids).astype(str)
    parent_set = set(parents)
    reference_set = set(references)
    missing = reference_set - parent_set
    if missing:
        raise KeyError(f"{target_id}: references absent from EPT index: {sorted(missing)}")
    reference_mask = np.asarray([parent in reference_set for parent in parents])

    if split == "molecule-random":
        visible = np.ones(len(parents), dtype=bool)
        ranked = ~reference_mask
    elif split == "series-disjoint":
        if series is None or target_id not in series:
            raise ValueError(f"{target_id}: missing series assignments")
        assignments = series[target_id]
        reference_series = {assignments[parent] for parent in references}
        visible = np.asarray(
            [
                parent in reference_set
                or assignments.get(parent) not in reference_series
                for parent in parents
            ]
        )
        ranked = visible & ~reference_mask
    else:
        target_root = split_root / "official_ave" / target_id
        candidates = read_smi_ids(target_root / "active_V.smi") | read_smi_ids(
            target_root / "inactive_V.smi"
        )
        candidates &= parent_set
        visible = np.asarray(
            [parent in reference_set or parent in candidates for parent in parents]
        )
        ranked = np.asarray([parent in candidates for parent in parents])
    return visible, ranked


def summarize(episodes: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    by_k = episodes.groupby("K")[list(METRICS)].agg(["mean", "std", "count"])
    by_k.columns = [f"{metric}_{stat}" for metric, stat in by_k.columns]
    by_k = by_k.reset_index()
    by_k.to_csv(output_dir / "summary_by_k.csv", index=False)

    target_means = episodes.groupby("target_id")[list(METRICS)].mean()
    overall = {
        metric: {
            "mean": float(target_means[metric].mean()),
            "std": float(target_means[metric].std(ddof=1)),
            "targets": int(target_means[metric].count()),
        }
        for metric in METRICS
    }
    pd.DataFrame(
        [{"metric": metric, **values} for metric, values in overall.items()]
    ).to_csv(output_dir / "summary_overall.csv", index=False)
    return by_k, overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one published TACTIVS benchmark reference protocol."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPLITS), required=True)
    parser.add_argument(
        "--split",
        choices=("molecule-random", "series-disjoint", "ave"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    allowed = BENCHMARK_SPLITS[args.benchmark]
    if args.split not in allowed:
        parser.error(
            f"{args.benchmark} supports {', '.join(sorted(allowed))}, not {args.split}"
        )
    ks = sorted(set(args.k))
    if not ks or ks[0] < 1 or ks[-1] > 10:
        parser.error("--k values must be between 1 and 10")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"--output-dir must be empty: {args.output_dir}")

    dataset_root = args.data_root / args.benchmark
    split_folder = "random" if args.split == "molecule-random" else args.split.replace("-", "_")
    split_root = dataset_root / "splits" / split_folder
    episode_file = split_root / "episodes.csv"
    episodes = read_episode_rows(episode_file, args.seed, set(ks))
    requested_targets = set(args.targets or [])
    available_targets = {row["target_id"] for row in episodes}
    unknown = requested_targets - available_targets
    if unknown:
        parser.error(f"targets unavailable for this split/seed: {sorted(unknown)}")
    if requested_targets:
        episodes = [row for row in episodes if row["target_id"] in requested_targets]

    configure_determinism()
    device = torch.device(args.device)
    cache = EmbeddingCache(dataset_root / "ept")
    projection = load_projection(dataset_root / "whitener.npz", device)
    params = json.loads(args.theta.read_text())
    series = (
        read_series(split_root / "series_assignments.csv")
        if args.split == "series-disjoint"
        else None
    )
    rows_by_target = defaultdict(list)
    for row in episodes:
        rows_by_target[row["target_id"]].append(row)

    output_rows = []
    for target_index, target_id in enumerate(sorted(rows_by_target), 1):
        if args.split == "ave":
            target_root = split_root / "official_ave" / target_id
            candidate_ids = read_smi_ids(
                target_root / "active_V.smi"
            ) | read_smi_ids(target_root / "inactive_V.smi")
            available_ids = set(cache.parent_ids[target_id].astype(str))
            selected_ids = (candidate_ids & available_ids) | {
                parent
                for episode in rows_by_target[target_id]
                for parent in episode["references"]
            }
            target = cache.get_subset(
                target_id,
                selected_ids,
                device,
                projection=projection,
                include_labels=True,
                retain_raw=True,
            )
        else:
            target = cache.get(
                target_id,
                device,
                projection=projection,
                include_labels=True,
                retain_raw=args.split == "series-disjoint",
            )
        for episode in sorted(rows_by_target[target_id], key=lambda row: row["K"]):
            references = episode["references"]
            visible, ranked = episode_masks(
                target.parent_ids,
                references,
                args.split,
                target_id,
                split_root,
                series,
            )
            labels = target.parent_labels[ranked]
            positives = int(labels.sum())
            negatives = int(len(labels) - positives)
            excluded = int((~visible).sum()) if args.split == "series-disjoint" else 0
            observed_counts = (
                int(ranked.sum()),
                positives,
                negatives,
                excluded,
            )
            expected_counts = (
                episode["expected_candidate_count"],
                episode["expected_positive_count"],
                episode["expected_negative_count"],
                episode["expected_excluded_count"],
            )
            if observed_counts != expected_counts:
                raise ValueError(
                    f"{target_id}, seed={args.seed}, K={episode['K']}: "
                    f"split counts {observed_counts} differ from saved counts "
                    f"{expected_counts}"
                )
            if positives == 0 or negatives == 0:
                raise ValueError(
                    f"{target_id}, K={episode['K']}: metrics require both classes"
                )
            episode_target = target if visible.all() else subset_target(target, visible)
            pool = ReferencePool.from_cached_ids(
                references, source="benchmark_actives"
            )
            episode_target, references = pool.materialize(
                episode_target, projection
            )
            result = score_library(episode_target, references, params)
            expected_ids = target.parent_ids[ranked].astype(str)
            if not np.array_equal(result.parent_ids.astype(str), expected_ids):
                raise RuntimeError(f"{target_id}: ranked library differs from split data")
            metrics = screening_metrics(labels, result.scores)
            output_row = {
                "benchmark": args.benchmark,
                "split": args.split,
                "target_id": target_id,
                "seed": args.seed,
                "K": episode["K"],
                "support_parent_ids": json.dumps(references, separators=(",", ":")),
                "support_count": len(references),
                "visible_count": int(visible.sum()),
                "candidate_count": int(ranked.sum()),
                "positive_count": positives,
                "negative_count": negatives,
                "excluded_count": excluded,
                **metrics,
            }
            output_rows.append(output_row)
            metric_text = " ".join(
                f"{name}={metrics[name]:.6g}" for name in METRICS
            )
            print(
                f"[{target_index}/{len(rows_by_target)}] target={target_id} "
                f"seed={args.seed} K={episode['K']} references={references} "
                f"candidates={len(labels)} positives={positives} negatives={negatives} "
                f"excluded={excluded} {metric_text}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(output_rows)
    frame.to_csv(args.output_dir / "episodes.csv", index=False)
    by_k, overall = summarize(frame, args.output_dir)
    print("\nTarget-macro summary by K")
    display = by_k[["K", *[f"{metric}_mean" for metric in METRICS]]].rename(
        columns={f"{metric}_mean": metric for metric in METRICS}
    )
    print(display.to_string(index=False))
    print("\nOverall target-macro summary")
    for metric, values in overall.items():
        print(
            f"{metric}: mean={values['mean']:.6g} "
            f"std={values['std']:.6g} targets={values['targets']}"
        )
    print(f"\nWrote {len(frame)} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
