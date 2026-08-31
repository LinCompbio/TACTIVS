#!/usr/bin/env python3
"""Evaluate a frozen theta; labels are accessed only in this script."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tactivs.cache import EmbeddingCache, load_projection
from tactivs.episodes import build_episode_plan, read_positive_manifest, series_groups
from tactivs.evaluation import score_episodes
from tactivs.metrics import paired_bootstrap
from tactivs.provenance import (
    PROTOCOL_VERSION,
    cache_sha256,
    configure_determinism,
    sha256,
    software_versions,
    source_sha256,
)
from tactivs.result_writer import MoleculeScoreWriter

ROOT = Path(__file__).resolve().parents[1]


METRICS = (
    "auc_roc",
    "ef_0.5%",
    "ef_1%",
    "ef_5%",
    "bedroc_20",
    "bedroc_80_5",
    "bayes_ef_1%",
)


def evaluate_dataset(spec, params_by_name, seeds, ks, protocols, device, score_writers):
    cache = EmbeddingCache(Path(spec["cache"]))
    projection = load_projection(Path(spec["whitener"]), device)
    manifest = read_positive_manifest(Path(spec["positive_manifest"]))
    manifest_positive_count = sum(len(records) for records in manifest.values())
    if (
        "expected_positives" in spec
        and manifest_positive_count != spec["expected_positives"]
    ):
        raise ValueError(
            f"{spec['name']}: expected {spec['expected_positives']} manifest positives, "
            f"found {manifest_positive_count}"
        )
    if "expected_targets" in spec and len(cache.target_ids) != spec["expected_targets"]:
        raise ValueError(
            f"{spec['name']}: expected {spec['expected_targets']} cached targets, "
            f"found {len(cache.target_ids)}"
        )
    episode_records = {}
    for target_id in cache.target_ids:
        labels = cache.get_parent_labels(target_id)
        if not set(map(int, np.unique(labels))) <= {0, 1}:
            raise ValueError(f"{spec['name']}/{target_id}: labels must be binary")
        cached_positives = set(map(str, cache.parent_ids[target_id][labels == 1]))
        manifest_positives = set(manifest.get(target_id, {}))
        if cached_positives != manifest_positives:
            raise ValueError(
                f"{spec['name']}/{target_id}: cache labels and positive manifest differ"
            )
        episode_records[target_id] = {
            parent: manifest[target_id][parent] for parent in cached_positives
        }

    plan = build_episode_plan(episode_records, protocols, seeds, max(ks))
    series = {
        target_id: series_groups(records)
        for target_id, records in episode_records.items()
    }
    planned_targets = set().union(*(set(plan[protocol]) for protocol in protocols))
    target_ids = sorted(planned_targets & set(cache.target_ids))
    if "expected_targets" in spec and len(target_ids) != spec["expected_targets"]:
        raise ValueError(
            f"{spec['name']}: only {len(target_ids)} targets satisfy the episode protocol; "
            f"expected {spec['expected_targets']}"
        )
    rows = []
    for index, target_id in enumerate(target_ids, 1):
        target = cache.get(
            target_id,
            device,
            projection=projection,
            include_labels=True,
            retain_raw="series" in protocols,
        )
        for config, params in params_by_name.items():
            for result in score_episodes(
                target,
                plan,
                ks,
                seeds,
                params,
                protocols=protocols,
                series_of_parent=series[target_id],
                score_callback=lambda *values, dataset=spec["name"], name=config: (
                    score_writers[name].write_episode(dataset, name, *values)
                ),
            ):
                rows.append({"dataset": spec["name"], "config": config, **result})
        if index % 20 == 0 or index == len(target_ids):
            print(f"[{spec['name']} {index}/{len(target_ids)}]", flush=True)
    return rows


def _write_metric_tables(episodes: pd.DataFrame, output_dir: Path) -> None:
    target_k = episodes.groupby(
        ["dataset", "config", "K", "target_id"], as_index=False
    )["ef_1%"].mean()
    target_k.groupby(["dataset", "config", "K"], as_index=False)["ef_1%"].mean().to_csv(
        output_dir / "per_dataset_k.csv", index=False
    )

    target_k_metrics = episodes.groupby(
        ["dataset", "config", "K", "target_id"], as_index=False
    )[list(METRICS)].mean()
    metric_summary = (
        target_k_metrics.groupby(["dataset", "config", "K"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    metric_summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in metric_summary.columns
    ]
    metric_summary.to_csv(output_dir / "metrics_per_dataset_k.csv", index=False)

    target_protocol_k_metrics = episodes.groupby(
        ["dataset", "config", "protocol", "K", "target_id"], as_index=False
    )[list(METRICS)].mean()
    protocol_metric_summary = (
        target_protocol_k_metrics.groupby(
            ["dataset", "config", "protocol", "K"]
        )[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    protocol_metric_summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in protocol_metric_summary.columns
    ]
    protocol_metric_summary.to_csv(
        output_dir / "metrics_per_dataset_protocol_k.csv", index=False
    )


def write_ablation_outputs(
    args,
    datasets,
    params,
    episodes,
    molecule_scores_paths,
    molecule_score_rows,
    device,
    ks,
    seeds,
) -> None:
    artifact_hashes = {
        dataset["name"]: {
            "cache_sha256": cache_sha256(Path(dataset["cache"])),
            "whitener_sha256": sha256(Path(dataset["whitener"])),
            "positive_manifest_sha256": sha256(Path(dataset["positive_manifest"])),
        }
        for dataset in datasets
    }
    code_hash = source_sha256(ROOT)
    summaries = []
    for config, config_params in params.items():
        output_dir = args.output_dir / config
        config_episodes = episodes.loc[episodes["config"] == config].copy()
        config_episodes.to_csv(output_dir / "episodes.csv", index=False)
        _write_metric_tables(config_episodes, output_dir)

        target = config_episodes.groupby(["dataset", "target_id"])["ef_1%"].mean()
        config_summary = [
            {
                "dataset": dataset,
                "configuration": config,
                "targets": len(frame),
                "ef_1%": float(frame.mean()),
            }
            for dataset, frame in target.groupby(level="dataset")
        ]
        summaries.extend(config_summary)
        pd.DataFrame(config_summary).to_csv(output_dir / "summary.csv", index=False)

        evaluation_summary = {
            "protocol_version": PROTOCOL_VERSION,
            "configuration": config,
            "aggregation": (
                "average seeds, K and available protocols within target; "
                "then target macro"
            ),
            "ks": ks,
            "seeds": seeds,
            "protocols": args.protocols,
            "episode_boundary": (
                "runtime support; every visible non-support parent is ranked"
            ),
            "datasets": config_summary,
        }
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(evaluation_summary, indent=2) + "\n"
        )

        molecule_scores_path = molecule_scores_paths[config]
        provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "configuration": config,
            "parameters": config_params,
            "theta": str(args.theta.resolve()),
            "theta_sha256": sha256(args.theta),
            "data_config": str(args.data_config.resolve()),
            "data_config_sha256": sha256(args.data_config),
            "seeds": seeds,
            "ks": ks,
            "protocols": args.protocols,
            "episode_boundary": (
                "runtime support; every visible non-support parent is ranked"
            ),
            "datasets": [dataset["name"] for dataset in datasets],
            "aggregation": (
                "average seeds and protocols within target and K; "
                "then mean and sample SD across targets"
            ),
            "bedroc_alpha": [20.0, 80.5],
            "molecule_scores": str(molecule_scores_path.resolve()),
            "molecule_scores_sha256": sha256(molecule_scores_path),
            "molecule_score_rows": molecule_score_rows[config],
            "source_sha256": code_hash,
            "software": software_versions(),
            "device": str(device),
            "artifacts": artifact_hashes,
        }
        (output_dir / "run_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    target = (
        episodes.groupby(["dataset", "config", "target_id"])["ef_1%"]
        .mean()
        .unstack("config")
    )
    ablations = []
    for dataset, frame in target.groupby(level="dataset"):
        full = frame["full"]
        for module, reduced_name in (
            ("direct", "graph_only"),
            ("graph", "direct_only"),
        ):
            gain, low, high = paired_bootstrap(
                (full - frame[reduced_name]).to_numpy()
            )
            ablations.append(
                {
                    "dataset": dataset,
                    "module": module,
                    "full_ef": float(full.mean()),
                    "reduced_configuration": reduced_name,
                    "reduced_ef": float(frame[reduced_name].mean()),
                    "module_contribution": gain,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    ablation_summary = pd.DataFrame(ablations)
    ablation_summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    print(ablation_summary.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("fixed_random", "series"),
        default=["fixed_random", "series"],
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="optional dataset names from the data config",
    )
    args = parser.parse_args()
    configure_determinism()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_config = json.loads(args.data_config.read_text())
    selected = json.loads(args.theta.read_text())
    if args.ablations:
        params = {
            "full": selected,
            "direct_only": {**selected, "graph_weight": 0.0},
            "graph_only": {**selected, "similarity_weight": 0.0},
        }
    else:
        params = {"selected_theta": selected}
    seeds = sorted(set(args.seeds))
    ks = sorted(set(args.ks))
    if not seeds or not ks or min(ks) < 1:
        parser.error("--seeds and positive --ks values are required")
    rows = []
    datasets = data_config["datasets"]
    if args.datasets:
        requested = set(args.datasets)
        available = {dataset["name"] for dataset in datasets}
        missing = sorted(requested - available)
        if missing:
            parser.error(f"unknown --datasets values: {', '.join(missing)}")
        datasets = [dataset for dataset in datasets if dataset["name"] in requested]
    if args.ablations:
        molecule_scores_paths = {
            name: args.output_dir / name / "molecule_scores.csv.gz" for name in params
        }
    else:
        shared_path = args.output_dir / "molecule_scores.csv.gz"
        molecule_scores_paths = {name: shared_path for name in params}
    for path in set(molecule_scores_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        writers_by_path = {
            path: stack.enter_context(MoleculeScoreWriter(path))
            for path in set(molecule_scores_paths.values())
        }
        score_writers = {
            name: writers_by_path[path] for name, path in molecule_scores_paths.items()
        }
        for dataset in datasets:
            rows.extend(
                evaluate_dataset(
                    dataset,
                    params,
                    seeds,
                    ks,
                    args.protocols,
                    device,
                    score_writers,
                )
            )
        molecule_score_rows = {
            name: writer.row_count for name, writer in score_writers.items()
        }
    episodes = pd.DataFrame(rows)

    if args.ablations:
        write_ablation_outputs(
            args,
            datasets,
            params,
            episodes,
            molecule_scores_paths,
            molecule_score_rows,
            device,
            ks,
            seeds,
        )
        return

    episodes.to_csv(args.output_dir / "episodes.csv", index=False)

    target_k = (
        episodes.groupby(["dataset", "config", "K", "target_id"])["ef_1%"]
        .mean()
        .reset_index()
    )
    per_k = (
        target_k.groupby(["dataset", "config", "K"])["ef_1%"]
        .mean()
        .reset_index()
    )
    per_k.to_csv(args.output_dir / "per_dataset_k.csv", index=False)

    # Match the published benchmark convention: average repeated episodes within
    # each target first, then report the mean and sample SD across targets.
    target_k_metrics = episodes.groupby(
        ["dataset", "config", "K", "target_id"], as_index=False
    )[list(METRICS)].mean()
    metric_summary = (
        target_k_metrics.groupby(["dataset", "config", "K"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    metric_summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in metric_summary.columns
    ]
    metric_summary.to_csv(args.output_dir / "metrics_per_dataset_k.csv", index=False)

    target_protocol_k_metrics = episodes.groupby(
        ["dataset", "config", "protocol", "K", "target_id"], as_index=False
    )[list(METRICS)].mean()
    protocol_metric_summary = (
        target_protocol_k_metrics.groupby(["dataset", "config", "protocol", "K"])[
            list(METRICS)
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    protocol_metric_summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in protocol_metric_summary.columns
    ]
    protocol_metric_summary.to_csv(
        args.output_dir / "metrics_per_dataset_protocol_k.csv", index=False
    )

    target = episodes.groupby(["dataset", "target_id"])["ef_1%"].mean()
    summary = [
        {
            "dataset": dataset,
            "targets": len(frame),
            "selected_theta": float(frame.mean()),
        }
        for dataset, frame in target.groupby(level="dataset")
    ]
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    evaluation_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "aggregation": (
            "average seeds, K and available protocols within target; then target macro"
        ),
        "ks": ks,
        "seeds": seeds,
        "protocols": args.protocols,
        "episode_boundary": "runtime support; every visible non-support parent is ranked",
        "datasets": summary,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2) + "\n"
    )
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "theta": str(args.theta.resolve()),
        "theta_sha256": sha256(args.theta),
        "data_config": str(args.data_config.resolve()),
        "data_config_sha256": sha256(args.data_config),
        "seeds": seeds,
        "ks": ks,
        "protocols": args.protocols,
        "episode_boundary": "runtime support; every visible non-support parent is ranked",
        "datasets": [dataset["name"] for dataset in datasets],
        "aggregation": "average seeds and protocols within target and K; then mean and sample SD across targets",
        "bedroc_alpha": [20.0, 80.5],
        "molecule_scores": str(shared_path.resolve()),
        "molecule_scores_sha256": sha256(shared_path),
        "molecule_score_rows": next(iter(molecule_score_rows.values())),
        "source_sha256": source_sha256(ROOT),
        "software": software_versions(),
        "device": str(device),
        "artifacts": {
            dataset["name"]: {
                "cache_sha256": cache_sha256(Path(dataset["cache"])),
                "whitener_sha256": sha256(Path(dataset["whitener"])),
                "positive_manifest_sha256": sha256(Path(dataset["positive_manifest"])),
            }
            for dataset in datasets
        },
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary_frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
