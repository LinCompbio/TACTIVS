#!/usr/bin/env python3
"""Run resumable direct/graph ablations with target-level score checkpoints."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tactivs.cache import EmbeddingCache, load_projection, subset_target
from tactivs.core import score_library
from tactivs.episodes import (
    build_episode_plan,
    episode_parent_masks,
    read_positive_manifest,
    series_groups,
)
from tactivs.metrics import paired_bootstrap, screening_metrics
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
CONFIGS = ("full", "direct_only", "graph_only")
METRICS = (
    "auc_roc",
    "ef_0.5%",
    "ef_1%",
    "ef_5%",
    "bedroc_20",
    "bedroc_80_5",
    "bayes_ef_1%",
)


def compress_chunk(task: tuple[str, int, int, str]) -> str:
    source, begin, end, destination = task
    with open(source, "rb") as input_handle, open(destination, "wb") as raw_output:
        input_handle.seek(begin)
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
        ) as output_handle:
            remaining = end - begin
            while remaining:
                block = input_handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("unexpected end of score CSV")
                output_handle.write(block)
                remaining -= len(block)
    return destination


def parallel_gzip(source: Path, destination: Path, workers: int) -> None:
    size = source.stat().st_size
    boundaries = [0]
    with source.open("rb") as handle:
        for index in range(1, workers):
            handle.seek(size * index // workers)
            handle.readline()
            boundary = handle.tell()
            if boundary > boundaries[-1] and boundary < size:
                boundaries.append(boundary)
    boundaries.append(size)
    parts = [
        destination.with_name(f"{destination.name}.part{index:02d}")
        for index in range(len(boundaries) - 1)
    ]
    tasks = [
        (str(source), begin, end, str(part))
        for begin, end, part in zip(boundaries, boundaries[1:], parts)
    ]
    with multiprocessing.get_context("spawn").Pool(len(tasks)) as pool:
        pool.map(compress_chunk, tasks)
    with destination.open("wb") as output_handle:
        for part in parts:
            with part.open("rb") as input_handle:
                shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
            part.unlink()
    source.unlink()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def validate_complete(marker: Path) -> dict | None:
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    for field in ("episodes", "molecule_scores"):
        artifact = marker.parent / record[field]["file"]
        if not artifact.exists() or artifact.stat().st_size == 0:
            return None
    return record


def prepare_dataset(spec: dict, seeds: list[int], ks: list[int], protocols: list[str]):
    cache = EmbeddingCache(Path(spec["cache"]))
    projection_path = Path(spec["whitener"])
    manifest = read_positive_manifest(Path(spec["positive_manifest"]))
    if len(cache.target_ids) != int(spec["expected_targets"]):
        raise ValueError(f"{spec['name']}: unexpected target count")
    if sum(map(len, manifest.values())) != int(spec["expected_positives"]):
        raise ValueError(f"{spec['name']}: unexpected positive count")

    episode_records = {}
    for target_id in cache.target_ids:
        labels = cache.get_parent_labels(target_id)
        cached = set(map(str, cache.parent_ids[target_id][labels == 1]))
        declared = set(manifest.get(target_id, {}))
        if cached != declared:
            raise ValueError(f"{spec['name']}/{target_id}: label-manifest mismatch")
        episode_records[target_id] = {
            parent: manifest[target_id][parent] for parent in cached
        }
    plan = build_episode_plan(episode_records, protocols, seeds, max(ks))
    series = {
        target_id: series_groups(records)
        for target_id, records in episode_records.items()
    }
    planned = set().union(*(set(plan[protocol]) for protocol in protocols))
    target_ids = sorted(planned & set(cache.target_ids))
    if len(target_ids) != int(spec["expected_targets"]):
        raise ValueError(f"{spec['name']}: incomplete episode target set")
    return cache, projection_path, plan, series, target_ids


def score_variants(result, graph_weight: float):
    return {
        "full": result,
        "direct_only": replace(result, scores=result.direct.copy()),
        "graph_only": replace(
            result, scores=(graph_weight * result.graph).copy()
        ),
    }


@torch.no_grad()
def evaluate_target(
    *,
    dataset: str,
    target,
    plan: dict,
    series_of_parent: dict[str, str],
    params: dict,
    seeds: list[int],
    ks: list[int],
    protocols: list[str],
    shard_dir: Path,
    compression_workers: int,
) -> dict:
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = shard_dir / "complete.json"
    complete = validate_complete(marker)
    if complete is not None:
        return complete

    episode_path = shard_dir / "episodes.csv"
    score_path = shard_dir / "molecule_scores.csv.gz"
    episode_tmp = shard_dir / "episodes.csv.tmp"
    score_tmp = shard_dir / "molecule_scores.csv.tmp"
    for stale in (episode_tmp, score_tmp):
        if stale.exists():
            stale.unlink()

    rows = []
    with MoleculeScoreWriter(score_tmp, compression=False) as writer:
        for protocol in protocols:
            if target.target_id not in plan[protocol]:
                continue
            for seed in seeds:
                support_order = plan[protocol][target.target_id][str(seed)][
                    "support_order"
                ]
                for known in ks:
                    support_ids = list(map(str, support_order[:known]))
                    visible_mask, ranked_mask = episode_parent_masks(
                        target.parent_ids,
                        support_ids,
                        protocol,
                        series_of_parent,
                    )
                    labels = target.parent_labels[ranked_mask]
                    positives = int((labels == 1).sum())
                    negatives = int((labels == 0).sum())
                    if positives == 0 or negatives == 0:
                        continue
                    episode_target = (
                        target
                        if visible_mask.all()
                        else subset_target(target, visible_mask)
                    )
                    base = score_library(episode_target, support_ids, params)
                    expected_ids = target.parent_ids[ranked_mask]
                    if not np.array_equal(
                        base.parent_ids.astype(str), expected_ids.astype(str)
                    ):
                        raise RuntimeError("scored and metric libraries differ")
                    variants = score_variants(base, float(params["graph_weight"]))
                    writer.write_episode(
                        dataset,
                        "components",
                        target,
                        protocol,
                        seed,
                        known,
                        support_ids,
                        visible_mask,
                        ranked_mask,
                        base,
                    )
                    for config, result in variants.items():
                        rows.append(
                            {
                                "dataset": dataset,
                                "config": config,
                                "target_id": target.target_id,
                                "protocol": protocol,
                                "seed": seed,
                                "K": known,
                                "support_parent_ids": json.dumps(
                                    support_ids, separators=(",", ":")
                                ),
                                "support_count": len(support_ids),
                                "visible_count": int(visible_mask.sum()),
                                "candidate_count": int(ranked_mask.sum()),
                                "positive_count": positives,
                                "negative_count": negatives,
                                "series_excluded_count": int((~visible_mask).sum()),
                                **screening_metrics(labels, result.scores),
                            }
                        )
        molecule_rows = writer.row_count

    pd.DataFrame(rows).to_csv(episode_tmp, index=False)
    episode_tmp.replace(episode_path)
    parallel_gzip(score_tmp, score_path, compression_workers)
    record = {
        "dataset": dataset,
        "target_id": target.target_id,
        "episode_rows": len(rows),
        "molecule_score_rows": molecule_rows,
        "episodes": {
            "file": episode_path.name,
        },
        "molecule_scores": {
            "file": score_path.name,
        },
    }
    write_json_atomic(marker, record)
    return record


def flat_metric_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    target_groups = groups + ["target_id"]
    target = frame.groupby(target_groups, as_index=False)[list(METRICS)].mean()
    summary = target.groupby(groups)[list(METRICS)].agg(["mean", "std", "count"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def contribution_table(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    target_groups = groups + ["target_id", "config"]
    target = (
        frame.groupby(target_groups, as_index=False)["ef_1%"]
        .mean()
        .pivot(index=groups + ["target_id"], columns="config", values="ef_1%")
        .reset_index()
    )
    rows = []
    for keys, subset in target.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(groups, keys))
        for module, reduced in (
            ("direct", "graph_only"),
            ("graph", "direct_only"),
        ):
            delta = (subset["full"] - subset[reduced]).to_numpy()
            mean, low, high = paired_bootstrap(delta)
            rows.append(
                {
                    **group_values,
                    "module": module,
                    "targets": len(delta),
                    "full_ef_1%": float(subset["full"].mean()),
                    "reduced_configuration": reduced,
                    "reduced_ef_1%": float(subset[reduced].mean()),
                    "module_contribution": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_outputs(
    output_dir: Path,
    specs: list[dict],
    records: list[dict],
    run_spec: dict,
    device: torch.device,
) -> None:
    episode_frames = []
    manifest_rows = []
    for record in records:
        shard_dir = (
            output_dir
            / record["dataset"]
            / "targets"
            / safe_name(record["target_id"])
        )
        episode_frames.append(pd.read_csv(shard_dir / record["episodes"]["file"]))
        manifest_rows.append(
            {
                "dataset": record["dataset"],
                "target_id": record["target_id"],
                "episode_rows": record["episode_rows"],
                "molecule_score_rows": record["molecule_score_rows"],
                "episodes": str(
                    (shard_dir / record["episodes"]["file"]).relative_to(output_dir)
                ),
                "molecule_scores": str(
                    (shard_dir / record["molecule_scores"]["file"]).relative_to(
                        output_dir
                    )
                ),
            }
        )
    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes.to_csv(
        output_dir / "episodes.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "target_artifacts.csv", index=False)

    target_summary = (
        episodes.groupby(["dataset", "config", "target_id"], as_index=False)[
            list(METRICS)
        ]
        .mean()
    )
    target_summary.to_csv(output_dir / "target_metrics.csv", index=False)
    flat_metric_summary(episodes, ["dataset", "config"]).to_csv(
        output_dir / "summary.csv", index=False
    )
    flat_metric_summary(episodes, ["dataset", "config", "K"]).to_csv(
        output_dir / "metrics_by_k.csv", index=False
    )
    flat_metric_summary(
        episodes, ["dataset", "config", "protocol", "K"]
    ).to_csv(output_dir / "metrics_by_protocol_k.csv", index=False)

    fixed = episodes.loc[episodes.protocol == "fixed_random"]
    flat_metric_summary(fixed, ["dataset", "config", "K"]).to_csv(
        output_dir / "table2_sampled_active.csv", index=False
    )
    contribution_table(episodes, ["dataset"]).to_csv(
        output_dir / "module_contributions.csv", index=False
    )
    contribution_table(episodes, ["dataset", "K"]).to_csv(
        output_dir / "module_contributions_by_k.csv", index=False
    )
    contribution_table(episodes, ["dataset", "protocol"]).to_csv(
        output_dir / "module_contributions_by_protocol.csv", index=False
    )
    contribution_table(episodes, ["dataset", "protocol", "K"]).to_csv(
        output_dir / "module_contributions_by_protocol_k.csv", index=False
    )

    provenance = {
        **run_spec,
        "aggregation": (
            "average seeds, K and available protocols within target; "
            "then target macro"
        ),
        "table2_aggregation": (
            "fixed_random only; average ten seeds within target and K; "
            "then mean and sample SD across targets"
        ),
        "configurations": {
            "full": "direct + graph",
            "direct_only": "graph_weight set to zero analytically",
            "graph_only": "direct component set to zero analytically",
        },
        "episode_rows": len(episodes),
        "molecule_score_rows": int(manifest.molecule_score_rows.sum()),
        "software": software_versions(),
        "device": str(device),
        "source_sha256": source_sha256(ROOT),
        "artifacts": {
            spec["name"]: {
                "cache_sha256": cache_sha256(Path(spec["cache"])),
                "whitener_sha256": sha256(Path(spec["whitener"])),
                "positive_manifest_sha256": sha256(
                    Path(spec["positive_manifest"])
                ),
            }
            for spec in specs
        },
    }
    write_json_atomic(output_dir / "run_provenance.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--ks", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("fixed_random", "series"),
        default=["fixed_random", "series"],
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--compression-workers", type=int, default=32)
    args = parser.parse_args()

    configure_determinism()
    seeds = sorted(set(args.seeds))
    ks = sorted(set(args.ks))
    protocols = list(dict.fromkeys(args.protocols))
    data_config = json.loads(args.data_config.read_text())
    specs = data_config["datasets"]
    if args.datasets:
        requested = set(args.datasets)
        specs = [spec for spec in specs if spec["name"] in requested]
        if {spec["name"] for spec in specs} != requested:
            parser.error("unknown dataset requested")
    params = json.loads(args.theta.read_text())
    run_spec = {
        "protocol_version": f"{PROTOCOL_VERSION}-density-free-ablation-v1",
        "theta": str(args.theta.resolve()),
        "theta_sha256": sha256(args.theta),
        "parameters": params,
        "data_config": str(args.data_config.resolve()),
        "data_config_sha256": sha256(args.data_config),
        "datasets": [spec["name"] for spec in specs],
        "seeds": seeds,
        "ks": ks,
        "protocols": protocols,
        "checkpoint_boundary": "one complete target with all configurations",
        "compression_workers": args.compression_workers,
        "compression": "ordered concatenated gzip members with mtime=0",
        "script_sha256": sha256(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.output_dir / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != run_spec:
        raise RuntimeError("run specification changed; use a new output directory")
    if not spec_path.exists():
        write_json_atomic(spec_path, run_spec)

    device = torch.device(args.device)
    records = []
    for spec in specs:
        cache, projection_path, plan, series, target_ids = prepare_dataset(
            spec, seeds, ks, protocols
        )
        projection = load_projection(projection_path, device)
        dataset_dir = args.output_dir / spec["name"]
        for index, target_id in enumerate(target_ids, 1):
            shard_dir = dataset_dir / "targets" / safe_name(target_id)
            marker = shard_dir / "complete.json"
            complete = validate_complete(marker)
            if complete is None:
                target = cache.get(
                    target_id,
                    device,
                    projection=projection,
                    include_labels=True,
                    retain_raw="series" in protocols,
                )
                complete = evaluate_target(
                    dataset=spec["name"],
                    target=target,
                    plan=plan,
                    series_of_parent=series[target_id],
                    params=params,
                    seeds=seeds,
                    ks=ks,
                    protocols=protocols,
                    shard_dir=shard_dir,
                    compression_workers=args.compression_workers,
                )
                status = "completed"
            else:
                status = "reused"
            records.append(complete)
            print(
                f"[{spec['name']} {index}/{len(target_ids)}] "
                f"{target_id} {status}",
                flush=True,
            )
    aggregate_outputs(args.output_dir, specs, records, run_spec, device)


if __name__ == "__main__":
    main()
