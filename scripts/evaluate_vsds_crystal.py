#!/usr/bin/env python3
"""Evaluate VSDS-vd with its fixed crystal ligand as an external K=1 support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tactivs.cache import EmbeddingCache, load_projection, molecule_balanced_mean
from tactivs.evaluation import score_external_reference
from tactivs.provenance import (
    PROTOCOL_VERSION,
    artifact_set_sha256,
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


def load_reference_cache(path: Path) -> dict[str, np.ndarray]:
    shards = {
        shard.name: np.load(shard, mmap_mode="r")
        for shard in sorted(path.glob("embeddings_*.npy"))
    }
    if not shards:
        raise FileNotFoundError(f"no reference embedding shards in {path}")
    rows: dict[str, list[np.ndarray]] = {}
    with (path / "metadata.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["target_id"], []).append(
                np.asarray(
                    shards[row["embedding_shard"]][int(row["embedding_row"])],
                    dtype=np.float32,
                )
            )
    return {target_id: np.stack(values) for target_id, values in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["truedecoy", "randomdecoy"])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    configure_determinism()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    params = json.loads(args.theta.read_text())
    config = json.loads(args.data_config.read_text())
    specs = {
        spec["name"]: spec
        for spec in config["datasets"]
        if spec["name"] in set(args.datasets)
    }
    if set(specs) != set(args.datasets):
        raise ValueError("requested crystal datasets are missing from the data config")
    raw_references = load_reference_cache(args.reference_cache)

    rows = []
    molecule_scores_path = args.output_dir / "molecule_scores.csv.gz"
    with MoleculeScoreWriter(molecule_scores_path) as score_writer:
        for dataset_name in args.datasets:
            spec = specs[dataset_name]
            cache = EmbeddingCache(Path(spec["cache"]))
            projection = load_projection(Path(spec["whitener"]), device)
            if len(cache.target_ids) != int(spec["expected_targets"]):
                raise ValueError(f"{dataset_name}: unexpected target count")
            missing = sorted(set(cache.target_ids) - set(raw_references))
            if missing:
                raise ValueError(f"{dataset_name}: missing crystal references {missing}")
            for index, target_id in enumerate(cache.target_ids, 1):
                raw = cache.get_raw(target_id, device)
                counts = cache.meta[target_id]["parent_counts"]
                target_mean = molecule_balanced_mean(raw, counts)
                target = cache.get(
                    target_id, device, projection=projection, include_labels=True
                )
                reference = F.normalize(
                    (
                        torch.as_tensor(raw_references[target_id], device=device)
                        - target_mean
                    )
                    @ projection,
                    dim=1,
                )
                summary_row, score_result = score_external_reference(
                    target, reference, params, return_scores=True
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "config": "selected_theta",
                        **summary_row,
                    }
                )
                score_writer.write_complete_library(
                    dataset_name,
                    "selected_theta",
                    target,
                    "crystal_reference",
                    1,
                    score_result,
                )
                if index % 20 == 0 or index == len(cache.target_ids):
                    print(
                        f"[{dataset_name} {index}/{len(cache.target_ids)}]", flush=True
                    )
        molecule_score_rows = score_writer.row_count

    episodes = pd.DataFrame(rows)
    episodes.to_csv(args.output_dir / "episodes.csv", index=False)
    summary = (
        episodes.groupby(["dataset", "config", "K"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(args.output_dir / "metrics.csv", index=False)
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "theta": str(args.theta.resolve()),
        "theta_sha256": sha256(args.theta),
        "data_config": str(args.data_config.resolve()),
        "data_config_sha256": sha256(args.data_config),
        "reference_cache": str(args.reference_cache.resolve()),
        "reference_metadata_sha256": sha256(args.reference_cache / "metadata.csv"),
        "reference_cache_sha256": artifact_set_sha256(
            [args.reference_cache / "metadata.csv"]
            + sorted(args.reference_cache.glob("embeddings_*.npy"))
        ),
        "datasets": args.datasets,
        "protocol": "one fixed external VSDS-vd crystal ligand per target",
        "K": 1,
        "candidate_library": "all cached active and decoy molecules; no support sampling",
        "aggregation": "mean and sample SD across targets",
        "molecule_scores": str(molecule_scores_path.resolve()),
        "molecule_scores_sha256": sha256(molecule_scores_path),
        "molecule_score_rows": molecule_score_rows,
        "source_sha256": source_sha256(ROOT),
        "software": software_versions(),
        "device": str(device),
        "artifacts": {
            dataset_name: {
                "cache_sha256": cache_sha256(Path(specs[dataset_name]["cache"])),
                "whitener_sha256": sha256(Path(specs[dataset_name]["whitener"])),
            }
            for dataset_name in args.datasets
        },
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
