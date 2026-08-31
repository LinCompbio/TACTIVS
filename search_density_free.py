#!/usr/bin/env python3
"""Run the reproducible 100-trial DEKOIS reference-only density-free search."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from tactivs.cache import EmbeddingCache, TargetData, load_projection
from tactivs.core import (
    build_graph,
    molecule_centers,
    pool_by_parent,
    propagate_graph,
    zscore,
)
from tactivs.episodes import build_episode_plan, read_positive_manifest, series_groups
from tactivs.metrics import ef_at_fraction, paired_bootstrap
from tactivs.provenance import (
    PROTOCOL_VERSION,
    REPRODUCIBILITY_SEED,
    cache_sha256,
    configure_determinism,
    sha256,
    software_versions,
    source_sha256,
)

ROOT = Path(__file__).resolve().parent
STUDY_NAME = "tactivs_density_free_dekois_reference_only_v1"
TRIAL_BUDGET = 100
SELECTION_KS = [3, 5, 7, 10]
SEEDS = [0, 1, 2]
PROTOCOLS = ("fixed_random", "series")
NEGATIVE_CONTRIBUTION_PENALTY = 5.0


@dataclass
class SurrogateTask:
    protocol: str
    seed: int
    known: int
    held_codes: list[int]
    reference_codes: list[int]


@dataclass
class PreparedTarget:
    target: TargetData
    pose_codes: torch.Tensor
    centers: torch.Tensor
    tasks: list[SurrogateTask]
    positive_codes: list[int]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def suggest(trial: optuna.Trial, name: str, spec: dict):
    if spec["type"] == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    return trial.suggest_float(
        name,
        float(spec["low"]),
        float(spec["high"]),
        log=bool(spec.get("log")),
    )


def prepare_target(target, series_of_parent, plan) -> PreparedTarget:
    lookup = {str(parent): index for index, parent in enumerate(target.parent_ids)}
    tasks = []
    for protocol in PROTOCOLS:
        if target.target_id not in plan[protocol]:
            continue
        for seed in SEEDS:
            support_order = plan[protocol][target.target_id][str(seed)][
                "support_order"
            ]
            for known in SELECTION_KS:
                selected = support_order[:known]
                groups: dict[str, list[int]] = defaultdict(list)
                for parent in selected:
                    groups[series_of_parent[parent]].append(lookup[parent])
                if len(groups) < 2:
                    continue
                all_codes = [lookup[parent] for parent in selected]
                for held_codes in groups.values():
                    held = set(held_codes)
                    references = [code for code in all_codes if code not in held]
                    if references:
                        tasks.append(
                            SurrogateTask(
                                protocol=protocol,
                                seed=seed,
                                known=known,
                                held_codes=held_codes,
                                reference_codes=references,
                            )
                        )
    if not tasks:
        raise ValueError(f"{target.target_id} has no valid surrogate tasks")
    positive_codes = sorted(
        {code for task in tasks for code in task.reference_codes}
    )
    pose_codes = torch.repeat_interleave(
        torch.arange(
            len(target.parent_ids), dtype=torch.int64, device=target.embeddings.device
        ),
        torch.as_tensor(
            target.parent_counts, dtype=torch.int64, device=target.embeddings.device
        ),
    )
    return PreparedTarget(
        target=target,
        pose_codes=pose_codes,
        centers=molecule_centers(target),
        tasks=tasks,
        positive_codes=positive_codes,
    )


def standardize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2 or float(values.std().cpu()) < 1e-8:
        return torch.zeros_like(values)
    return zscore(values)


@torch.no_grad()
def score_target(prepared: PreparedTarget, params: dict) -> list[dict]:
    target = prepared.target
    embeddings = target.embeddings
    device = embeddings.device
    n_parents = len(target.parent_ids)
    graph_indices, graph_weights = build_graph(
        prepared.centers,
        int(params["graph_neighbors"]),
        float(params["graph_temperature"]),
        float(params["graph_indegree_exponent"]),
        str(params.get("graph_search_precision", "fp32")),
    )
    positive_column = {
        code: column for column, code in enumerate(prepared.positive_codes)
    }
    graph_seeds = torch.zeros(
        (n_parents, len(prepared.positive_codes)),
        dtype=embeddings.dtype,
        device=device,
    )
    graph_seeds[
        prepared.positive_codes,
        torch.arange(len(prepared.positive_codes), device=device),
    ] = 1.0
    graph_responses = propagate_graph(
        graph_seeds,
        graph_indices,
        graph_weights,
        float(params["graph_restart_probability"]),
        int(params["graph_propagation_steps"]),
    )

    max_conformers = int(params["max_reference_conformers"])
    fold_rows = []
    for task in prepared.tasks:
        references = task.reference_codes
        pool_mask = np.ones(n_parents, dtype=bool)
        pool_mask[references] = False
        pool_mask_t = torch.as_tensor(pool_mask, device=device)
        library_rows = torch.nonzero(
            pool_mask_t[prepared.pose_codes], as_tuple=False
        ).flatten()
        library_codes = prepared.pose_codes[library_rows]

        support_rows = []
        for code in references:
            begin = int(target.parent_starts[code])
            support_rows.extend(
                range(
                    begin,
                    begin
                    + min(max_conformers, int(target.parent_counts[code])),
                )
            )
        support_rows_t = torch.as_tensor(support_rows, device=device)
        similarity = embeddings[library_rows] @ embeddings[support_rows_t].T
        support_k = min(
            int(params["support_similarity_topk"]), similarity.shape[1]
        )
        pose_similarity = torch.topk(
            similarity, k=support_k, dim=1
        ).values.mean(1)
        direct = standardize(
            pool_by_parent(
                pose_similarity, library_codes, pool_mask, n_parents
            )
        )

        columns = [positive_column[code] for code in references]
        graph = standardize(
            graph_responses[pool_mask_t][:, columns].mean(1)
        )
        full = direct + float(params["graph_weight"]) * graph
        held = set(task.held_codes)
        labels = np.asarray(
            [int(code in held) for code in np.flatnonzero(pool_mask)],
            dtype=np.int8,
        )
        fold_rows.extend(
            {
                "target_id": target.target_id,
                "protocol": task.protocol,
                "seed": task.seed,
                "K": task.known,
                "method": method,
                "surrogate_ef": ef_at_fraction(labels, score),
            }
            for method, score in (
                ("full", full.cpu().numpy()),
                ("direct_only", direct.cpu().numpy()),
                ("graph_only", graph.cpu().numpy()),
            )
        )

    frame = pd.DataFrame(fold_rows)
    return (
        frame.groupby(
            ["target_id", "protocol", "seed", "K", "method"],
            as_index=False,
        )
        .surrogate_ef.mean()
        .to_dict(orient="records")
    )


def aggregate(rows: list[dict]) -> tuple[dict, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    target_k = (
        frame.groupby(
            ["method", "protocol", "target_id", "K"], as_index=False
        )
        .surrogate_ef.mean()
    )
    per_k = (
        target_k.groupby(["method", "protocol", "K"], as_index=False)
        .surrogate_ef.mean()
    )
    per_protocol = (
        per_k.groupby(["method", "protocol"], as_index=False)
        .surrogate_ef.mean()
    )
    table = per_protocol.pivot(
        index="method", columns="protocol", values="surrogate_ef"
    )
    metrics = {}
    for method in ("full", "direct_only", "graph_only"):
        for protocol in PROTOCOLS:
            metrics[f"{method}_{protocol}"] = float(table.loc[method, protocol])
        metrics[f"{method}_overall"] = float(
            np.mean([metrics[f"{method}_{protocol}"] for protocol in PROTOCOLS])
        )
    for protocol in (*PROTOCOLS, "overall"):
        metrics[f"graph_contribution_{protocol}"] = (
            metrics[f"full_{protocol}"] - metrics[f"direct_only_{protocol}"]
        )
        metrics[f"direct_contribution_{protocol}"] = (
            metrics[f"full_{protocol}"] - metrics[f"graph_only_{protocol}"]
        )
    contributions = [
        metrics[f"{module}_contribution_{protocol}"]
        for module in ("direct", "graph")
        for protocol in PROTOCOLS
    ]
    metrics["minimum_protocol_module_contribution"] = float(min(contributions))
    metrics["negative_contribution_shortfall"] = float(
        sum(max(0.0, -value) for value in contributions)
    )
    metrics["selection_objective"] = float(
        metrics["full_overall"]
        - NEGATIVE_CONTRIBUTION_PENALTY
        * metrics["negative_contribution_shortfall"]
    )
    return metrics, per_k


def evaluate_params(prepared_targets, params):
    rows = []
    for prepared in prepared_targets:
        rows.extend(score_target(prepared, params))
    metrics, per_k = aggregate(rows)
    return rows, metrics, per_k


def target_paired_module_statistics(frame: pd.DataFrame) -> list[dict]:
    target = (
        frame.groupby(["method", "target_id", "protocol"], as_index=False)
        .surrogate_ef.mean()
    )
    full = target[target.method == "full"]
    rows = []
    for module, ablation in (
        ("direct", "graph_only"),
        ("graph", "direct_only"),
    ):
        reduced = target[target.method == ablation]
        paired = full.merge(
            reduced,
            on=["target_id", "protocol"],
            suffixes=("_full", "_reduced"),
        )
        paired["delta"] = paired.surrogate_ef_full - paired.surrogate_ef_reduced
        for protocol in (*PROTOCOLS, "combined"):
            if protocol == "combined":
                values = paired.groupby("target_id").delta.mean().to_numpy()
            else:
                values = paired[paired.protocol == protocol].delta.to_numpy()
            mean, low, high = paired_bootstrap(
                values, seed=REPRODUCIBILITY_SEED
            )
            rows.append(
                {
                    "module": module,
                    "protocol": protocol,
                    "targets": len(values),
                    "mean_contribution": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((values > 1e-12).sum()),
                    "ties": int((np.abs(values) <= 1e-12).sum()),
                    "losses": int((values < -1e-12).sum()),
                }
            )
    return rows


def density_free_params(
    params: dict, graph_search_precision: str = "fp32"
) -> dict:
    keys = (
        "max_reference_conformers",
        "support_similarity_topk",
        "graph_neighbors",
        "graph_temperature",
        "graph_indegree_exponent",
        "graph_restart_probability",
        "graph_propagation_steps",
        "graph_weight",
    )
    selected = {key: params[key] for key in keys}
    selected["graph_search_precision"] = graph_search_precision
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--graph-search-precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="KNN search precision; fp32 preserves the original selection run.",
    )
    args = parser.parse_args()
    configure_determinism()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_space_path = ROOT / "density_free_search_space.json"
    search_space = json.loads(search_space_path.read_text())
    data_config = json.loads(args.data_config.read_text())
    spec = next(
        item for item in data_config["datasets"] if item["name"] == "dekois2"
    )
    cache_path = Path(spec["cache"])
    whitener_path = Path(spec["whitener"])
    positive_manifest_path = Path(spec["positive_manifest"])
    run_spec = {
        "protocol_version": f"{PROTOCOL_VERSION}-density-free-v1",
        "study_name": STUDY_NAME,
        "trial_budget": TRIAL_BUDGET,
        "selection_boundary": (
            "DEKOIS only; K selected positives; one selected series held for "
            "surrogate scoring; cached labels unopened; external benchmarks unopened"
        ),
        "selection_ks": SELECTION_KS,
        "seeds": SEEDS,
        "protocols": list(PROTOCOLS),
        "negative_contribution_penalty": NEGATIVE_CONTRIBUTION_PENALTY,
        "graph_search_precision": args.graph_search_precision,
        "sampler": {
            "name": "TPESampler",
            "seed": REPRODUCIBILITY_SEED,
            "multivariate": True,
            "n_startup_trials": 16,
        },
        "data_config": str(args.data_config.resolve()),
        "data_config_sha256": sha256(args.data_config),
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha256(cache_path),
        "whitener": str(whitener_path.resolve()),
        "whitener_sha256": sha256(whitener_path),
        "positive_manifest": str(positive_manifest_path.resolve()),
        "positive_manifest_sha256": sha256(positive_manifest_path),
        "search_space_sha256": sha256(search_space_path),
        "search_script_sha256": sha256(Path(__file__)),
        "source_sha256": source_sha256(ROOT),
    }
    spec_path = args.output_dir / "run_spec.json"
    database_path = args.output_dir / "study.sqlite3"
    existing = list(args.output_dir.iterdir())
    if existing and not spec_path.exists():
        raise RuntimeError(
            f"refusing to reuse non-empty output directory without {spec_path.name}"
        )
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != run_spec:
            raise RuntimeError(
                "search inputs or source changed; use a new output directory"
            )
    else:
        write_json(spec_path, run_spec)

    device = torch.device(args.device)
    cache = EmbeddingCache(cache_path)
    projection = load_projection(whitener_path, device)
    manifest = read_positive_manifest(positive_manifest_path)
    plan = build_episode_plan(
        manifest, list(PROTOCOLS), SEEDS, max(SELECTION_KS)
    )
    target_ids = sorted(
        (set(plan["fixed_random"]) | set(plan["series"]))
        & set(cache.target_ids)
    )
    expected_targets = int(spec["expected_targets"])
    if len(target_ids) != expected_targets:
        raise RuntimeError(
            f"expected {expected_targets} DEKOIS targets, found {len(target_ids)}"
        )
    series = {
        target_id: series_groups(manifest[target_id])
        for target_id in target_ids
    }
    prepared_targets = []
    for index, target_id in enumerate(target_ids, 1):
        target = cache.get(
            target_id,
            device,
            projection=projection,
            include_labels=False,
        )
        prepared_targets.append(
            prepare_target(target, series[target_id], plan)
        )
        print(f"prepared [{index}/{len(target_ids)}] {target_id}", flush=True)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{database_path.resolve()}",
        heartbeat_interval=60,
        grace_period=180,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=REPRODUCIBILITY_SEED,
            multivariate=True,
            n_startup_trials=16,
        ),
    )

    def objective(trial: optuna.Trial):
        params = {
            name: suggest(trial, name, spec)
            for name, spec in search_space.items()
        }
        params["graph_search_precision"] = args.graph_search_precision
        _, metrics, _ = evaluate_params(prepared_targets, params)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["selection_objective"]

    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    if completed_before > TRIAL_BUDGET:
        raise RuntimeError(
            f"study has {completed_before} completed trials; budget is "
            f"{TRIAL_BUDGET}"
        )

    def progress(study, trial):
        completed = sum(
            item.state == optuna.trial.TrialState.COMPLETE
            for item in study.trials
        )
        print(
            f"trial={trial.number} completed={completed}/{TRIAL_BUDGET} "
            f"objective={trial.value:.6f} "
            f"full={trial.user_attrs['full_overall']:.6f} "
            "min_contribution="
            f"{trial.user_attrs['minimum_protocol_module_contribution']:.6f}",
            flush=True,
        )

    completed_now = completed_before
    while completed_now < TRIAL_BUDGET:
        before = completed_now
        study.optimize(
            objective,
            n_trials=TRIAL_BUDGET - completed_now,
            callbacks=[progress],
            show_progress_bar=False,
        )
        completed_now = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        if completed_now == before:
            raise RuntimeError("search made no progress toward successful trials")
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) != TRIAL_BUDGET:
        raise RuntimeError(
            f"search ended with {len(completed)}/{TRIAL_BUDGET} completed trials"
        )

    trial_rows = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            **trial.user_attrs,
            **trial.params,
        }
        for trial in study.trials
    ]
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.output_dir / "trials.csv", index=False)
    feasible = trials[
        (trials.state == "COMPLETE")
        & (trials.minimum_protocol_module_contribution >= 0.0)
    ]
    if len(feasible):
        selected_row = feasible.sort_values(
            ["full_overall", "selection_objective"], ascending=False
        ).iloc[0]
        selection_status = "all module contributions nonnegative in both protocols"
    else:
        selected_row = trials[trials.state == "COMPLETE"].sort_values(
            [
                "minimum_protocol_module_contribution",
                "full_overall",
            ],
            ascending=False,
        ).iloc[0]
        selection_status = (
            "no fully feasible trial; selected maximum minimum module contribution"
        )
    selected_trial = study.trials[int(selected_row.number)]
    selected_params = density_free_params(
        selected_trial.params, args.graph_search_precision
    )
    best_full_row = trials[trials.state == "COMPLETE"].sort_values(
        "full_overall", ascending=False
    ).iloc[0]
    best_full_trial = study.trials[int(best_full_row.number)]

    selected_rows, selected_metrics, selected_per_k = evaluate_params(
        prepared_targets, selected_params
    )
    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(
        args.output_dir / "selected_conditions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    selected_per_k.to_csv(
        args.output_dir / "selected_per_k_protocol.csv", index=False
    )
    module_rows = target_paired_module_statistics(selected_frame)
    pd.DataFrame(module_rows).to_csv(
        args.output_dir / "selected_module_contributions.csv", index=False
    )

    result = {
        "protocol_version": run_spec["protocol_version"],
        "selection_boundary": run_spec["selection_boundary"],
        "selection_status": selection_status,
        "targets": len(target_ids),
        "selection_ks": SELECTION_KS,
        "deployment_ks": list(range(1, 11)),
        "k1_note": (
            "The reference-only held-series surrogate is undefined at K=1; "
            "the frozen global theta transfers to K=1."
        ),
        "seeds": SEEDS,
        "completed_trials": len(completed),
        "selected_trial": int(selected_trial.number),
        "selected_objective": float(selected_row.selection_objective),
        "selected_params": selected_params,
        "selected_metrics": selected_metrics,
        "unconstrained_best_full_trial": int(best_full_trial.number),
        "unconstrained_best_full_params": density_free_params(
            best_full_trial.params, args.graph_search_precision
        ),
        "unconstrained_best_full_metrics": {
            key: value
            for key, value in best_full_trial.user_attrs.items()
            if key.startswith(("full_", "direct_", "graph_", "minimum_"))
        },
        "module_contributions": module_rows,
        "provenance": {
            **run_spec,
            "software": software_versions(include_optuna=True),
            "device": str(device),
            "study_database": str(database_path.resolve()),
        },
    }
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "theta_density_free.json", selected_params)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
