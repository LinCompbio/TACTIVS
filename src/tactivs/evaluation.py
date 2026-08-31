"""Batched labelled evaluation, isolated from the inference and selection paths."""

from __future__ import annotations

import json

import numpy as np
import torch

from .cache import TargetData, subset_target
from .core import (
    ScoreResult,
    build_graph,
    molecule_centers,
    pool_by_parent,
    propagate_graph,
    score_library,
    zscore,
)
from .episodes import episode_parent_masks
from .metrics import screening_metrics
from .reference_pool import ReferencePool


@torch.no_grad()
def score_external_reference(
    target: TargetData, reference: torch.Tensor, params, return_scores: bool = False
):
    """Score a complete target library from one external reference molecule."""
    if target.parent_labels is None:
        raise ValueError("evaluation requires labels")
    device = target.embeddings.device
    embeddings = target.embeddings
    n_parents = len(target.parent_ids)
    pose_codes = torch.repeat_interleave(
        torch.arange(n_parents, dtype=torch.int32, device=device),
        torch.as_tensor(target.parent_counts, dtype=torch.int64, device=device),
    )
    reference = reference[: int(params["max_reference_conformers"])]
    if not len(reference):
        raise ValueError("external reference has no conformers")

    support_k = min(int(params["support_similarity_topk"]), reference.shape[0])
    pose_similarity = torch.topk(
        embeddings @ reference.T, k=support_k, dim=1
    ).values.mean(1)
    pose_score = float(params.get("similarity_weight", 1.0)) * zscore(
        pose_similarity
    )
    all_parents = np.ones(n_parents, dtype=bool)
    molecule_score = zscore(
        pool_by_parent(pose_score, pose_codes, all_parents, n_parents)
    )

    centers = molecule_centers(target)
    reference_center = torch.nn.functional.normalize(reference.mean(0), dim=0)
    graph_nodes = torch.cat((centers, reference_center[None, :]), dim=0)
    graph_indices, graph_weights = build_graph(
        graph_nodes,
        int(params["graph_neighbors"]),
        float(params["graph_temperature"]),
        float(params["graph_indegree_exponent"]),
        str(params.get("graph_search_precision", "fp32")),
    )
    graph_seed = torch.zeros(
        (len(graph_nodes), 1), dtype=embeddings.dtype, device=device
    )
    graph_seed[-1, 0] = 1.0
    graph = zscore(
        propagate_graph(
            graph_seed,
            graph_indices,
            graph_weights,
            float(params["graph_restart_probability"]),
            int(params["graph_propagation_steps"]),
        )[:-1, 0]
    )
    final = molecule_score + float(params["graph_weight"]) * graph
    summary = {
        "target_id": target.target_id,
        "protocol": "crystal_reference",
        "K": 1,
        "reference_conformers": int(reference.shape[0]),
        **screening_metrics(target.parent_labels, final.cpu().numpy()),
    }
    if not return_scores:
        return summary
    result = ScoreResult(
        parent_ids=target.parent_ids.copy(),
        scores=final.cpu().numpy(),
        similarity=pool_by_parent(
            pose_similarity, pose_codes, all_parents, n_parents
        )
        .cpu()
        .numpy(),
        direct=molecule_score.cpu().numpy(),
        graph=graph.cpu().numpy(),
    )
    return summary, result


@torch.no_grad()
def score_episodes(
    target: TargetData,
    plan,
    ks,
    seeds,
    params,
    protocols=("fixed_random", "series"),
    series_of_parent=None,
    score_callback=None,
):
    """Score runtime K-shot episodes with one observable-set boundary."""
    if target.parent_labels is None:
        raise ValueError("evaluation requires labels")
    rows = []
    for protocol in protocols:
        if target.target_id not in plan[protocol]:
            continue
        for seed in seeds:
            support_order = plan[protocol][target.target_id][str(seed)]["support_order"]
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
                pool = ReferencePool.from_cached_ids(
                    support_ids, source="benchmark_labeled_sample"
                )
                episode_target, support_ids = pool.materialize(
                    episode_target, episode_target.projection
                )
                result = score_library(episode_target, support_ids, params)
                expected_ids = target.parent_ids[ranked_mask]
                if not np.array_equal(
                    result.parent_ids.astype(str), expected_ids.astype(str)
                ):
                    raise RuntimeError(
                        "scored library differs from the episode metric set"
                    )
                metric_values = screening_metrics(labels, result.scores)
                if score_callback is not None:
                    score_callback(
                        target,
                        protocol,
                        seed,
                        known,
                        support_ids,
                        visible_mask,
                        ranked_mask,
                        result,
                    )
                rows.append(
                    {
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
                        **metric_values,
                    }
                )
    return rows
