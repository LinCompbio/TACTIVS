"""Label-free TACTIVS scoring primitives and single-target inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .cache import TargetData

Graph = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ScoreResult:
    parent_ids: np.ndarray
    scores: np.ndarray
    similarity: np.ndarray
    direct: np.ndarray
    graph: np.ndarray


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std() + 1e-6)


@torch.no_grad()
def molecule_centers(target: TargetData) -> torch.Tensor:
    counts = torch.as_tensor(
        target.parent_counts,
        dtype=target.embeddings.dtype,
        device=target.embeddings.device,
    )
    codes = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int64, device=counts.device),
        counts.to(torch.int64),
    )
    centers = torch.zeros(
        (len(counts), target.embeddings.shape[1]),
        dtype=target.embeddings.dtype,
        device=target.embeddings.device,
    )
    centers.index_add_(0, codes, target.embeddings)
    return F.normalize(centers / counts[:, None], dim=1)


@torch.no_grad()
def build_graph(
    centers,
    requested_k: int,
    temperature: float,
    indegree_exp: float,
    search_precision: str = "fp32",
):
    """Build the KNN graph, optionally using native CUDA FP16 for neighbor search.

    Only the dense similarity products and top-k selection use FP16. Selected
    similarities are promoted to FP32 before edge weighting and propagation.
    """
    if search_precision not in {"fp16", "fp32"}:
        raise ValueError("graph search precision must be 'fp16' or 'fp32'")
    n = centers.shape[0]
    k = min(requested_k, n - 1)
    use_fp16 = search_precision == "fp16" and centers.device.type == "cuda"
    search_centers = centers.to(torch.float16) if use_fp16 else centers
    value_dtype = torch.float32 if use_fp16 else centers.dtype
    indices = torch.empty((n, k), dtype=torch.int64, device=centers.device)
    values = torch.empty((n, k), dtype=value_dtype, device=centers.device)
    for begin in range(0, n, 1024):
        end = min(begin + 1024, n)
        similarity = search_centers[begin:end] @ search_centers.T
        rows = torch.arange(end - begin, device=centers.device)
        similarity[rows, begin + rows] = -float("inf")
        top = torch.topk(similarity, k=k, dim=1)
        indices[begin:end] = top.indices
        values[begin:end] = top.values.to(value_dtype)
    indegree = (
        torch.bincount(indices.flatten(), minlength=n).to(values.dtype).clamp_min(1)
    )
    logits = values / temperature
    weights = torch.exp(logits) / indegree[indices].pow(indegree_exp)
    weights /= weights.sum(1, keepdim=True).clamp_min(1e-12)
    return indices, weights


@torch.no_grad()
def prepare_graph(target: TargetData, params: dict) -> Graph:
    """Build the target graph once so repeated support episodes can reuse it."""
    return build_graph(
        molecule_centers(target),
        int(params["graph_neighbors"]),
        float(params["graph_temperature"]),
        float(params["graph_indegree_exponent"]),
        str(params.get("graph_search_precision", "fp32")),
    )


@torch.no_grad()
def propagate_graph(seeds, neighbors, weights, restart: float, steps: int):
    state = seeds
    for _ in range(steps):
        state = restart * seeds + (1.0 - restart) * (
            weights[..., None] * state[neighbors]
        ).sum(1)
    return state


def pool_by_parent(scores, pose_codes, parent_mask, n_parents):
    pooled = torch.full(
        (n_parents,), -float("inf"), dtype=scores.dtype, device=scores.device
    )
    pooled.scatter_reduce_(
        0, pose_codes.to(torch.int64), scores, reduce="amax", include_self=True
    )
    return pooled[torch.as_tensor(parent_mask, device=scores.device)]


def topk_mean_similarity(
    queries: torch.Tensor,
    references: torch.Tensor,
    k: int,
    query_chunk: int,
) -> torch.Tensor:
    """Compute mean top-k similarities without materializing the full matrix."""
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    output = torch.empty(len(queries), dtype=queries.dtype, device=queries.device)
    for begin in range(0, len(queries), query_chunk):
        end = min(begin + query_chunk, len(queries))
        similarity = queries[begin:end] @ references.T
        output[begin:end] = torch.topk(similarity, k=k, dim=1).values.mean(1)
    return output


@torch.no_grad()
def score_library(
    target: TargetData,
    reference_ids: list[str],
    params: dict,
    query_chunk: int = 32768,
    graph: Graph | None = None,
) -> ScoreResult:
    """Rank an unlabeled target library from K known positive references."""
    if not reference_ids:
        raise ValueError("at least one positive reference is required")
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("reference IDs must be unique")
    lookup = {str(parent): index for index, parent in enumerate(target.parent_ids)}
    missing = [parent for parent in reference_ids if parent not in lookup]
    if missing:
        raise KeyError(f"reference IDs absent from target cache: {missing}")
    support_codes = [lookup[parent] for parent in reference_ids]
    n_parents = len(target.parent_ids)
    device = target.embeddings.device
    pose_codes = torch.repeat_interleave(
        torch.arange(n_parents, dtype=torch.int32, device=device),
        torch.as_tensor(target.parent_counts, dtype=torch.int64, device=device),
    )

    query_mask = np.ones(n_parents, dtype=bool)
    query_mask[support_codes] = False
    if not query_mask.any():
        raise ValueError("query library is empty after removing references")

    graph_indices, graph_weights = graph or prepare_graph(target, params)
    seed = torch.zeros((n_parents, 1), dtype=target.embeddings.dtype, device=device)
    seed[support_codes, 0] = 1.0 / len(support_codes)
    graph_state = propagate_graph(
        seed,
        graph_indices,
        graph_weights,
        float(params["graph_restart_probability"]),
        int(params["graph_propagation_steps"]),
    )[:, 0]

    support_rows = []
    max_conformers = int(params["max_reference_conformers"])
    for code in support_codes:
        begin = int(target.parent_starts[code])
        support_rows.extend(
            range(begin, begin + min(max_conformers, int(target.parent_counts[code])))
        )
    support_rows_t = torch.as_tensor(support_rows, device=device)
    query_mask_t = torch.as_tensor(query_mask, device=device)
    library_rows = torch.nonzero(
        query_mask_t[pose_codes.to(torch.int64)], as_tuple=False
    ).flatten()
    library_codes = pose_codes[library_rows]
    support_k = min(int(params["support_similarity_topk"]), len(support_rows_t))
    pose_similarity = topk_mean_similarity(
        target.embeddings[library_rows],
        target.embeddings[support_rows_t],
        support_k,
        query_chunk,
    )
    pose_score = float(params.get("similarity_weight", 1.0)) * zscore(pose_similarity)
    molecule_score = zscore(
        pool_by_parent(pose_score, library_codes, query_mask, n_parents)
    )
    molecule_similarity = pool_by_parent(
        pose_similarity, library_codes, query_mask, n_parents
    )
    graph = zscore(graph_state[query_mask_t])
    final = molecule_score + float(params["graph_weight"]) * graph
    return ScoreResult(
        parent_ids=target.parent_ids[query_mask],
        scores=final.cpu().numpy(),
        similarity=molecule_similarity.cpu().numpy(),
        direct=molecule_score.cpu().numpy(),
        graph=graph.cpu().numpy(),
    )
