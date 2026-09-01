import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tactivs.cache import TargetData
from tactivs.core import build_graph, prepare_graph, score_library
from tactivs.evaluation import score_episodes

PARAMS = {
    "max_reference_conformers": 2,
    "support_similarity_topk": 2,
    "graph_neighbors": 2,
    "graph_temperature": 0.1,
    "graph_indegree_exponent": 0.5,
    "graph_restart_probability": 0.2,
    "graph_propagation_steps": 3,
    "graph_weight": 0.1,
    "graph_search_precision": "fp16",
}


def synthetic_target():
    generator = torch.Generator().manual_seed(7)
    embeddings = F.normalize(torch.randn(8, 6, generator=generator), dim=1)
    return TargetData(
        target_id="synthetic",
        embeddings=embeddings,
        parent_ids=np.asarray(["ref", "a", "b", "c"]),
        parent_starts=np.asarray([0, 2, 4, 6]),
        parent_counts=np.asarray([2, 2, 2, 2]),
    )


def test_inference_is_deterministic_and_excludes_reference():
    target = synthetic_target()
    first = score_library(target, ["ref"], PARAMS)
    second = score_library(target, ["ref"], PARAMS)
    assert first.parent_ids.tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(first.scores, second.scores)
    assert np.isfinite(first.scores).all()


def test_chunked_similarity_matches_single_chunk():
    target = synthetic_target()
    single = score_library(target, ["ref"], PARAMS, query_chunk=100)
    chunked = score_library(target, ["ref"], PARAMS, query_chunk=1)
    np.testing.assert_allclose(chunked.similarity, single.similarity, atol=1e-7)
    np.testing.assert_allclose(chunked.direct, single.direct, atol=1e-7)
    np.testing.assert_allclose(chunked.scores, single.scores, atol=1e-7)


def test_precomputed_graph_matches_inline_graph():
    target = synthetic_target()
    inline = score_library(target, ["ref"], PARAMS)
    reused = score_library(target, ["ref"], PARAMS, graph=prepare_graph(target, PARAMS))
    np.testing.assert_allclose(reused.scores, inline.scores, atol=1e-7)


def test_empty_references_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        score_library(synthetic_target(), [], PARAMS)


def test_graph_weights_use_direct_temperature_scaled_similarity():
    centers = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
            ]
        ),
        dim=1,
    )
    neighbors, weights = build_graph(
        centers, requested_k=2, temperature=0.25, indegree_exp=0.0
    )
    similarities = centers @ centers.T
    expected = torch.softmax(
        torch.gather(similarities, 1, neighbors) / 0.25,
        dim=1,
    )
    torch.testing.assert_close(weights, expected)


def test_fp16_graph_search_falls_back_to_fp32_off_cuda():
    centers = F.normalize(
        torch.randn(8, 6, generator=torch.Generator().manual_seed(3)), dim=1
    )
    fp32 = build_graph(centers, 3, 0.2, 0.5, search_precision="fp32")
    fp16 = build_graph(centers, 3, 0.2, 0.5, search_precision="fp16")
    torch.testing.assert_close(fp16[0], fp32[0])
    torch.testing.assert_close(fp16[1], fp32[1])


def test_unknown_graph_search_precision_is_rejected():
    centers = F.normalize(torch.randn(3, 2), dim=1)
    with pytest.raises(ValueError, match="graph search precision"):
        build_graph(centers, 2, 0.2, 0.5, search_precision="bf16")


def test_fixed_random_ef_library_contains_every_non_support_parent():
    generator = torch.Generator().manual_seed(11)
    target = TargetData(
        target_id="episode",
        embeddings=F.normalize(torch.randn(10, 6, generator=generator), dim=1),
        parent_ids=np.asarray(
            [f"a{i}" for i in range(5)] + [f"d{i}" for i in range(5)]
        ),
        parent_starts=np.arange(10),
        parent_counts=np.ones(10, dtype=np.int64),
        parent_labels=np.asarray([1] * 5 + [0] * 5, dtype=np.int8),
    )
    plan = {
        "fixed_random": {
            "episode": {"0": {"support_order": [f"a{i}" for i in range(5)]}}
        }
    }

    row = score_episodes(
        target,
        plan,
        ks=[3],
        seeds=[0],
        params=PARAMS,
        protocols=["fixed_random"],
    )[0]

    assert row["support_count"] == 3
    assert row["visible_count"] == 10
    assert row["candidate_count"] == 7
    assert row["positive_count"] == 2
    assert row["negative_count"] == 5
