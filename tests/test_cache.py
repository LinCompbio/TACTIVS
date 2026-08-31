import numpy as np
import torch

from tactivs.cache import EmbeddingCache, external_whiten, subset_target


def test_cache_does_not_load_labels_unless_requested(tmp_path):
    np.save(tmp_path / "embeddings_000000.npy", np.eye(4, dtype=np.float32))
    np.savez_compressed(
        tmp_path / "target_manifest.npz",
        target_ids=np.asarray(["t"]),
        target_starts=np.asarray([0]),
        target_ends=np.asarray([4]),
        parent_offsets=np.asarray([0, 2]),
        parent_starts=np.asarray([0, 2]),
        parent_counts=np.asarray([2, 2]),
        parent_labels=np.asarray([1, 0]),
    )
    np.savez_compressed(
        tmp_path / "target_parent_ids.npz",
        target_ids=np.asarray(["t"]),
        parent_offsets=np.asarray([0, 2]),
        parent_ids=np.asarray(["p1", "p2"]),
    )
    cache = EmbeddingCache(tmp_path)
    unlabeled = cache.get("t", torch.device("cpu"), include_labels=False)
    labeled = cache.get("t", torch.device("cpu"), include_labels=True)
    assert unlabeled.parent_labels is None
    assert labeled.parent_labels.tolist() == [1, 0]


def test_subset_target_recomputes_episode_local_centering(tmp_path):
    raw = np.asarray([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=np.float32)
    np.save(tmp_path / "embeddings_000000.npy", raw)
    np.savez_compressed(
        tmp_path / "target_manifest.npz",
        target_ids=np.asarray(["t"]),
        target_starts=np.asarray([0]),
        target_ends=np.asarray([3]),
        parent_offsets=np.asarray([0, 3]),
        parent_starts=np.asarray([0, 1, 2]),
        parent_counts=np.asarray([1, 1, 1]),
        parent_labels=np.asarray([1, 0, 0]),
    )
    np.savez_compressed(
        tmp_path / "target_parent_ids.npz",
        target_ids=np.asarray(["t"]),
        parent_offsets=np.asarray([0, 3]),
        parent_ids=np.asarray(["p1", "p2", "p3"]),
    )
    projection = torch.eye(2)
    target = EmbeddingCache(tmp_path).get(
        "t",
        torch.device("cpu"),
        projection=projection,
        include_labels=True,
        retain_raw=True,
    )
    subset = subset_target(target, np.asarray([True, True, False]))

    expected = external_whiten(
        torch.from_numpy(raw[:2]), projection, np.asarray([1, 1])
    )
    torch.testing.assert_close(subset.embeddings, expected)
    assert subset.parent_ids.tolist() == ["p1", "p2"]
