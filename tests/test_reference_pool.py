import numpy as np
import torch

from tactivs.cache import TargetData
from tactivs.reference_pool import ReferencePool


def target_data():
    raw = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
    )
    return TargetData(
        target_id="target",
        embeddings=raw,
        parent_ids=np.asarray(["active_a", "active_b", "inactive"]),
        parent_starts=np.asarray([0, 1, 2]),
        parent_counts=np.asarray([1, 1, 2]),
        parent_labels=np.asarray([1, 1, 0]),
        raw_embeddings=raw,
        projection=torch.eye(2),
    )


def test_benchmark_actives_form_reference_pool():
    target = target_data()
    pool = ReferencePool.sample_labeled(target, k=2, seed=0)
    materialized, reference_ids = pool.materialize(target, target.projection)

    assert set(reference_ids) == {"active_a", "active_b"}
    assert materialized.parent_labels is None
    assert materialized.parent_ids.tolist() == target.parent_ids.tolist()


def test_external_actives_are_added_before_whitening():
    target = target_data()
    pool = ReferencePool(
        parent_ids=np.asarray(["external_active"]),
        parent_counts=np.asarray([1]),
        raw_embeddings=torch.tensor([[0.9, 0.1]]),
        source="external_actives",
    )
    materialized, reference_ids = pool.materialize(target, torch.eye(2))

    assert reference_ids == ["external_active"]
    assert materialized.parent_ids.tolist()[-1] == "external_active"
    assert materialized.parent_counts.tolist() == [1, 1, 2, 1]
    assert materialized.embeddings.shape == (5, 2)
