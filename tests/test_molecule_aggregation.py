import numpy as np
import torch
import torch.nn.functional as F

from tactivs.cache import TargetData, molecule_balanced_mean, molecule_means
from tactivs.core import molecule_centers


def test_molecule_aggregation_reuses_conformer_means() -> None:
    raw = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
    counts = np.asarray([2, 1], dtype=np.int64)
    expected = torch.tensor([[2.0, 1.0], [0.0, 2.0]])
    target = TargetData(
        target_id="test",
        embeddings=raw,
        parent_ids=np.asarray(["a", "b"]),
        parent_starts=np.asarray([0, 2]),
        parent_counts=counts,
    )

    assert torch.equal(molecule_means(raw, counts), expected)
    assert torch.equal(molecule_balanced_mean(raw, counts), expected.mean(0))
    assert torch.equal(molecule_centers(target), F.normalize(expected, dim=1))
