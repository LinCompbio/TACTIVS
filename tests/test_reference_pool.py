import numpy as np
import torch

from tactivs.cache import TargetData, external_whiten
from tactivs.reference_pool import (
    ReferencePool,
    build_uploaded_pool,
    read_uploaded_molecules,
)


def target_with_labels():
    raw = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]]
    )
    counts = np.ones(4, dtype=np.int64)
    return TargetData(
        target_id="target",
        embeddings=external_whiten(raw, torch.eye(2), counts),
        parent_ids=np.asarray(["a1", "d1", "a2", "a3"]),
        parent_starts=np.arange(4),
        parent_counts=counts,
        parent_labels=np.asarray([1, 0, 1, 1], dtype=np.int8),
        raw_embeddings=raw,
        projection=torch.eye(2),
    )


def test_labeled_reference_pool_is_nested_and_uses_only_actives():
    target = target_with_labels()
    pool_two = ReferencePool.sample_labeled(target, 2, seed=7)
    pool_three = ReferencePool.sample_labeled(target, 3, seed=7)
    assert pool_two.parent_ids.tolist() == pool_three.parent_ids[:2].tolist()
    assert set(pool_three.parent_ids) == {"a1", "a2", "a3"}
    materialized, reference_ids = pool_two.materialize(target, torch.eye(2))
    assert materialized.parent_labels is None
    torch.testing.assert_close(materialized.embeddings, target.embeddings)
    assert reference_ids == pool_two.parent_ids.tolist()


def test_external_pool_is_appended_before_target_local_calibration(tmp_path):
    target = target_with_labels()
    pool = ReferencePool(
        parent_ids=np.asarray(["external"]),
        parent_counts=np.asarray([2]),
        raw_embeddings=torch.tensor([[1.0, 1.0], [1.5, 0.5]]),
        source="uploaded_molecules",
        canonical_smiles=np.asarray(["CC"]),
    )
    artifact = tmp_path / "pool.npz"
    pool.save(artifact)
    loaded = ReferencePool.load(artifact, torch.device("cpu"))
    combined, reference_ids = loaded.materialize(target, torch.eye(2))

    expected_raw = torch.cat((target.raw_embeddings, pool.raw_embeddings), dim=0)
    expected_counts = np.asarray([1, 1, 1, 1, 2])
    expected = external_whiten(expected_raw, torch.eye(2), expected_counts)
    torch.testing.assert_close(combined.embeddings, expected)
    assert combined.parent_ids.tolist() == ["a1", "d1", "a2", "a3", "external"]
    assert reference_ids == ["external"]


def test_uploaded_csv_is_conformerized_and_encoded(tmp_path):
    path = tmp_path / "references.csv"
    path.write_text(
        "parent_molecule_id,smiles\n"
        "ref_a,CCO\n"
        "ref_b,c1ccccc1\n"
    )
    records = read_uploaded_molecules(path)

    class FakeEncoder:
        def encode(self, conformers):
            return torch.tensor(
                [[float(molecule.GetNumAtoms()), float(index)] for index, molecule in enumerate(conformers)]
            )

    pool = build_uploaded_pool(
        records,
        FakeEncoder(),
        conformers_per_molecule=2,
        seed=11,
    )
    assert pool.parent_ids.tolist() == ["ref_a", "ref_b"]
    assert pool.parent_counts.tolist() == [2, 2]
    assert pool.raw_embeddings.shape == (4, 2)
