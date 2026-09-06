import numpy as np
import pytest
import torch

from tactivs.cache import EmbeddingCache, load_projection
from tactivs.encoding import build_embedding_cache
from tactivs.reference_pool import MoleculeRecord
from tactivs.whitening import fit_whitener


class FakeEncoder:
    def encode(self, conformers):
        return torch.tensor(
            [
                [float(molecule.GetNumAtoms()), float(index % 3), float(index)]
                for index, molecule in enumerate(conformers)
            ]
        )


def molecule_records():
    return [
        MoleculeRecord("molecule_a", "CCO"),
        MoleculeRecord("molecule_b", "CCN"),
        MoleculeRecord("molecule_c", "c1ccccc1"),
        MoleculeRecord("molecule_d", "CC(=O)O"),
    ]


def test_build_user_cache(tmp_path):
    output = tmp_path / "cache"
    summary = build_embedding_cache(
        molecule_records(),
        FakeEncoder(),
        output,
        target_id="my_target",
        conformers_per_molecule=2,
        molecule_batch_size=2,
    )

    target = EmbeddingCache(output).get("my_target", torch.device("cpu"))
    assert target.parent_ids.tolist() == [
        "molecule_a",
        "molecule_b",
        "molecule_c",
        "molecule_d",
    ]
    assert target.parent_counts.tolist() == [2, 2, 2, 2]
    assert target.embeddings.shape == (8, 3)
    assert summary == {"molecules": 4, "conformers": 8, "shards": 2}
    with pytest.raises(ValueError, match="does not contain activity labels"):
        EmbeddingCache(output).get_parent_labels("my_target")


def test_fit_user_whitener(tmp_path):
    cache_path = tmp_path / "cache"
    build_embedding_cache(
        molecule_records(),
        FakeEncoder(),
        cache_path,
        target_id="my_target",
        conformers_per_molecule=2,
    )
    whitener_path = tmp_path / "whitener.npz"
    summary = fit_whitener(cache_path, whitener_path, molecule_batch_size=2)

    projection = load_projection(whitener_path, torch.device("cpu"))
    assert projection.shape == (3, 3)
    assert torch.isfinite(projection).all()
    assert summary == {"targets": 1, "molecules": 4, "dimension": 3}
    with np.load(whitener_path) as data:
        assert "mean" not in data
