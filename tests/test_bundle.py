import json

import numpy as np
import pytest

from tactivs.bundle import BUNDLE_SCHEMA, load_bundle_specs


def test_bundle_paths_are_resolved_and_validated(tmp_path):
    dataset = tmp_path / "datasets" / "litpcba"
    (dataset / "embeddings").mkdir(parents=True)
    np.save(dataset / "embeddings" / "embeddings_000000.npy", np.eye(2))
    np.savez_compressed(dataset / "index.npz", value=np.asarray([1]))
    np.savez_compressed(dataset / "whitener.npz", projection=np.eye(2))
    (dataset / "active_manifest.csv").write_text(
        "target_id,parent_molecule_id,canonical_smiles\n"
    )
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "datasets": [
                    {
                        "name": "litpcba",
                        "path": "datasets/litpcba",
                        "targets": 15,
                        "positives": 1,
                    }
                ],
            }
        )
    )

    spec = load_bundle_specs(tmp_path)[0]

    assert spec["name"] == "litpcba"
    assert spec["cache"] == dataset.resolve()


def test_bundle_rejects_parent_path_escape(tmp_path):
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "datasets": [
                    {
                        "name": "bad",
                        "path": "../outside",
                        "targets": 1,
                        "positives": 1,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="inside the bundle"):
        load_bundle_specs(tmp_path)
