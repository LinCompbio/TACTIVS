import csv
import gzip

import numpy as np
import torch

from tactivs.cache import TargetData
from tactivs.core import ScoreResult
from tactivs.result_writer import MoleculeScoreWriter


def test_episode_writer_records_every_parent_and_role(tmp_path):
    target = TargetData(
        target_id="t",
        embeddings=torch.eye(4),
        parent_ids=np.asarray(["a1", "a2", "a3", "d1"]),
        parent_starts=np.arange(4),
        parent_counts=np.ones(4, dtype=np.int64),
        parent_labels=np.asarray([1, 1, 1, 0], dtype=np.int8),
    )
    result = ScoreResult(
        parent_ids=np.asarray(["a3", "d1"]),
        scores=np.asarray([2.0, 1.0]),
        similarity=np.asarray([0.8, 0.2]),
        direct=np.asarray([1.2, -1.2]),
        graph=np.asarray([0.6, 0.4]),
    )
    visible = np.asarray([True, False, True, True])
    ranked = np.asarray([False, False, True, True])

    paths = [tmp_path / "first.csv.gz", tmp_path / "second.csv.gz"]
    for path in paths:
        with MoleculeScoreWriter(path) as writer:
            writer.write_episode(
                "dekois2",
                "selected_theta",
                target,
                "series",
                0,
                1,
                ["a1"],
                visible,
                ranked,
                result,
            )

    assert paths[0].read_bytes() == paths[1].read_bytes()
    with gzip.open(paths[0], mode="rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["role"] for row in rows] == [
        "support",
        "series_excluded",
        "candidate",
        "candidate",
    ]
    assert [row["score"] for row in rows] == ["", "", "2.0", "1.0"]
