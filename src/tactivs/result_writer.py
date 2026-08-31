"""Streaming, deterministic molecule-level result artifacts."""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import numpy as np

MOLECULE_SCORE_FIELDS = (
    "dataset",
    "config",
    "target_id",
    "protocol",
    "seed",
    "K",
    "parent_molecule_id",
    "label",
    "role",
    "score",
    "similarity",
    "direct",
    "graph",
)


class MoleculeScoreWriter:
    """Write every episode member without retaining molecule rows in memory."""

    def __init__(self, path: Path, *, compression: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("wb")
        self._gzip = (
            gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
            if compression
            else None
        )
        stream = self._gzip if self._gzip is not None else self._raw
        self._text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._text, fieldnames=MOLECULE_SCORE_FIELDS)
        self._writer.writeheader()
        self.row_count = 0

    def write_episode(
        self,
        dataset,
        config,
        target,
        protocol,
        seed,
        known,
        support_ids,
        visible_mask,
        ranked_mask,
        result,
    ) -> None:
        support = set(map(str, support_ids))
        candidate = 0
        for index, parent in enumerate(map(str, target.parent_ids)):
            if parent in support:
                role = "support"
            elif not visible_mask[index]:
                role = "series_excluded"
            else:
                role = "candidate"
                if not ranked_mask[index] or str(result.parent_ids[candidate]) != parent:
                    raise RuntimeError("molecule score rows differ from the ranked library")

            scored = role == "candidate"
            self._writer.writerow(
                {
                    "dataset": dataset,
                    "config": config,
                    "target_id": target.target_id,
                    "protocol": protocol,
                    "seed": seed,
                    "K": known,
                    "parent_molecule_id": parent,
                    "label": int(target.parent_labels[index]),
                    "role": role,
                    "score": float(result.scores[candidate]) if scored else "",
                    "similarity": float(result.similarity[candidate]) if scored else "",
                    "direct": float(result.direct[candidate]) if scored else "",
                    "graph": float(result.graph[candidate]) if scored else "",
                }
            )
            candidate += int(scored)
            self.row_count += 1
        if candidate != int(np.asarray(ranked_mask).sum()):
            raise RuntimeError("not every ranked molecule received a score row")

    def write_complete_library(self, dataset, config, target, protocol, known, result):
        if not np.array_equal(result.parent_ids.astype(str), target.parent_ids.astype(str)):
            raise RuntimeError("complete-library scores have unexpected parent IDs")
        for index, parent in enumerate(map(str, target.parent_ids)):
            self._writer.writerow(
                {
                    "dataset": dataset,
                    "config": config,
                    "target_id": target.target_id,
                    "protocol": protocol,
                    "seed": "",
                    "K": known,
                    "parent_molecule_id": parent,
                    "label": int(target.parent_labels[index]),
                    "role": "candidate",
                    "score": float(result.scores[index]),
                    "similarity": float(result.similarity[index]),
                    "direct": float(result.direct[index]),
                    "graph": float(result.graph[index]),
                }
            )
            self.row_count += 1

    def close(self) -> None:
        if self._text.closed:
            return
        self._text.flush()
        self._text.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
