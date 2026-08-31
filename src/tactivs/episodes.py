"""Positive-reference manifests and reproducible few-shot episode plans."""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha1(key.encode()).hexdigest()
    return (int(digest[:8], 16) + int(seed)) % (2**32)


def read_positive_manifest(path: Path) -> dict[str, dict[str, dict]]:
    """Read only positive reference identities and structures from a CSV manifest."""
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_id", "parent_molecule_id", "canonical_smiles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"positive manifest missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            target_id = row["target_id"].strip()
            parent_id = row["parent_molecule_id"].strip()
            if not target_id or not parent_id:
                raise ValueError(f"empty target or parent ID on manifest line {line_number}")
            if parent_id in output[target_id]:
                raise ValueError(
                    f"duplicate positive {target_id}/{parent_id} on line {line_number}"
                )
            molecule = Chem.MolFromSmiles(row["canonical_smiles"])
            if molecule is None:
                raise ValueError(f"invalid SMILES on manifest line {line_number}")
            output[target_id][parent_id] = {
                "smiles": Chem.MolToSmiles(molecule, canonical=True),
                "fp": MORGAN.GetFingerprint(molecule),
            }
    return dict(output)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def series_groups(records: dict[str, dict], threshold: float = 0.5) -> dict[str, str]:
    parents = sorted(records)
    union = UnionFind(len(parents))
    for i in range(1, len(parents)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            records[parents[i]]["fp"], [records[parents[j]]["fp"] for j in range(i)]
        )
        for j, similarity in enumerate(similarities):
            if similarity >= threshold:
                union.union(i, j)
    members: dict[int, list[str]] = defaultdict(list)
    for index, parent in enumerate(parents):
        members[union.find(index)].append(parent)
    names = {root: f"series::{min(group)}" for root, group in members.items()}
    return {parent: names[union.find(index)] for index, parent in enumerate(parents)}


def build_episode_plan(manifest, protocols, seeds, max_k: int):
    """Create reproducible nested support orders without fixing a query pool."""
    plan = {protocol: {} for protocol in protocols}
    for target_id, records in sorted(manifest.items()):
        for protocol in protocols:
            if len(records) < max_k:
                continue
            assignment = series_groups(records) if protocol == "series" else None
            if assignment is not None and len(set(assignment.values())) < 2:
                continue
            target_plan = {}
            for seed in seeds:
                rng = random.Random(stable_seed(seed, f"{protocol}:{target_id}"))
                if assignment is None:
                    support_order = sorted(records)
                    rng.shuffle(support_order)
                else:
                    groups: dict[str, list[str]] = defaultdict(list)
                    for parent, group in assignment.items():
                        groups[group].append(parent)
                    group_order = sorted(groups)
                    rng.shuffle(group_order)
                    support_order = []
                    for group in group_order:
                        members = groups[group].copy()
                        rng.shuffle(members)
                        support_order.extend(members)
                target_plan[str(seed)] = {"support_order": support_order}
            plan[protocol][target_id] = target_plan
    return plan


def episode_parent_masks(
    parent_ids: np.ndarray,
    support_ids: list[str],
    protocol: str,
    series_of_parent: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible and ranked-library masks for one runtime episode."""
    parent_strings = np.asarray(list(map(str, parent_ids)))
    support = set(map(str, support_ids))
    support_mask = np.asarray([parent in support for parent in parent_strings])
    if int(support_mask.sum()) != len(support):
        missing = sorted(support - set(parent_strings))
        raise KeyError(f"support IDs absent from target cache: {missing}")

    visible = np.ones(len(parent_strings), dtype=bool)
    if protocol == "series":
        if series_of_parent is None:
            raise ValueError("series protocol requires ECFP4 series assignments")
        support_series = {series_of_parent[parent] for parent in support}
        for index, parent in enumerate(parent_strings):
            if parent not in support and series_of_parent.get(parent) in support_series:
                visible[index] = False
    elif protocol != "fixed_random":
        raise ValueError(f"unknown episode protocol: {protocol}")

    ranked = visible & ~support_mask
    return visible, ranked
