#!/usr/bin/env python3
"""Convert <root>/<target>/actives.smi files to the common positive manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem

from tactivs.cache import EmbeddingCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = None
    if args.cache is not None:
        cache = EmbeddingCache(args.cache)
        allowed = {
            target_id: set(map(str, cache.parent_ids[target_id]))
            for target_id in cache.target_ids
        }
    rows = []
    for target_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        path = target_dir / "actives.smi"
        if not path.exists():
            continue
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 2:
                    raise ValueError(f"missing molecule ID at {path}:{line_number}")
                molecule = Chem.MolFromSmiles(fields[0])
                if molecule is None:
                    raise ValueError(f"invalid SMILES at {path}:{line_number}")
                if allowed is not None and fields[1] not in allowed.get(target_dir.name, set()):
                    continue
                rows.append(
                    {
                        "target_id": target_dir.name,
                        "parent_molecule_id": fields[1],
                        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_id", "parent_molecule_id", "canonical_smiles"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} positive records to {args.output}")


if __name__ == "__main__":
    main()
