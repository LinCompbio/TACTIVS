"""Discovery and validation for downloadable TACTIVS benchmark bundles."""

from __future__ import annotations

import json
from pathlib import Path

BUNDLE_SCHEMA = "tactivs-benchmark-bundle-v1"


def load_bundle_specs(root: Path) -> list[dict]:
    root = Path(root).resolve()
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(
            f"unsupported benchmark bundle schema: {manifest.get('schema')}"
        )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("benchmark bundle contains no datasets")
    specs = []
    names = set()
    for record in datasets:
        name = str(record["name"])
        if name in names:
            raise ValueError(f"duplicate dataset in benchmark bundle: {name}")
        names.add(name)
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"dataset path must stay inside the bundle: {relative}")
        dataset = (root / relative).resolve()
        required = {
            "cache": dataset,
            "whitener": dataset / "whitener.npz",
            "positive_manifest": dataset / "active_manifest.csv",
        }
        missing = [path for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete bundle dataset {name}: {missing}")
        specs.append(
            {
                "name": name,
                **required,
                "expected_targets": int(record["targets"]),
                "expected_positives": int(record["positives"]),
                "protocols": list(record.get("protocols", ["fixed_random"])),
                "cache_sha256": record.get("cache_sha256"),
                "whitener_sha256": record.get("whitener_sha256"),
                "positive_manifest_sha256": record.get("active_manifest_sha256"),
            }
        )
    return specs
