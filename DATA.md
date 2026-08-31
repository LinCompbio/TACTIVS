# Data and Artifacts

Benchmark data, molecular structures, labels, embeddings, model checkpoints,
and whitening artifacts are not included in this repository. Users must obtain
them under their respective licenses and configure local paths using
`configs/data.example.json`.

## Embedding Cache

TACTIVS expects an EPT embedding cache with this layout:

```text
cache/conformer/
  embeddings_000000.npy
  embeddings_000001.npy
  ...
  target_manifest.npz
  target_parent_ids.npz
```

An encoder export may instead contain `metadata.csv` plus the embedding shards.
Build the compact manifests with:

```bash
python scripts/prepare_cache_manifests.py --cache /path/to/cache/conformer
```

`metadata.csv` must be sorted contiguously by target, molecule, and conformer,
and contain `embedding_index`, `target_id`, and `parent_molecule_id`. A `label`
column is optional for inference and required for retrospective evaluation.

`target_manifest.npz` stores target row ranges and per-molecule conformer
counts. It may contain labels, but `EmbeddingCache.get(...,
include_labels=False)` does not read them. `target_parent_ids.npz` stores parent
IDs independently of the label array.

## Positive Manifest

Parameter selection and series-disjoint evaluation additionally require a CSV
with at least:

```text
target_id,parent_molecule_id,canonical_smiles
```

The positive IDs must exactly match the cached positive IDs. Structures are
used only to construct ECFP4 connected components for reference episodes.

For a benchmark organized as per-target active SMILES files, use:

```bash
python scripts/build_positive_manifest.py \
  --root /path/to/benchmark \
  --cache /path/to/cache/conformer \
  --output /path/to/active_manifest.csv
```

## Whitener

The whitener is an NPZ file containing a `projection` array and no fixed
`mean`. TACTIVS applies this covariance-only projection after computing an
unlabeled, molecule-balanced mean for the current target library.

## EPT Encoding Boundary

The reported cache was produced by canonicalizing each molecule, generating up
to ten pocket-independent ETKDGv3 conformers through nvMolKit, and applying the
frozen epoch-49 EPT ligand-only graph-representation path. The EPT source tree,
nvMolKit package, checkpoint, and whitener are upstream software/model
artifacts rather than benchmark data and are not vendored here.

The candidate-library workflow starts at the encoder boundary: an encoder must
emit one float32 row per conformer into `embeddings_NNNNNN.npy`, in the same
order as `metadata.csv`. User-supplied references can instead be processed with
`tactivs-build-reference-pool`, which calls the upstream encoder and produces a
reusable NPZ pool.

The reference-pool command requires both `--ept-root` and
`--ept-ranking-root`. They are explicit because the EPT ranking tree is an
upstream dependency and its checkout layout is not part of the TACTIVS API.
The adapter includes a narrowly scoped compatibility patch for attributes that
are absent from the frozen epoch-49 checkpoint schema; it is not used by the
TACTIVS scoring implementation.

After cache creation, inference requires neither SMILES nor activity labels.
Retrospective scripts access labels only after parameters are fixed, for
reference sampling and metric computation as documented in `PROTOCOL.md`.
