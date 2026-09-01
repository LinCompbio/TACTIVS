# TACTIVS

TACTIVS (Test-time ACTIves induction for flexible target-specific Virtual
Screening) is a reference-only, transductive molecular-library readout. It
combines direct similarity to known actives with graph propagation over an unlabeled
candidate-library graph while keeping the molecular encoder frozen.

## Method

For one target, TACTIVS receives:

- `K` known active reference molecules;
- an otherwise unlabeled molecular library;
- precomputed multi-conformer EPT embeddings; and
- a frozen covariance-only whitening projection.

The library is centered using a molecule-balanced, target-local mean and then
projected and normalized. TACTIVS scores every non-reference molecule using:

1. **Direct similarity:** top-k conformer similarity to the reference
   conformers, max-pooled to the molecule level.
2. **Graph propagation:** propagate reference scores with restart over a
   degree-corrected k-nearest-neighbor graph.
3. **Fusion:** a standardized direct score plus a fixed weighted graph score.

The inference path does not read candidate activity labels.

## Installation

Python 3.9 or later is required. A CUDA-capable PyTorch installation is
recommended for full benchmark evaluation.

```bash
git clone <repository-url>
cd tactivs
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[test]'
pytest -q
```

Alternatively, create the tested base environment with Conda:

```bash
conda env create -f environment.yml
conda activate tactivs
```

Install a CUDA build of PyTorch appropriate for the host before a full GPU
benchmark run if the resolver selected a CPU build. EPT reference encoding also
requires the dependencies of the upstream EPT source tree, including a
PyTorch/CUDA-compatible `torch-scatter` build.

The publication environment used PyTorch 2.7.1+cu128, NumPy 1.26.4, pandas
2.3.3, Optuna 4.9.0, and RDKit 2022.03.5.

## Inference

For a labelled benchmark or internal collection, reproducibly sample `K`
cached actives as references:

```bash
tactivs-infer \
  --cache /path/to/cache/conformer \
  --whitener /path/to/covariance_only_whitener.npz \
  --target TARGET_ID \
  --sample-labeled 3 \
  --seed 0 \
  --theta final.json \
  --output ranking.csv
```

Labels are used only to select the references and are removed before scoring.
For prospective inference, build an external reference pool from a CSV with
`parent_molecule_id,canonical_smiles` columns or from an SDF:

```bash
tactivs-build-reference-pool \
  --molecules known_actives.csv \
  --output known_actives.npz \
  --ept-root /path/to/ept \
  --ept-ranking-root /path/to/ept/molfunnel_ranking_epoch49 \
  --encoder-checkpoint /path/to/ept.ckpt

tactivs-infer \
  --cache /path/to/cache/conformer \
  --whitener /path/to/covariance_only_whitener.npz \
  --target TARGET_ID \
  --reference-pool known_actives.npz \
  --theta final.json \
  --output ranking.csv
```

The output columns are `parent_molecule_id`, `score`, `similarity`, `direct`,
and `graph`.

### User scenario: screen a bundled library with private actives

A prospective user can select a target library from the public benchmark
bundle and rank it using one or more experimentally confirmed active ligands
that are not already members of that candidate library. Candidate embeddings
and the whitener come from the bundle; only the private reference ligands need
to be encoded.

Download the original EPT `epoch49_step215752.ckpt` from the
[public EPT checkpoint folder](https://drive.google.com/drive/folders/1tBqGwC_jcTdq3QArFZox_auSCzxDjA0P).
This is the checkpoint family expected by the legacy epoch-49 adapter in this
repository. The adapter extracts the ligand-only EPT `graph_repr`, matching the
512-dimensional raw representation used to construct the bundled candidate
caches. The checkpoint is not redistributed by TACTIVS.

Prepare `known_actives.csv`:

```csv
parent_molecule_id,canonical_smiles
private_active_1,CC(=O)Oc1ccccc1C(=O)O
private_active_2,CN1CCC[C@H]1c1cccnc1
```

Generate ten deterministic ETKDGv3 conformers per reference and encode them:

```bash
tactivs-build-reference-pool \
  --molecules known_actives.csv \
  --output known_actives.npz \
  --ept-root /path/to/upstream-ept \
  --ept-ranking-root /path/to/upstream-ept \
  --encoder-checkpoint /path/to/epoch49_step215752.ckpt \
  --conformers 10 \
  --seed 0
```

Then rank a target from one bundled dataset without reading its candidate
labels:

```bash
tactivs-infer \
  --cache /path/to/tactivs_benchmarks_zenodo_v1/datasets/litpcba \
  --whitener /path/to/tactivs_benchmarks_zenodo_v1/datasets/litpcba/whitener.npz \
  --target ADRB2 \
  --reference-pool known_actives.npz \
  --theta final.json \
  --output adrb2_ranking.csv
```

The EPT source checkout is required because the original checkpoint serializes
upstream Python model classes. A same-named checkpoint alone is not sufficient
without compatible upstream `models/` and `data/` modules. If a private active
is already present in the bundled library, use the labelled benchmark sampling
mode for a retrospective test or remove that molecule from a prospective
candidate cache; external reference IDs intentionally cannot collide with
candidate IDs.

`final.json` enables native CUDA FP16 for the dense graph-neighbor search. Only
the similarity matrix multiplication and top-k selection use FP16; selected
edge similarities, graph weights, propagation, direct scoring, and fusion use
FP32. Direct-similarity queries use a larger FP32 chunk to reduce CUDA launch
overhead without changing their numerical result. On non-CUDA devices the graph
search setting falls back to FP32. Set
`graph_search_precision` to `fp32` for a strict full-precision baseline.

## Reproduction

Download and extract the TACTIVS benchmark bundle from Zenodo. The extracted
directory has one versioned manifest and the same dataset layout for LIT-PCBA,
RandomDecoy, and TrueDecoy:

```text
tactivs-benchmarks/
  bundle.json
  datasets/
    litpcba/
    randomdecoy/
    truedecoy/
      index.npz
      active_manifest.csv
      whitener.npz
      embeddings/embeddings_*.npy
```

Run all three benchmarks without writing the much larger molecule-level score
artifact:

```bash
python scripts/evaluate_benchmarks.py \
  --bundle /path/to/tactivs-benchmarks \
  --theta final.json \
  --output-dir runs/publication \
  --skip-molecule-scores
```

The command writes `episodes.csv`, `metrics_per_dataset_k.csv`,
`metrics_per_dataset_protocol_k.csv`, `summary.csv`, and
`run_provenance.json`. LIT-PCBA uses molecule-random reference sampling;
RandomDecoy and TrueDecoy additionally use the series-disjoint protocol. Add
`--datasets litpcba` or `--ks 1 --seeds 0` for a smaller run. Omit
`--skip-molecule-scores` when candidate-level rankings are required.

The bundle can be staged from the original caches with
`scripts/build_benchmark_bundle.py` and the source template in
`configs/bundle_sources.example.json`. Hard links are used by default so local
staging does not duplicate the embedding shards. Zenodo users do not need this
construction step.

The legacy path-based configuration remains supported. Create a local
configuration from [configs/data.example.json](configs/data.example.json) and
replace every placeholder path.

Parameter selection on DEKOIS2:

```bash
python search_density_free.py \
  --data-config configs/data.local.json \
  --output-dir runs/density_free_dekois_reference_only_100
```

The search defaults to FP32 to preserve the original selection run. Add
`--graph-search-precision fp16` for an accelerated, numerically near-equivalent
search recorded under a distinct run specification.

Frozen benchmark evaluation:

```bash
python scripts/evaluate_benchmarks.py \
  --data-config configs/data.local.json \
  --theta final.json \
  --output-dir runs/final_evaluation
```

Module ablations:

```bash
python scripts/evaluate_module_ablations.py \
  --data-config configs/data.local.json \
  --theta final.json \
  --output-dir runs/module_ablations
```

These commands require separately obtained benchmark structures, EPT
embeddings, the frozen EPT checkpoint, and whitening artifacts. They are not
redistributed here.

## Repository Layout

```text
configs/        portable data-configuration template
scripts/        cache preparation and evaluation entry points
src/tactivs/    inference, episodes, metrics, and artifact writers
tests/          synthetic unit and integration tests
```

`final.json` contains the frozen global parameter vector and graph-search
execution precision.
`density_free_search_space.json` contains the formal search space.
