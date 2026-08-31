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

The inference path does not read candidate activity labels. See
[PROTOCOL.md](PROTOCOL.md) for the experimental definition and [DATA.md](DATA.md)
for the cache schema and encoder boundary.

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

The publication environment used PyTorch 2.7.1+cu128, NumPy 1.26.4, pandas
2.3.3, Optuna 4.9.0, and RDKit 2022.03.5. `uv.lock` records a fully resolved
development and testing environment; the publication versions above remain
the authoritative environment for reproducing reported numerical results.

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

`final.json` enables native CUDA FP16 for the dense graph-neighbor search. Only
the similarity matrix multiplication and top-k selection use FP16; selected
edge similarities, graph weights, propagation, direct scoring, and fusion use
FP32. Direct-similarity queries use a larger FP32 chunk to reduce CUDA launch
overhead without changing their numerical result. On non-CUDA devices the graph
search setting falls back to FP32. Set
`graph_search_precision` to `fp32` for a strict full-precision baseline.

## Reproduction

Create a local configuration from [configs/data.example.json](configs/data.example.json)
and replace every placeholder path.

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
redistributed here; see [DATA.md](DATA.md).

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

## License and Citation

Research and other noncommercial use is permitted under the
[TACTIVS Research and Commercial Notice License 1.0](LICENSE). Commercial use
is permitted only after sending the prior written notice described in
[COMMERCIAL_USE.md](COMMERCIAL_USE.md). This is a source-available license, not
an OSI-approved open-source license. Citation metadata is in
[CITATION.cff](CITATION.cff).
