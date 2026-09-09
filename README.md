# TACTIVS

## Overview

**TACTIVS** is a ligand-based transductive inference framework that conditions
a frozen molecular representation directly at test time for flexible,
target-specific virtual screening. Starting from a small pool of known active
molecules, TACTIVS combines target-local centering, covariance whitening,
direct reference similarity, and graph propagation to rank an unlabeled
molecular library without target-specific model training.

<div align="center">
<img src="figures/fig1.jpeg" width="90%" alt="TACTIVS framework overview" />
</div>

## Installation

```bash
conda env create -f environment.yml
conda activate tactivs
```

The environment uses PyTorch 2.7.1 with CUDA 12.8. This build supports NVIDIA
GPU architectures from `sm_75` through `sm_120`, including RTX 20, 30, 40, and
50 series GPUs. A recent NVIDIA driver with CUDA 12.8 support is required.

Download the published EPT epoch-49 checkpoint from the
[EPT pretrained checkpoint folder](https://drive.google.com/drive/folders/1ISCsnXss6YueYUvAIiR4wpm3k0TGjb44)
and pass it with `--encoder-checkpoint`.

## Input Format

Candidate libraries and external reference pools use the same CSV format:

```csv
parent_molecule_id,canonical_smiles
molecule_1,CCO
molecule_2,CCN
```

SDF and MOL files are also accepted. For SDF input, the molecule title is used
as `parent_molecule_id`.

## User Inference

First encode the candidate library:

```bash
python tactivs.py build-cache \
  --molecules candidates.csv \
  --output cache/my_target \
  --target MY_TARGET \
  --encoder-checkpoint /path/to/EPT.ckpt \
  --conformers 10
```

Inference defaults to the bundled `whitener.npz` (approximately 1 MB), a
512-dimensional PDBscreen covariance projection for the EPT epoch-49 encoder.
No separate whitener download or fitting is required.

For small screening libraries, we recommend this provided whitening projection
alongside the target-local mean computed at inference. For sufficiently large
libraries, we recommend estimating both the mean and the whitening projection
from the screening library to localize both centering and covariance scaling to
its molecular distribution. The fitting command computes the library mean and
covariance together from unlabeled embeddings:

```bash
python tactivs.py fit-whitener \
  --cache cache/my_target \
  --output cache/my_target_whitener.npz
```

EPT embeddings have 512 dimensions, so this command requires at least 513
molecules. This is a computational minimum; a substantially larger, diverse
library is preferable for estimating its covariance. The library mean is used
to estimate the covariance but is not saved; inference recomputes the
target-local mean after combining candidates and references. Fitting remains
an explicit step: omit it to use the bundled PDBscreen whitener.

Encode the known actives as the reference pool:

```bash
python tactivs.py build-reference-pool \
  --molecules known_actives.csv \
  --output cache/my_target_references.npz \
  --encoder-checkpoint /path/to/EPT.ckpt \
  --conformers 10
```

Run inference:

```bash
python tactivs.py infer \
  --cache cache/my_target \
  --target MY_TARGET \
  --reference-pool cache/my_target_references.npz \
  --theta final.json \
  --output ranking.csv
```

To use your own fitted projection, add
`--whitener cache/my_target_whitener.npz` to the inference command.
The bundled projection is the PDBscreen artifact used for DEKOIS2 development
and TrueDecoy evaluation; it stores only the projection, without a fixed mean.

Candidate and reference embeddings are combined before target-local centering,
whitening, and graph construction. References supply both direct-similarity
conformers and graph seeds; they participate in propagation and are excluded
only from the ranked output. Benchmark episodes follow the same scoring path:
their references are already in the cache, while external references are appended
to the candidate cache in memory. Identical embeddings, projection, parameters,
and molecule order therefore define the same scoring problem in both modes.
`ranking.csv` contains `parent_molecule_id`, `score`, `similarity`, `direct`,
and `graph`; no benchmark metrics are calculated in this path.

Cache and reference-pool construction display progress bars for conformer
generation and EPT encoding. Cache construction also shows overall batch progress.
For benchmark reproduction, the evaluator uses each dataset's supplied whitener
to preserve its benchmark-specific overlap exclusions.

## Benchmark Reproduction

Download [`tactivs_data.tar.zst`](https://drive.google.com/open?id=1AmjcaP54jM3MmtE_mTnlrzvC_hsQjajc)
and extract it before running the benchmarks:

```bash
tar --zstd -xf tactivs_data.tar.zst
```

The released benchmark cache contains both active and inactive molecules with
labels. For every episode, its selected active molecules form the reference
pool, and all remaining molecules in the episode are ranked by the same
inference path used above. Labels are read again only after ranking to calculate
metrics.

```bash
python tactivs.py benchmark \
  --data-root /path/to/tactivs_data \
  --benchmark truedecoy \
  --split series-disjoint \
  --seed 0 \
  --k 1 2 3 4 5 6 7 8 9 10 \
  --theta final.json \
  --output-dir runs/truedecoy-series
```

TrueDecoy and RandomDecoy support `molecule-random` and `series-disjoint`.
LIT-PCBA supports `molecule-random` and `ave`. `--k` accepts one or more unique
values from 1 through 10 and `--seed` defaults to 0. Each target is written to
`<output-dir>-<UTC timestamp>/<target_id>/seed_<seed>/`, containing
`summary.csv` and `scores.csv`. Both files include a `K` column.

The benchmark data layout is:

```text
tactivs_data/
  truedecoy/
    ept/index.npz
    ept/embeddings/
    whitener.npz
    splits/
  randomdecoy/
  litpcba/
```

## Acknowledgements

TACTIVS is built on the [Equivariant Pretrained Transformer
(EPT)](https://doi.org/10.1038/s41467-026-69185-7). We thank the EPT authors for
their outstanding work. The minimal EPT inference components required by
TACTIVS are included in `src/tactivs/_vendor/ept` under the original license.
