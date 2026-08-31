# Experimental protocol

## Research question

Can one global readout configuration be selected from a small set of known
active references and an unlabeled target library, without using query or decoy
labels, and then transfer to external decoy benchmarks?

## Representation

All molecules are encoded by the same frozen EPT encoder. Each molecule may
have multiple generic conformers. A frozen covariance-only projection is
applied after subtracting the molecule-balanced mean of the current unlabeled
target library. This target-local centering is transductive and label-free.

The encoder and whitening artifacts are inputs to this repository; they are not
trained here.

## Readout

Given K reference molecules, TACTIVS combines two components:

1. **Similarity**: for every query conformer, average the highest similarities
   to reference conformers, then max-pool conformers to a molecule score.
2. **Graph propagation**: propagate reference mass over a degree-corrected
   molecule similarity graph with restart.

The accelerated configuration uses native CUDA FP16 only to compute graph KNN
similarities and select neighbors. Selected similarities are converted to FP32;
edge weighting, propagation, direct scoring, standardization, and score fusion
remain FP32. CPU execution and the explicit `fp32` configuration use FP32 for
the complete graph construction.

All target-library covariance and graph calculations are permitted unlabeled
transductive operations. The observable set is also the metric set:
apart from the K references, every molecule allowed to affect these operations
must receive a ranking score and enter retrospective evaluation.

## Parameter selection

- Selection dataset: DEKOIS2 only, 81 targets.
- Known labels per episode: exactly K positive references.
- Selection K: 3, 5, 7 and 10.
- Episode protocols: molecule-random and ECFP4-series-disjoint.
- Episode seeds: 0, 1 and 2.
- Surrogate: hold out one ECFP4 series inside the K references and measure its
  EF@1% against the otherwise unlabeled library.
- Search: Optuna TPE, seed `20260720`, multivariate sampling, 16 startup trials.
- Fixed search budget: 100 completed trials.
- Selection rule: among trials with nonnegative direct and graph contributions
  in both protocols, select the maximum full-model reference surrogate.
- Persistence: a SQLite study checkpoints every trial; a restart completes only
  the remaining trial budget and refuses changed inputs or source code.

The surrogate constructs pseudo-negative labels for metric computation by
treating the unlabeled pool as background. It does not inspect the stored
active/decoy labels. Unknown positives therefore remain in that pool, as
required by the positive-unlabeled setting.

### K=1 boundary

With one known positive, no independent molecular positive remains for an
honest held-out objective. K=1 therefore does not contribute to Optuna. The
single global vector selected from K={3,5,7,10} is transferred unchanged to
K=1 and is evaluated there. No generated or benchmark negative is introduced
to avoid this limitation.

## Evaluation

After theta is frozen, labels are used only for retrospective metric
calculation.

- DEKOIS2: selection/development benchmark.
- RandomDecoy: external benchmark, absent from selection.
- TrueDecoy: external benchmark, absent from selection.
- Evaluation K: 1 through 10.
- Evaluation seeds: 0, 1 and 2.
- Protocols: molecule-random and ECFP4-series-disjoint.
- Molecule-random episodes draw K nested references from all cached positives at
  runtime. The K references are removed and every remaining cached molecule is
  ranked, so a target with P positives and N total molecules has P-K evaluated
  positives and N-K candidates.
- Series-disjoint episodes also draw references at runtime. Unselected positives
  in a reference ECFP4 connected component are removed from the complete episode,
  including centering, graph construction and metrics. Every other
  non-reference molecule is ranked.
- Episode rows record support, visible, candidate, positive, negative and
  series-excluded counts so the EF population is auditable.
- Molecule-level artifacts record every cached parent in every episode, its
  support/candidate/series-excluded role, and all component scores when ranked.
- Aggregation: average seeds and protocols within target, then target-macro.
- Uncertainty: target-level paired bootstrap, 20,000 samples.
- Descriptive benchmark tables: sample standard deviation across target-level
  values after averaging seeds and protocols within each target.
- Reported enrichment metrics: EF@0.5%, EF@1%, and EF@5%.
- BEDROC: alpha 20 for the internal analysis; alpha 80.5 is additionally
  reported for direct comparison with the published UniDock-Pro tables.

## Claims supported by this protocol

Supported:

- parameters were selected without DEKOIS query or decoy labels;
- external benchmark labels did not participate in selection;
- inference uses K known positives and an otherwise unlabeled library;
- target-library statistics are transductive.
- graph nodes and EF candidates share the same observable boundary.

Not supported:

- fully inductive inference independent of the target library;
- direct reference-only optimization for K=1;
- encoder-level recovery of activity cliffs absent from the representation.
