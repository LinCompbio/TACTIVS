# Preliminary findings

The full-target diagnostic was run on all three publication benchmarks using
the same target centering and covariance whitening as the production code.

| Dataset | Molecules | Median R | 5th percentile | R < 0.75 | Minimum R |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEKOIS2 | 100,324 | 0.883 | 0.783 | 1.16% | 0.579 |
| RandomDecoy | 787,262 | 0.888 | 0.795 | 0.54% | 0.608 |
| TrueDecoy | 233,702 | 0.886 | 0.768 | 2.81% | 0.562 |

Across all multi-conformer molecules, median mean pairwise conformer cosine is
0.763 (5th to 95th percentile: 0.579 to 0.942). The directional ensemble is
therefore generally concentrated: normalization is not commonly amplifying a
near-cancelled conformer mean.

Nearly all molecules have ten conformers. Within each dataset, Spearman
correlation between conformer count and `R` is only -0.01 to -0.02, so the
dataset-level distributions are not explained by varying conformer count.

Actives have slightly lower `R` than inactives in RandomDecoy and TrueDecoy.
This means `R` must not be treated as a label-free quality weight without an
outcome ablation: down-weighting low-`R` nodes could preferentially down-weight
some actives.

As an integration smoke test, one TrueDecoy target (`P10415`) was joined to 100
existing fixed-random episodes. Mean per-episode Spearman correlation between
`R` and graph score was -0.014, indicating almost no direct association for
that target. This is not a benchmark-level result.

## Interpretation

The distribution analysis is already useful for answering whether conformer
averaging is geometrically reasonable: it is, for almost all molecules in
these caches. It does not establish that discarding `R` is optimal. The next
experiment, if needed, should be a controlled graph ablation comparing the
current unit mean direction against an `R`-aware graph on the frozen episode
plan. Series-disjoint `R` must be computed after the episode-specific visible
set and target mean are formed.
