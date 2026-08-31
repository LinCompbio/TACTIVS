# Conformer resultant-length exploration

This directory contains disposable analyses that are not part of the public
TACTIVS inference path.

`conformer_resultant.py` measures the information discarded when conformer
means are normalized for graph construction. For molecule `i`, it computes

```text
a_i = mean_c(z_ic)
R_i = ||a_i||_2
m_i = a_i / R_i
```

where `z_ic` is the already whitened and L2-normalized conformer embedding.
The production graph uses `m_i`; this analysis records `R_i` immediately before
that normalization.

Run the distribution analysis with the publication data configuration:

```bash
uv run python explore/conformer_resultant.py \
  --data-config /home/s2523227/tactivs/configs/data.publication.json \
  --output-dir explore/results/conformer_resultant \
  --device cuda
```

Use `--datasets dekois2` or `--max-targets 5` for a quick check. An optional
score artifact can be joined to the resultant lengths:

```bash
uv run python explore/conformer_resultant.py \
  --data-config /home/s2523227/tactivs/configs/data.publication.json \
  --output-dir explore/results/conformer_resultant_dekois2 \
  --datasets dekois2 \
  --score-files runs/example/molecule_scores.csv.gz
```

The score association deliberately retains only `fixed_random` candidate rows.
Series-disjoint episodes recompute the target-local mean after removing a
series, so their `R_i` values must be recomputed inside each episode rather than
joined to the full-target values produced here.

Outputs:

- `molecule_resultants.csv.gz`: one row per molecule.
- `summary.csv`: distribution summaries, separating single- and
  multi-conformer molecules and active/inactive labels.
- `by_conformer_count.csv`: distribution by exact conformer count.
- `score_associations.csv`: optional per-episode Spearman associations between
  `R_i` and graph/direct/final scores.
