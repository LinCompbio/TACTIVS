"""Evaluation metrics kept separate from label-free inference."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

BAYES_EF_FRACTIONS = (0.005, 0.01, 0.05)


def bayes_enrichment_factors(
    labels: np.ndarray,
    scores: np.ndarray,
    fractions: tuple[float, ...] = BAYES_EF_FRACTIONS,
    *,
    require_measurable: bool = False,
) -> tuple[float, ...]:
    """Return TPR/FPR at the first complete threshold reaching each FPR."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError(
            "labels and scores must be one-dimensional with the same shape"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("both positive and negative labels are required")
    if not fractions or any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("fractions must contain values in (0, 1]")

    negative_count = int(np.count_nonzero(labels == 0))
    minimum_fpr = 1.0 / negative_count
    if require_measurable and any(fraction < minimum_fpr for fraction in fractions):
        raise ValueError(
            f"requested FPR is not measurable with {negative_count} negatives; "
            f"minimum measurable FPR is {minimum_fpr:g}"
        )

    fpr, tpr, _ = roc_curve(
        labels.astype(np.int8, copy=False),
        scores,
        pos_label=1,
        drop_intermediate=False,
    )
    values = []
    for fraction in fractions:
        cutoff = int(np.searchsorted(fpr, fraction, side="left"))
        values.append(float(tpr[cutoff] / fpr[cutoff]))
    return tuple(values)


def screening_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the ranking metrics used for retrospective VS evaluation."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if len(np.unique(labels)) < 2:
        raise ValueError("both positive and negative labels are required")

    from rdkit.ML.Scoring import Scoring

    order = np.argsort(-scores, kind="stable")
    feed = np.column_stack((labels[order], scores[order]))
    ef_0_5, ef_1, ef_5 = Scoring.CalcEnrichment(feed, 0, [0.005, 0.01, 0.05])
    bayes_ef_0_5, bayes_ef_1, bayes_ef_5 = bayes_enrichment_factors(labels, scores)

    bedroc_20 = float(Scoring.CalcBEDROC(feed, 0, alpha=20.0))
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "ef_0.5%": float(ef_0_5),
        "ef_1%": float(ef_1),
        "ef_5%": float(ef_5),
        "bedroc_20": bedroc_20,
        "bedroc_80.5": float(Scoring.CalcBEDROC(feed, 0, alpha=80.5)),
        "bedroc_85": float(Scoring.CalcBEDROC(feed, 0, alpha=85.0)),
        "bayes_ef_0.5%": bayes_ef_0_5,
        "bayes_ef_1%": bayes_ef_1,
        "bayes_ef_5%": bayes_ef_5,
    }
