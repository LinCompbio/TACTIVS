"""Evaluation metrics kept separate from label-free inference."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


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
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)

    def bayes_enrichment(fraction: float) -> float:
        cutoff = min(
            int(np.searchsorted(fpr, fraction, side="left")), len(fpr) - 1
        )
        return (
            float(tpr[cutoff] / fpr[cutoff])
            if fpr[cutoff] > 0
            else float("nan")
        )

    bedroc_20 = float(Scoring.CalcBEDROC(feed, 0, alpha=20.0))
    bedroc_80_5 = float(Scoring.CalcBEDROC(feed, 0, alpha=80.5))
    return {
        "auc_roc": float(roc_auc_score(labels, scores)),
        "auc_pr": float(average_precision_score(labels, scores)),
        "ef_0.5%": float(ef_0_5),
        "ef_1%": float(ef_1),
        "ef_5%": float(ef_5),
        "bedroc_20": bedroc_20,
        "bedroc_80_5": bedroc_80_5,
        "bedroc_85": float(Scoring.CalcBEDROC(feed, 0, alpha=85.0)),
        "bayes_ef_0.5%": bayes_enrichment(0.005),
        "bayes_ef_1%": bayes_enrichment(0.01),
        "bayes_ef_5%": bayes_enrichment(0.05),
    }
