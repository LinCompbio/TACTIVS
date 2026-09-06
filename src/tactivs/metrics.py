"""Evaluation metrics kept separate from label-free inference."""

from __future__ import annotations

import numpy as np


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
    roc = Scoring.CalcROC(feed, 0)
    fpr = np.asarray(roc.FPR)
    tpr = np.asarray(roc.TPR)
    sorted_scores = feed[:, 1]

    def bayes_enrichment(fraction: float) -> float:
        cutoff = min(
            int(np.searchsorted(fpr, fraction, side="left")), len(fpr) - 1
        )
        # Include every molecule at the threshold when scores are tied.
        while (
            cutoff + 1 < len(sorted_scores)
            and sorted_scores[cutoff + 1] == sorted_scores[cutoff]
        ):
            cutoff += 1
        return (
            float(tpr[cutoff] / fpr[cutoff])
            if fpr[cutoff] > 0
            else float("nan")
        )

    ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(feed[:, 0])[ends]
    precision = true_positives / (ends + 1)
    recall = true_positives / int(labels.sum())
    bedroc_20 = float(Scoring.CalcBEDROC(feed, 0, alpha=20.0))
    bedroc_80_5 = float(Scoring.CalcBEDROC(feed, 0, alpha=80.5))
    return {
        "auc_roc": float(Scoring.CalcAUC(feed, 0)),
        "auc_pr": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
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


def ef_at_fraction(labels: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    prevalence = float(labels.mean())
    if prevalence <= 0.0:
        raise ValueError("at least one positive is required")
    n_top = max(1, int(np.ceil(fraction * len(labels))))
    top = np.argpartition(scores, -n_top)[-n_top:]
    return float(labels[top].mean() / prevalence)


def paired_bootstrap(
    differences: np.ndarray, samples: int = 20000, seed: int = 20260720
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for begin in range(0, samples, 1000):
        size = min(1000, samples - begin)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        draws[begin : begin + size] = values[indices].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)
