import numpy as np
from rdkit.ML.Scoring import Scoring

from tactivs.metrics import ef_at_fraction, screening_metrics


def test_enrichment_is_prevalence_normalized():
    labels = np.asarray([1, 0, 0, 0])
    scores = np.asarray([4.0, 3.0, 2.0, 1.0])
    assert ef_at_fraction(labels, scores, fraction=0.25) == 4.0


def test_screening_metrics_reports_both_bedroc_conventions():
    labels = np.asarray([0, 1, 0, 1, 0, 0], dtype=np.int8)
    scores = np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    order = np.argsort(-scores, kind="stable")
    feed = np.column_stack((labels[order], scores[order]))

    observed = screening_metrics(labels, scores)

    assert observed["ef_0.5%"] == Scoring.CalcEnrichment(feed, 0, [0.005])[0]
    assert observed["bedroc_20"] == Scoring.CalcBEDROC(feed, 0, alpha=20.0)
    assert observed["bedroc_80_5"] == Scoring.CalcBEDROC(feed, 0, alpha=80.5)
