import numpy as np
import pytest

from tactivs.metrics import bayes_enrichment_factors, screening_metrics


def test_bayes_enrichment_uses_complete_tied_threshold() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([1.0, 0.8, 0.8, 0.1])

    assert bayes_enrichment_factors(labels, scores, (0.5,)) == (2.0,)


def test_screening_metrics_reuses_central_bayes_enrichment() -> None:
    labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int8)
    scores = np.asarray([0.99, 0.95, 0.90, 0.70, 0.40, 0.10])
    expected = bayes_enrichment_factors(labels, scores)

    observed = screening_metrics(labels, scores)

    assert observed["bayes_ef_0.5%"] == expected[0]
    assert observed["bayes_ef_1%"] == expected[1]
    assert observed["bayes_ef_5%"] == expected[2]


def test_bayes_enrichment_strict_mode_rejects_unmeasurable_fpr() -> None:
    labels = np.asarray([1, 0, 0, 0], dtype=np.int8)
    scores = np.asarray([1.0, 0.8, 0.4, 0.1])

    with pytest.raises(ValueError, match="not measurable"):
        bayes_enrichment_factors(labels, scores, (0.01,), require_measurable=True)
