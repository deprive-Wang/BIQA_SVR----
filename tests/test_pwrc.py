"""PWRC formula and saved-result backfill checks."""

import numpy as np
import pytest
from scipy.special import expit

from pwrc import pwrc_auc


def test_pwrc_analytic_integral_matches_numerical_curve() -> None:
    mos = np.array([5.0, 10.0, 20.0, 35.0, 55.0])
    prediction = np.array([2.0, 1.0, 3.0, 4.0, 5.0])
    thresholds = np.linspace(2.0, 20.0, 10001)
    rank = np.arange(1, 6)
    predicted_rank = prediction.astype(int)
    left, right = np.triu_indices(5, k=1)
    deviation = (np.abs(rank[left] - predicted_rank[left])
                 + np.abs(rank[right] - predicted_rank[right])) / 8
    level = (np.maximum(rank[left], rank[right]) - 1) / 4
    weight = np.expm1(deviation) + np.expm1(level)
    agreement = (np.sign(rank[left] - rank[right])
                 * np.sign(predicted_rank[left] - predicted_rank[right]))
    distance = np.abs(mos[left] - mos[right]) * 2
    curve = (expit(0.175 * (distance[:, None] - thresholds[None, :]))
             * (agreement * weight)[:, None]).sum(axis=0) / weight.sum()

    actual = pwrc_auc(mos, prediction, mos_min=5, mos_max=55,
                      threshold_min=2, threshold_max=20)

    assert actual == pytest.approx(np.trapezoid(curve, thresholds), abs=1e-7)


def test_pwrc_perfect_and_reversed_order() -> None:
    mos = np.array([5.0, 10.0, 20.0, 35.0, 55.0])
    forward = pwrc_auc(mos, mos, mos_min=5, mos_max=55,
                       threshold_min=1, threshold_max=20)
    reversed_score = pwrc_auc(mos, -mos, mos_min=5, mos_max=55,
                              threshold_min=1, threshold_max=20)

    assert 0 < forward <= 19
    assert -19 <= reversed_score < 0


def test_pwrc_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match='aligned'):
        pwrc_auc(np.array([1.0, 2.0]), np.array([1.0]), mos_min=0, mos_max=3,
                 threshold_min=1, threshold_max=2)
    with pytest.raises(ValueError, match='range'):
        pwrc_auc(np.array([1.0, 2.0]), np.array([1.0, 2.0]), mos_min=0, mos_max=3,
                 threshold_min=2, threshold_max=1)
