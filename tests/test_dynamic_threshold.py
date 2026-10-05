"""Tests for `src/anomaly/dynamic_threshold.py`."""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.dynamic_threshold import (
    DynamicThreshold,
    apply_dynamic_threshold,
    fit_dynamic_threshold,
)


def test_dynamic_threshold_uses_rolling_mean_and_std():
    # window_size=2, but history is STRICTLY BEFORE each index (see the
    # 3 Sep 2026 bug-fix note in dynamic_threshold.py), so at most
    # `i` prior scores are available at index i, capped at window_size.
    #   i=0: no history -> threshold = scores[0] = 1.0
    #   i=1: history = [1.0]            -> mean=1.0, std=0.0 -> 1.0
    #   i=2: history = [1.0, 2.0]       -> mean=1.5, std=0.5 -> 1.5 + 1*0.5 = 2.0
    scores = np.array([1.0, 2.0, 3.0])

    thresholds = fit_dynamic_threshold(
        scores,
        window_size=2,
        lambda_=1.0,
    )

    expected = np.array([
        1.0,
        1.0,
        2.0,
    ])

    np.testing.assert_allclose(thresholds, expected)


def test_window_size_one_gives_previous_score_threshold():
    # With window_size=1 and strictly-prior history, the threshold at
    # index i is just scores[i - 1] (the single preceding score), not
    # scores[i] itself -- the current score is never part of its own
    # threshold.
    scores = np.array([1.0, 5.0, 2.0])

    thresholds = fit_dynamic_threshold(
        scores,
        window_size=1,
        lambda_=2.0,
    )

    expected = np.array([
        1.0,  # no history -> falls back to its own score
        1.0,  # history = [scores[0]] = [1.0]
        5.0,  # history = [scores[1]] = [5.0]
    ])

    np.testing.assert_allclose(thresholds, expected)


def test_lambda_zero_gives_rolling_mean():
    # i=0: no history -> threshold = 1.0
    # i=1: history = [1.0]      -> mean = 1.0
    # i=2: history = [1.0, 3.0] -> mean = 2.0
    scores = np.array([1.0, 3.0, 5.0])

    thresholds = fit_dynamic_threshold(
        scores,
        window_size=2,
        lambda_=0.0,
    )

    expected = np.array([
        1.0,
        1.0,
        2.0,
    ])

    np.testing.assert_allclose(thresholds, expected)


def test_larger_lambda_gives_higher_threshold():
    scores = np.array([1.0, 2.0, 5.0, 3.0])

    low = fit_dynamic_threshold(
        scores,
        window_size=3,
        lambda_=1.0,
    )

    high = fit_dynamic_threshold(
        scores,
        window_size=3,
        lambda_=3.0,
    )

    assert np.all(high >= low)
    assert np.any(high > low)


def test_apply_dynamic_threshold_flags_only_scores_above_threshold():
    scores = np.array([1.0, 5.0, 5.1, 10.0])
    thresholds = np.array([2.0, 5.0, 5.0, 10.0])

    alerts = apply_dynamic_threshold(scores, thresholds)

    assert alerts.tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_apply_dynamic_threshold_shape_mismatch_raises():
    scores = np.array([1.0, 2.0, 3.0])
    thresholds = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        apply_dynamic_threshold(scores, thresholds)


def test_empty_scores_raise():
    with pytest.raises(ValueError):
        fit_dynamic_threshold(
            np.array([]),
            window_size=3,
            lambda_=2.0,
        )


def test_non_1d_scores_raise():
    with pytest.raises(ValueError):
        fit_dynamic_threshold(
            np.array([[1.0, 2.0]]),
            window_size=2,
            lambda_=2.0,
        )


def test_invalid_window_size_raises():
    scores = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        fit_dynamic_threshold(scores, window_size=0, lambda_=2.0)


def test_negative_lambda_raises():
    scores = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        fit_dynamic_threshold(scores, window_size=2, lambda_=-1.0)


def test_dynamic_threshold_dataclass():
    threshold = DynamicThreshold(
        window_size=10,
        lambda_=2.5,
    )

    assert threshold.window_size == 10
    assert threshold.lambda_ == 2.5