"""Unit tests for baseline models and evaluation metric formulas."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_bias,
    calculate_mae,
    calculate_wape,
)
from src.models.baselines import (
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalNaiveBaseline,
    generate_all_baselines,
)


def test_naive_baseline():
    """Verify Naive baseline predicts the last observed value."""
    history = pd.DataFrame({
        "id": ["item_1", "item_1", "item_2", "item_2"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
        "sales": [5, 12, 3, 7],
    })
    test_df = pd.DataFrame({
        "id": ["item_1", "item_2", "item_1"],
        "date": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-04"]),
    })

    model = NaiveBaseline().fit(history)
    preds = model.predict(test_df)

    # item_1 last was 12, item_2 last was 7
    np.testing.assert_array_equal(preds, np.array([12.0, 7.0, 12.0], dtype=np.float32))


def test_seasonal_naive_baseline():
    """Verify Seasonal Naive repeats sales from the same day of week."""
    # 7 days of history: Monday (dow=0) to Sunday (dow=6)
    dates = pd.date_range("2025-01-06", periods=7, freq="D")  # 2025-01-06 is Monday
    history = pd.DataFrame({
        "id": ["item_A"] * 7,
        "date": dates,
        "sales": [10, 12, 14, 16, 18, 30, 25],  # Mon=10, Tue=12, ..., Sat=30, Sun=25
    })

    # Predict next week: next Monday and next Saturday
    test_dates = pd.to_datetime(["2025-01-13", "2025-01-18"])  # Monday, Saturday
    test_df = pd.DataFrame({
        "id": ["item_A", "item_A"],
        "date": test_dates,
    })

    model = SeasonalNaiveBaseline(seasonality=7).fit(history)
    preds = model.predict(test_df)

    # Monday should be 10, Saturday should be 30
    np.testing.assert_array_equal(preds, np.array([10.0, 30.0], dtype=np.float32))


def test_moving_average_baseline():
    """Verify Moving Average calculates correct historical window mean."""
    history = pd.DataFrame({
        "id": ["item_X"] * 5,
        "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        "sales": [2, 4, 6, 8, 10],
    })
    test_df = pd.DataFrame({
        "id": ["item_X", "item_X"],
        "date": pd.date_range("2025-01-06", periods=2, freq="D"),
    })

    ma_3 = MovingAverageBaseline(window=3).fit(history)
    # Window of 3 takes tail: [6, 8, 10] -> mean = 8.0
    preds_3 = ma_3.predict(test_df)
    np.testing.assert_array_equal(preds_3, np.array([8.0, 8.0], dtype=np.float32))

    ma_5 = MovingAverageBaseline(window=5).fit(history)
    # Window of 5 takes [2, 4, 6, 8, 10] -> mean = 6.0
    preds_5 = ma_5.predict(test_df)
    np.testing.assert_array_equal(preds_5, np.array([6.0, 6.0], dtype=np.float32))


def test_metric_mathematical_definitions():
    """Verify MAE, WAPE, and Bias formulas against hand-calculated ground truth."""
    y_true = np.array([10.0, 0.0, 5.0, 20.0])
    y_pred = np.array([12.0, 2.0, 5.0, 15.0])

    # Absolute errors: [2, 2, 0, 5] -> sum = 9.0
    # True sum: 35.0
    # Errors (y_hat - y): [2, 2, 0, -5] -> sum = -1.0
    expected_mae = 9.0 / 4.0      # 2.25
    expected_wape = 9.0 / 35.0    # ~0.25714
    expected_bias = -1.0 / 35.0   # ~-0.02857

    assert pytest.approx(calculate_mae(y_true, y_pred), rel=1e-4) == expected_mae
    assert pytest.approx(calculate_wape(y_true, y_pred), rel=1e-4) == expected_wape
    assert pytest.approx(calculate_bias(y_true, y_pred), rel=1e-4) == expected_bias


def test_generate_all_baselines(sample_panel_df):
    """Verify generate_all_baselines returns predictions for all 4 baseline variants."""
    unique_dates = sorted(sample_panel_df["date"].unique())
    split_date = unique_dates[-14]

    train_df = sample_panel_df[sample_panel_df["date"] <= split_date]
    test_df = sample_panel_df[sample_panel_df["date"] > split_date]

    preds = generate_all_baselines(train_df, test_df)

    assert "naive" in preds
    assert "seasonal_naive_7d" in preds
    assert "moving_average_7d" in preds
    assert "moving_average_28d" in preds
    for k, p in preds.items():
        assert len(p) == len(test_df), f"Baseline {k} length mismatch"
