"""Unit tests for LightGBM forecaster and rolling-origin temporal splits."""

import numpy as np
import pandas as pd

from src.models.backtest import RollingOriginBacktest
from src.models.lgbm import LightGBMForecaster


def test_lgbm_fit_and_predict():
    """Verify LightGBMForecaster fits and predicts non-negative demand values."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "lag_28": np.random.uniform(0, 10, n),
        "sell_price": np.random.uniform(1, 5, n),
        "is_weekend": np.random.choice([0, 1], n),
    })
    y = np.maximum(0, X["lag_28"] * 0.8 - X["sell_price"] * 0.5 + np.random.normal(0, 1, n))

    model = LightGBMForecaster(n_estimators=30, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == n
    assert (preds >= 0.0).all(), "Predictions contain negative values"

    # Check feature importances
    imp = model.get_feature_importance()
    assert len(imp) == 3
    assert set(imp["feature"]) == {"lag_28", "sell_price", "is_weekend"}


def test_lgbm_serialization(tmp_path):
    """Verify model persistence to disk and reloading produces identical predictions."""
    X = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0], "feat_2": [10.0, 20.0, 30.0, 40.0]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0])

    model = LightGBMForecaster(n_estimators=20)
    model.fit(X, y)
    orig_preds = model.predict(X)

    save_file = tmp_path / "model.joblib"
    model.save(save_file)
    assert save_file.exists()

    loaded_model = LightGBMForecaster.load(save_file)
    loaded_preds = loaded_model.predict(X)

    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)


def test_rolling_origin_temporal_isolation():
    """Verify RollingOriginBacktest produces strictly isolated chronological folds."""
    dates = pd.date_range("2025-01-01", periods=150, freq="D")
    df = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "id": ["item_1", "item_2"] * 150,
        "sales": np.random.poisson(3, 300),
    })

    backtester = RollingOriginBacktest(horizon_days=28, num_folds=3, step_days=28)
    splits = backtester.generate_splits(df)

    assert len(splits) == 3

    # For every fold, train end date must be strictly before test start date
    for split in splits:
        assert split["train_end_date"] < split["test_start_date"]
        # Test window must be exactly 28 days
        test_days = (split["test_end_date"] - split["test_start_date"]).days + 1
        assert test_days == 28

    # Ensure fold test windows do not overlap
    for i in range(len(splits) - 1):
        assert splits[i]["test_end_date"] < splits[i + 1]["test_start_date"]
