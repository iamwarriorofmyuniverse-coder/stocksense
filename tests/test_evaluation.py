"""Unit tests for demand velocity classification and failure analysis."""

import numpy as np
import pandas as pd

from src.evaluation.classification import classify_demand_patterns
from src.evaluation.failure_analysis import run_failure_analysis


def test_syntetos_boylan_classification():
    """Verify time series are categorized into correct Syntetos-Boylan quadrants."""
    n_days = 60
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    # 1. Smooth: Daily non-zero demand with constant size (ADI=1.0, CV2=0.0)
    smooth_sales = np.full(n_days, 5)

    # 2. Intermittent: Non-zero demand every 3 days with constant size (ADI=3.0, CV2=0.0)
    intermittent_sales = np.zeros(n_days)
    intermittent_sales[::3] = 4

    # 3. Erratic: Daily non-zero demand with wild size variance (ADI=1.0, CV2 >= 0.49)
    erratic_sales = np.array([2, 50, 3, 45, 1, 60] * 10)

    # 4. Lumpy: Infrequent demand with wild size variance (ADI >= 1.32, CV2 >= 0.49)
    lumpy_sales = np.zeros(n_days)
    lumpy_sales[::4] = [2, 50, 3, 45, 1, 60, 2, 70, 4, 80, 1, 90, 3, 50, 2]

    df = pd.DataFrame({
        "id": (
            ["s_smooth"] * n_days
            + ["s_intermittent"] * n_days
            + ["s_erratic"] * n_days
            + ["s_lumpy"] * n_days
        ),
        "date": list(dates) * 4,
        "sales": np.concatenate([smooth_sales, intermittent_sales, erratic_sales, lumpy_sales]),
    })

    result = classify_demand_patterns(df).set_index("id")

    assert result.loc["s_smooth", "demand_type"] == "Smooth"
    assert result.loc["s_intermittent", "demand_type"] == "Intermittent"
    assert result.loc["s_erratic", "demand_type"] == "Erratic"
    assert result.loc["s_lumpy", "demand_type"] == "Lumpy"


def test_failure_analysis_execution(tmp_path):
    """Verify run_failure_analysis executes on mock predictions and exports summary."""
    n = 28
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    mock_preds = pd.DataFrame({
        "id": ["item_1"] * n + ["item_2"] * n,
        "item_id": ["i1"] * n + ["i2"] * n,
        "dept_id": ["d1"] * n + ["d2"] * n,
        "cat_id": ["FOODS"] * n + ["HOBBIES"] * n,
        "store_id": ["CA_1"] * n + ["CA_1"] * n,
        "date": list(dates) * 2,
        "sales": [5] * n + [0, 1, 0, 0, 2, 0, 1] * 4,
        "sell_price": [2.5] * (2 * n),
        "fold": [1] * (2 * n),
        "pred_lgbm": [4.8] * n + [0.5] * n,
        "pred_moving_average_28d": [5.0] * n + [0.4] * n,
        "pred_moving_average_7d": [5.1] * n + [0.3] * n,
        "pred_naive": [5.0] * n + [0.0] * n,
        "pred_seasonal_naive_7d": [5.0] * n + [0.5] * n,
    })

    preds_path = tmp_path / "mock_preds.parquet"
    mock_preds.to_parquet(preds_path, index=False)

    report = run_failure_analysis(predictions_path=preds_path, output_dir=tmp_path)

    assert "failure_summary" in report
    assert report["failure_summary"]["total_skus_evaluated"] == 2
    assert (tmp_path / "failure_analysis.parquet").exists()
    assert (tmp_path / "granular_evaluation.json").exists()
