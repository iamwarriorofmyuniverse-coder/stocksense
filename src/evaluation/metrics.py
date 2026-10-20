"""Evaluation metrics for retail demand forecasting and inventory impact."""

import numpy as np
import pandas as pd


def calculate_mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error: (1 / N) * sum(|y - y_hat|)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error (WAPE): sum(|y - y_hat|) / sum(y).

    Why WAPE is the retail industry standard over MAPE:
    1. Zero-division immunity: In retail, daily sales frequently contain zeros (intermittent demand).
       Standard MAPE (1/N * sum(|y - y_hat| / y)) divides by zero and explodes.
    2. Volume-weighted correctness: WAPE weights errors by product volume. An error of 10 units on
       a high-volume product matters far more to store operations than 1 unit on a slow mover.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    sum_actual = np.sum(y_t)
    if sum_actual == 0:
        return 0.0 if np.sum(y_p) == 0 else 1.0
    return float(np.sum(np.abs(y_t - y_p)) / sum_actual)


def calculate_bias(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Normalized Forecast Bias: sum(y_hat - y) / sum(y).

    Interpretation for Retail Operations:
    - Positive Bias (> 0): Systematic over-forecasting. Drives bloated working capital,
      shelf congestion, and inventory spoilage/salvage write-offs.
    - Negative Bias (< 0): Systematic under-forecasting. Drives shelf stock-outs,
      lost customer sales, and brand damage.
    - Bias = 0.0: Unbiased forecast (errors cancel out in aggregate).
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    sum_actual = np.sum(y_t)
    if sum_actual == 0:
        return 0.0
    return float(np.sum(y_p - y_t) / sum_actual)


def evaluate_forecasts(
    df: pd.DataFrame,
    actual_col: str = "sales",
    pred_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate MAE, WAPE, and Normalized Bias globally or segmented by groups."""
    if group_cols is None:
        mae = calculate_mae(df[actual_col], df[pred_col])
        wape = calculate_wape(df[actual_col], df[pred_col])
        bias = calculate_bias(df[actual_col], df[pred_col])
        return pd.DataFrame([{
            "mae": round(mae, 4),
            "wape": round(wape, 4),
            "bias": round(bias, 4),
            "total_actual": float(df[actual_col].sum()),
            "total_pred": float(df[pred_col].sum()),
            "count": len(df),
        }])

    records = []
    for keys, grp in df.groupby(group_cols, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys, strict=False))
        rec["mae"] = round(calculate_mae(grp[actual_col], grp[pred_col]), 4)
        rec["wape"] = round(calculate_wape(grp[actual_col], grp[pred_col]), 4)
        rec["bias"] = round(calculate_bias(grp[actual_col], grp[pred_col]), 4)
        rec["total_actual"] = float(grp[actual_col].sum())
        rec["total_pred"] = float(grp[pred_col].sum())
        rec["count"] = len(grp)
        records.append(rec)

    return pd.DataFrame(records)
