"""Lag and rolling statistics with strict multi-step horizon leakage guards."""

import numpy as np
import pandas as pd


def build_lag_features(
    df: pd.DataFrame,
    base_lag: int = 28,
    lag_periods: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Engineer historical lags and rolling aggregations with zero future leakage.

    CRITICAL LEAKAGE GUARD:
    For a forecast horizon h = 28 days, features evaluated at time t to predict
    [t ... t+27] must NOT observe any realized demand from [t-27 ... t].

    Therefore:
    1. Every lag must satisfy lag >= base_lag (>= 28).
    2. All rolling statistics are strictly calculated over `sales.shift(base_lag)`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel containing ['id', 'date', 'sales'].
    base_lag : int
        Minimum lag shift to eliminate horizon overlap (default 28).
    lag_periods : list[int]
        Lags to extract. Must all be >= base_lag. Default: [28, 35, 42, 49, 56].
    rolling_windows : list[int]
        Window lengths (in days) for rolling statistics over shifted sales. Default: [7, 14, 28].
    """
    if lag_periods is None:
        lag_periods = [28, 35, 42, 49, 56]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]

    # Validate leakage guardrail
    for lag in lag_periods:
        if lag < base_lag:
            raise ValueError(
                f"Leakage Violation: Lag {lag} is smaller than base_lag {base_lag}. "
                f"All lags must be >= {base_lag} to prevent horizon leakage."
            )

    out = df.copy()
    out.sort_values(by=["id", "date"], inplace=True)

    # 1. Direct Lags (grouped by time-series id)
    grouped = out.groupby("id", observed=False)["sales"]
    for lag in lag_periods:
        col_name = f"lag_{lag}"
        out[col_name] = grouped.shift(lag).astype(np.float32)

    # 2. Rolling Statistics on Base Shifted Sales
    # First obtain the base shifted series (shift=base_lag)
    shifted_base = grouped.shift(base_lag)

    for w in rolling_windows:
        # Grouped rolling mean
        mean_col = f"rolling_mean_{w}_lag_{base_lag}"
        out[mean_col] = (
            shifted_base.groupby(out["id"], observed=False)
            .transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean())
            .astype(np.float32)
        )

        # Grouped rolling std
        std_col = f"rolling_std_{w}_lag_{base_lag}"
        out[std_col] = (
            shifted_base.groupby(out["id"], observed=False)
            .transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).std().fillna(0.0))
            .astype(np.float32)
        )

    # Rolling min and max over longest window to capture recent bounds
    max_w = max(rolling_windows)
    out[f"rolling_min_{max_w}_lag_{base_lag}"] = (
        shifted_base.groupby(out["id"], observed=False)
        .transform(lambda s: s.rolling(max_w, min_periods=max(1, max_w // 2)).min())
        .astype(np.float32)
    )
    out[f"rolling_max_{max_w}_lag_{base_lag}"] = (
        shifted_base.groupby(out["id"], observed=False)
        .transform(lambda s: s.rolling(max_w, min_periods=max(1, max_w // 2)).max())
        .astype(np.float32)
    )

    return out
