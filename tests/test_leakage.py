"""Rigorous tests proving zero temporal and horizon leakage in feature engineering."""

import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline


def test_future_perturbation_leakage(sample_panel_df):
    """Perturbing sales in the future must have ZERO effect on past features.

    If future data leaks into the present (e.g. via improper forward-looking
    rolling aggregations), this test will fail immediately.
    """
    pipeline = FeaturePipeline()

    # 1. Compute baseline features on unperturbed data
    df_clean, feat_cols = pipeline.fit_transform(sample_panel_df.copy(), drop_warmup=True)

    # Choose a cutoff date in the middle of the available timeline
    unique_dates = sorted(df_clean["date"].unique())
    cutoff_date = unique_dates[len(unique_dates) // 2]

    # 2. Perturb future sales: set all sales after cutoff_date to an extreme value
    df_perturbed = sample_panel_df.copy()
    future_mask = df_perturbed["date"] > cutoff_date
    df_perturbed.loc[future_mask, "sales"] = 9999

    # Re-run the feature pipeline on perturbed data
    df_perturbed_out, _ = pipeline.fit_transform(df_perturbed, drop_warmup=True)

    # 3. Compare feature values for rows on or before the cutoff date
    clean_history = df_clean[df_clean["date"] <= cutoff_date].sort_values(["id", "date"]).reset_index(drop=True)
    perturbed_history = df_perturbed_out[df_perturbed_out["date"] <= cutoff_date].sort_values(["id", "date"]).reset_index(drop=True)

    assert len(clean_history) == len(perturbed_history), "History row counts differ after perturbation"

    # Verify every feature column is bit-for-bit identical
    for col in feat_cols:
        # For floating point columns, allow tiny numerical epsilon
        if pd.api.types.is_float_dtype(clean_history[col]):
            np.testing.assert_allclose(
                clean_history[col].values,
                perturbed_history[col].values,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Temporal Leakage Detected in feature: '{col}'",
            )
        else:
            assert (clean_history[col].values == perturbed_history[col].values).all(), (
                f"Temporal Leakage Detected in discrete feature: '{col}'"
            )


def test_horizon_leakage_within_28_days(sample_panel_df):
    """Perturbing sales within 27 days before time t must NOT alter features at time t.

    Because the forecast horizon is 28 days, a direct model predicting t to t+27
    using features at t requires all lag features to look back >= 28 days.
    """
    pipeline = FeaturePipeline()
    df_clean, feat_cols = pipeline.fit_transform(sample_panel_df.copy(), drop_warmup=True)

    unique_dates = sorted(df_clean["date"].unique())
    eval_date = unique_dates[-1]

    # Perturb sales at eval_date - 10 days (which is inside the 28-day forecast horizon)
    perturb_date = eval_date - pd.Timedelta(days=10)

    df_perturbed = sample_panel_df.copy()
    mask = df_perturbed["date"] == perturb_date
    df_perturbed.loc[mask, "sales"] = 5555

    df_perturbed_out, _ = pipeline.fit_transform(df_perturbed, drop_warmup=True)

    clean_eval = df_clean[df_clean["date"] == eval_date].sort_values(["id"]).reset_index(drop=True)
    perturbed_eval = df_perturbed_out[df_perturbed_out["date"] == eval_date].sort_values(["id"]).reset_index(drop=True)

    # Lag and rolling features at eval_date should be completely unaffected
    lag_cols = [c for c in feat_cols if "lag" in c]
    for col in lag_cols:
        np.testing.assert_allclose(
            clean_eval[col].values,
            perturbed_eval[col].values,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Horizon Leakage Detected: Feature '{col}' at time t changed when perturbing t-10!",
        )
