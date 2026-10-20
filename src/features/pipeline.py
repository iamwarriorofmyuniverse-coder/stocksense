"""Feature engineering pipeline orchestrating calendar, price, and lag transformations."""

from pathlib import Path

import pandas as pd
import yaml

from src.features.calendar import build_calendar_features
from src.features.lags import build_lag_features
from src.features.price import build_price_features


class FeaturePipeline:
    """End-to-end feature engineering pipeline for multi-store retail panel data."""

    def __init__(self, config_path: str | Path | None = None):
        self.config = self._load_config(config_path)
        self.feature_names: list[str] = []
        self.categorical_cols: list[str] = self.config.get("categoricals", [
            "item_id", "dept_id", "cat_id", "store_id", "state_id", "wday", "month"
        ])

    @staticmethod
    def _load_config(config_path: str | Path | None) -> dict:
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "lags": {"periods": [28, 35, 42, 49, 56]},
            "rolling_statistics": {"base_lag": 28, "windows": [7, 14, 28]},
            "categoricals": ["item_id", "dept_id", "cat_id", "store_id", "state_id", "wday", "month"],
        }

    def fit_transform(self, df: pd.DataFrame, drop_warmup: bool = True) -> tuple[pd.DataFrame, list[str]]:
        """Fit feature definitions and transform panel DataFrame into feature matrix."""
        # 1. Calendar Features
        out = build_calendar_features(df)

        # 2. Price Dynamics & Promotional Proxies
        out = build_price_features(out)

        # 3. Strict Horizon Lags & Rolling Statistics
        lag_cfg = self.config.get("lags", {})
        periods = lag_cfg.get("periods", [28, 35, 42, 49, 56])
        rolling_cfg = self.config.get("rolling_statistics", {})
        base_lag = rolling_cfg.get("base_lag", 28)
        windows = rolling_cfg.get("windows", [7, 14, 28])

        out = build_lag_features(
            out,
            base_lag=base_lag,
            lag_periods=periods,
            rolling_windows=windows,
        )

        # 4. Handle Categoricals
        for cat in self.categorical_cols:
            if cat in out.columns:
                out[cat] = out[cat].astype("category")

        # 5. Drop Warmup Period (initial rows where longest lag is NaN)
        if drop_warmup:
            longest_lag = max(periods)
            # Find earliest date that has valid longest_lag for all series
            out = out.dropna(subset=[f"lag_{longest_lag}"]).copy()
            out.reset_index(drop=True, inplace=True)

        # Define feature column list (exclude metadata and target)
        exclude_cols = {
            "id", "date", "d", "wm_yr_wk", "sales", "weekday",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"
        }
        self.feature_names = [c for c in out.columns if c not in exclude_cols]

        return out, self.feature_names
