"""Heuristic baseline models for retail demand forecasting.

Baselines are the benchmark against which machine learning must justify its complexity:
1. Naive (Persistence): Repeats the most recent observed daily sale.
2. Seasonal Naive (s=7): Repeats the sales from the exact same day of the prior week.
3. Moving Average (7-day & 28-day): Constant flat forecast of recent historical mean.
"""

import numpy as np
import pandas as pd


class NaiveBaseline:
    """Naive persistence model: y_hat_{t+h} = y_t (last observed value)."""

    def __init__(self):
        self.last_values: dict[str, float] = {}

    def fit(self, history_df: pd.DataFrame) -> "NaiveBaseline":
        """Learn the last observed sales value for each series id."""
        # Sort chronologically
        sorted_df = history_df.sort_values(by=["id", "date"])
        last_rows = sorted_df.groupby("id", observed=False).last()
        self.last_values = last_rows["sales"].to_dict()
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict last observed sales for each row in test_df."""
        return np.array([self.last_values.get(sid, 0.0) for sid in test_df["id"]], dtype=np.float32)


class SeasonalNaiveBaseline:
    """Seasonal Naive model (s=7): y_hat_{t+h} = y_{t+h-7}.

    In retail, day-of-week seasonality is predominant (e.g. Saturday sales mirror
    prior Saturday sales). This baseline captures weekly cyclicality without training.
    """

    def __init__(self, seasonality: int = 7):
        self.seasonality = seasonality
        # Stores mapping: (series_id, day_of_week) -> last observed sale on that day of week
        self.seasonal_lookup: dict[tuple[str, int], float] = {}

    def fit(self, history_df: pd.DataFrame) -> "SeasonalNaiveBaseline":
        """Extract the most recent observation for each (series_id, day_of_week)."""
        df = history_df.copy()
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.dayofweek

        # Sort chronologically so last() gets the most recent occurrence of each day of week
        df.sort_values(by=["id", "date"], inplace=True)
        recent_by_dow = df.groupby(["id", "day_of_week"], observed=False).last().reset_index()

        self.seasonal_lookup = {
            (row["id"], row["day_of_week"]): float(row["sales"])
            for _, row in recent_by_dow.iterrows()
        }
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict sales matching each row's day of week from recent history."""
        df = test_df.copy()
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.dayofweek

        preds = np.zeros(len(df), dtype=np.float32)
        for idx, (_, row) in enumerate(df.iterrows()):
            key = (row["id"], row["day_of_week"])
            preds[idx] = self.seasonal_lookup.get(key, 0.0)

        return preds


class MovingAverageBaseline:
    """Moving Average model: y_hat_{t+h} = mean(y_{t-window+1 ... t})."""

    def __init__(self, window: int = 7):
        self.window = window
        self.mean_values: dict[str, float] = {}

    def fit(self, history_df: pd.DataFrame) -> "MovingAverageBaseline":
        """Calculate historical rolling mean over the last `window` days for each series."""
        df = history_df.sort_values(by=["id", "date"])
        # Take the tail(window) for each series and compute the mean
        tail_df = df.groupby("id", observed=False).tail(self.window)
        means = tail_df.groupby("id", observed=False)["sales"].mean()
        self.mean_values = means.to_dict()
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict flat historical moving average for each row in test_df."""
        return np.array([self.mean_values.get(sid, 0.0) for sid in test_df["id"]], dtype=np.float32)


def generate_all_baselines(history_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit all baseline models on history and predict on test_df."""
    naive = NaiveBaseline().fit(history_df)
    s_naive = SeasonalNaiveBaseline(seasonality=7).fit(history_df)
    ma_7 = MovingAverageBaseline(window=7).fit(history_df)
    ma_28 = MovingAverageBaseline(window=28).fit(history_df)

    return {
        "naive": naive.predict(test_df),
        "seasonal_naive_7d": s_naive.predict(test_df),
        "moving_average_7d": ma_7.predict(test_df),
        "moving_average_28d": ma_28.predict(test_df),
    }
