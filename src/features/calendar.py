"""Calendar and holiday feature extraction."""

import numpy as np
import pandas as pd


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal, weekend, holiday, and SNAP features.

    All calendar features are strictly exogenous and known in advance.
    """
    out = df.copy()

    # Day of week (0=Monday, 6=Sunday)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(np.int8)

    # Cyclical day of week encoding (sin/cos to preserve periodic continuity)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7).astype(np.float32)

    # Month and cyclical month
    out["month_int"] = out["date"].dt.month.astype(np.int8)
    out["month_sin"] = np.sin(2 * np.pi * (out["month_int"] - 1) / 12).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * (out["month_int"] - 1) / 12).astype(np.float32)

    # Day of month
    out["day_of_month"] = out["date"].dt.day.astype(np.int8)

    # Weekend flag (Saturday=5, Sunday=6)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(np.int8)

    # Event / holiday presence
    if "event_name_1" in out.columns:
        out["has_event"] = (
            out["event_name_1"].notna() & (out["event_name_1"] != "")
        ).astype(np.int8)
    else:
        out["has_event"] = 0

    return out
