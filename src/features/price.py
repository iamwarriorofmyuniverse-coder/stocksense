"""Price dynamic features and promotional proxies."""

import numpy as np
import pandas as pd


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price momentum, discount depth, and price volatility features.

    In retail forecasting, relative price and promotion depth are strong demand drivers.
    Computed within each (store_id, item_id) series.
    """
    out = df.copy()

    # Ensure required columns exist
    if "sell_price" not in out.columns:
        raise KeyError("DataFrame must contain 'sell_price' to extract price features.")

    # Calculate item-level historical maximum price to compute discount proxy
    max_price = out.groupby(["store_id", "item_id"], observed=False)["sell_price"].transform("max")
    out["price_max"] = max_price.astype(np.float32)

    # Discount percentage: (max_price - current_price) / max_price
    # Clip between 0 and 1
    discount = (out["price_max"] - out["sell_price"]) / (out["price_max"] + 1e-5)
    out["price_discount"] = np.clip(discount, 0.0, 1.0).astype(np.float32)

    # Price momentum: ratio of current price to rolling 4-week mean price
    # Sort chronologically
    out.sort_values(by=["id", "date"], inplace=True)

    # Rolling 28-day price mean per series
    rolling_price_28 = (
        out.groupby("id", observed=False)["sell_price"]
        .transform(lambda s: s.rolling(28, min_periods=1).mean())
    )
    out["price_momentum"] = (out["sell_price"] / (rolling_price_28 + 1e-5)).astype(np.float32)

    # Binary promotion flag (discount >= 10%)
    out["is_promo"] = (out["price_discount"] >= 0.10).astype(np.int8)

    return out
