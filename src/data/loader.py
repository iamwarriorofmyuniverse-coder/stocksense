"""Data loader for Kaggle M5 and synthetic datasets with memory downcasting."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validator import validate_raw_data


class DataLoader:
    """Loads and formats M5 retail time-series tables."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        validate_raw_data(self.data_dir)

    def load_raw(
        self,
        sample_stores: list[str] | None = None,
        max_items_per_category: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load calendar, sell_prices, and sales_train CSVs with downcasting."""
        # 1. Calendar
        cal_path = self.data_dir / "calendar.csv"
        cal_dtypes = {
            "wm_yr_wk": "int32",
            "wday": "int8",
            "month": "int8",
            "year": "int16",
            "event_name_1": "category",
            "event_type_1": "category",
            "event_name_2": "category",
            "event_type_2": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        }
        calendar_df = pd.read_csv(cal_path, dtype=cal_dtypes)
        calendar_df["date"] = pd.to_datetime(calendar_df["date"], format="%Y-%m-%d")

        # 2. Sell Prices
        prices_path = self.data_dir / "sell_prices.csv"
        price_dtypes = {
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int32",
            "sell_price": "float32",
        }
        prices_df = pd.read_csv(prices_path, dtype=price_dtypes)

        # 3. Sales
        sales_path = self.data_dir / "sales_train_evaluation.csv"
        if not sales_path.exists():
            sales_path = self.data_dir / "sales_train_validation.csv"

        sales_df = pd.read_csv(sales_path)

        # Filter stores if requested
        if sample_stores:
            sales_df = sales_df[sales_df["store_id"].isin(sample_stores)].copy()
            prices_df = prices_df[prices_df["store_id"].isin(sample_stores)].copy()

        # Stratified sampling of items per category if requested
        if max_items_per_category and max_items_per_category > 0:
            selected_ids = []
            for _, group in sales_df.groupby(["store_id", "cat_id"]):
                selected_ids.extend(group["id"].head(max_items_per_category).tolist())
            sales_df = sales_df[sales_df["id"].isin(selected_ids)].copy()
            prices_df = prices_df[prices_df["item_id"].isin(sales_df["item_id"].unique())].copy()

        return calendar_df, prices_df, sales_df

    @staticmethod
    def melt_and_merge(
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_d: int = 1,
        end_d: int | None = None,
    ) -> pd.DataFrame:
        """Unpivot wide sales into long time-series format and merge calendar + prices.

        Downcasts integer and float columns to minimize memory footprint.
        """
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # Determine day columns to melt
        all_day_cols = [c for c in sales_df.columns if c.startswith("d_")]
        if end_d is not None:
            target_cols = [f"d_{i}" for i in range(start_d, end_d + 1) if f"d_{i}" in sales_df.columns]
        else:
            target_cols = [c for c in all_day_cols if int(c.split("_")[1]) >= start_d]

        # Melt sales to long format: (id, d, sales)
        melted = pd.melt(
            sales_df,
            id_vars=id_vars,
            value_vars=target_cols,
            var_name="d",
            value_name="sales",
        )
        melted["sales"] = melted["sales"].astype(np.int16)

        # Merge calendar
        cal_cols = [
            "d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"
        ]
        cal_subset = calendar_df[[c for c in cal_cols if c in calendar_df.columns]]
        merged = pd.merge(melted, cal_subset, on="d", how="left")

        # Resolve state-specific SNAP column into a single boolean feature
        merged["is_snap"] = 0
        for state in ["CA", "TX", "WI"]:
            snap_col = f"snap_{state}"
            if snap_col in merged.columns:
                mask = merged["state_id"] == state
                merged.loc[mask, "is_snap"] = merged.loc[mask, snap_col].fillna(0).astype(np.int8)

        # Merge sell prices
        merged = pd.merge(
            merged,
            prices_df,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )

        # If sell_price is missing, impute with item-level mean price
        if merged["sell_price"].isna().any():
            mean_item_price = merged.groupby("item_id")["sell_price"].transform("mean")
            merged["sell_price"] = merged["sell_price"].fillna(mean_item_price).fillna(1.0).astype(np.float32)

        # Drop temporary SNAP columns
        cols_to_drop = [c for c in ["snap_CA", "snap_TX", "snap_WI"] if c in merged.columns]
        if cols_to_drop:
            merged.drop(columns=cols_to_drop, inplace=True)

        # Downcast categorical columns
        for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            merged[col] = merged[col].astype("category")

        # Sort chronologically per time series
        merged.sort_values(by=["id", "date"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        return merged
