"""Data validation and sanity checking for raw M5 and processed feature tables."""

from pathlib import Path

import pandas as pd


class DataValidationError(Exception):
    """Raised when dataset sanity checks or schema requirements fail."""
    pass


def validate_raw_data(data_dir: str | Path) -> dict[str, bool]:
    """Validate raw M5 CSV files: calendar.csv, sell_prices.csv, sales_train_*.csv.

    Raises DataValidationError if critical schema or business constraints are violated.
    """
    path = Path(data_dir)

    # 1. Check file existence
    cal_file = path / "calendar.csv"
    price_file = path / "sell_prices.csv"

    # Check for evaluation or validation sales file
    sales_file = path / "sales_train_evaluation.csv"
    if not sales_file.exists():
        sales_file = path / "sales_train_validation.csv"

    if not cal_file.exists():
        raise DataValidationError(f"Missing calendar file: {cal_file}")
    if not price_file.exists():
        raise DataValidationError(f"Missing prices file: {price_file}")
    if not sales_file.exists():
        raise DataValidationError(f"Missing sales file in {data_dir} (expected sales_train_evaluation.csv or sales_train_validation.csv)")

    # 2. Validate Calendar
    cal_df = pd.read_csv(cal_file, nrows=100)
    required_cal_cols = {"date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"}
    missing_cal = required_cal_cols - set(cal_df.columns)
    if missing_cal:
        raise DataValidationError(f"Calendar file missing required columns: {missing_cal}")

    # Validate calendar date parsing and sequence
    full_cal = pd.read_csv(cal_file, usecols=["date", "d"])
    parsed_dates = pd.to_datetime(full_cal["date"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        raise DataValidationError("Calendar contains invalid or unparseable dates.")

    # Ensure dates are monotonically increasing
    if not parsed_dates.is_monotonic_increasing:
        raise DataValidationError("Calendar dates are not strictly monotonic increasing.")

    # 3. Validate Sell Prices
    price_df = pd.read_csv(price_file, nrows=100)
    required_price_cols = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    missing_price = required_price_cols - set(price_df.columns)
    if missing_price:
        raise DataValidationError(f"Sell prices file missing required columns: {missing_price}")

    # Quick sanity check on price bounds
    sample_prices = pd.read_csv(price_file, usecols=["sell_price"], nrows=1000)
    if (sample_prices["sell_price"] <= 0).any():
        raise DataValidationError("Sell prices contain non-positive values (<= 0).")

    # 4. Validate Sales
    sales_df_sample = pd.read_csv(sales_file, nrows=50)
    required_sales_metadata = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing_sales_meta = required_sales_metadata - set(sales_df_sample.columns)
    if missing_sales_meta:
        raise DataValidationError(f"Sales file missing metadata columns: {missing_sales_meta}")

    # Verify day columns pattern d_1, d_2 ...
    day_cols = [c for c in sales_df_sample.columns if c.startswith("d_")]
    if not day_cols:
        raise DataValidationError("Sales file contains no 'd_' day columns.")

    # Check for negative sales
    if (sales_df_sample[day_cols] < 0).any().any():
        raise DataValidationError("Sales file contains negative sales values.")

    return {
        "calendar_valid": True,
        "sell_prices_valid": True,
        "sales_valid": True,
        "total_days": len(full_cal),
        "total_series": len(sales_df_sample),
    }


def validate_processed_data(df: pd.DataFrame, feature_cols: list[str] | None = None) -> bool:
    """Validate processed feature dataset before model consumption."""
    required_keys = {"id", "date", "sales"}
    missing = required_keys - set(df.columns)
    if missing:
        raise DataValidationError(f"Processed dataset missing required keys: {missing}")

    # Check for negative sales in target
    if (df["sales"] < 0).any():
        raise DataValidationError("Processed dataset target 'sales' contains negative values.")

    # Check that feature columns do not contain NaNs in valid training rows
    if feature_cols:
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise DataValidationError(f"Missing declared feature columns: {missing_features}")

        # Identify features with high null rate
        null_counts = df[feature_cols].isna().sum()
        high_nulls = null_counts[null_counts > 0.05 * len(df)]
        if not high_nulls.empty:
            raise DataValidationError(f"Feature columns have excessive (>5%) NaNs: {high_nulls.to_dict()}")

    return True
