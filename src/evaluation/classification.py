"""Demand profiling and Syntetos-Boylan demand pattern classification."""

import numpy as np
import pandas as pd


def classify_demand_patterns(
    df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> pd.DataFrame:
    """Classify retail time series into the 4 Syntetos-Boylan quadrants.

    Parameters
    ----------
    df : pd.DataFrame
        Historical panel containing ['id', 'sales'].
    adi_threshold : float
        Average Demand Interval boundary (default 1.32).
        ADI = Total periods / Non-zero demand periods.
    cv2_threshold : float
        Squared Coefficient of Variation boundary (default 0.49).
        CV^2 = (std(non-zero sales) / mean(non-zero sales))^2.

    Quadrants:
    1. Smooth (ADI < 1.32, CV^2 < 0.49): Frequent, low-variance demand. ML models thrive here.
    2. Intermittent (ADI >= 1.32, CV^2 < 0.49): Sporadic arrivals with uniform basket sizes.
    3. Erratic (ADI < 1.32, CV^2 >= 0.49): Frequent arrivals with highly variable basket sizes.
    4. Lumpy (ADI >= 1.32, CV^2 >= 0.49): Sporadic arrivals with highly variable order quantities.
       Heuristic models (e.g. Croston, MA) often outperform ML here.
    """
    records = []

    for series_id, grp in df.groupby("id", observed=False):
        sales_arr = grp["sales"].to_numpy(dtype=np.float64)
        total_periods = len(sales_arr)
        non_zero_sales = sales_arr[sales_arr > 0]
        non_zero_count = len(non_zero_sales)

        if non_zero_count == 0:
            # Entirely dead product
            adi = float(total_periods)
            cv2 = 0.0
            demand_type = "Intermittent"
        else:
            adi = total_periods / non_zero_count
            if non_zero_count > 1:
                mean_nz = np.mean(non_zero_sales)
                std_nz = np.std(non_zero_sales, ddof=1)
                cv2 = (std_nz / (mean_nz + 1e-5)) ** 2
            else:
                cv2 = 0.0

            # Syntetos-Boylan categorization
            if adi < adi_threshold and cv2 < cv2_threshold:
                demand_type = "Smooth"
            elif adi < adi_threshold and cv2 >= cv2_threshold:
                demand_type = "Erratic"
            elif adi >= adi_threshold and cv2 < cv2_threshold:
                demand_type = "Intermittent"
            else:
                demand_type = "Lumpy"

        records.append({
            "id": series_id,
            "total_periods": total_periods,
            "non_zero_periods": non_zero_count,
            "zero_percentage": round((1.0 - non_zero_count / total_periods) * 100, 2),
            "adi": round(float(adi), 3),
            "cv2": round(float(cv2), 3),
            "demand_type": demand_type,
        })

    return pd.DataFrame(records)
