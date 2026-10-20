"""Inventory replenishment policy: Safety stock, Reorder Point, and Order quantities."""

import numpy as np
import scipy.stats as stats


def calculate_safety_stock(
    service_level: float,
    lead_time_days: int,
    sigma_daily_error: float,
) -> float:
    """Calculate statistical safety stock for target cycle service level.

    Formula:
        SS = z * sigma_LTD
        where z = Phi^{-1}(Service Level)
        and sigma_LTD = sqrt(Lead Time) * sigma_daily_error
    """
    if service_level >= 1.0 or service_level <= 0.0:
        raise ValueError(f"Service level must be strictly between 0 and 1, got {service_level}")

    # Standard normal quantile (z-score)
    z = float(stats.norm.ppf(service_level))
    sigma_ltd = np.sqrt(lead_time_days) * max(0.01, sigma_daily_error)
    return float(max(0.0, z * sigma_ltd))


def calculate_reorder_parameters(
    forecast_horizon: np.ndarray | list[float],
    lead_time_days: int,
    review_period_days: int,
    service_level: float,
    sigma_daily_error: float,
) -> tuple[float, float, float]:
    """Calculate Reorder Point (s) and Order-Up-To Level (S) for (R, s, S) periodic review.

    Parameters
    ----------
    forecast_horizon : np.ndarray
        Vector of daily demand forecasts for the upcoming horizon [t+1 ... t+H].
    lead_time_days : int
        Supplier delivery lead time in days (L).
    review_period_days : int
        Review interval in days (R).
    service_level : float
        Target cycle service level (e.g. 0.95 for 95%).
    sigma_daily_error : float
        Standard deviation of daily forecast residual errors.

    Returns
    -------
    tuple[float, float, float]
        (safety_stock, reorder_point, order_up_to_level)
    """
    f_arr = np.asarray(forecast_horizon, dtype=np.float64)
    total_horizon = len(f_arr)

    # Lead time demand: sum of forecasts over lead time window [1 ... L]
    ltd_days = min(lead_time_days, total_horizon)
    ltd = float(np.sum(f_arr[:ltd_days]))

    # Protection period = Review Period + Lead Time (R + L)
    protection_days = min(review_period_days + lead_time_days, total_horizon)
    protection_demand = float(np.sum(f_arr[:protection_days]))

    # Safety stock for lead time
    ss = calculate_safety_stock(
        service_level=service_level,
        lead_time_days=lead_time_days,
        sigma_daily_error=sigma_daily_error,
    )

    # Safety stock for protection period
    ss_protection = calculate_safety_stock(
        service_level=service_level,
        lead_time_days=review_period_days + lead_time_days,
        sigma_daily_error=sigma_daily_error,
    )

    reorder_point = ltd + ss
    order_up_to_level = protection_demand + ss_protection

    return float(ss), float(reorder_point), float(order_up_to_level)
