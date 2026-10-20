"""Unit tests for replenishment policies and inventory simulation balance equations."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.policy import calculate_reorder_parameters, calculate_safety_stock
from src.simulation.simulator import InventorySimulator


def test_safety_stock_properties():
    """Verify safety stock increases monotonically with service level and lead time."""
    ss_90 = calculate_safety_stock(service_level=0.90, lead_time_days=3, sigma_daily_error=2.0)
    ss_95 = calculate_safety_stock(service_level=0.95, lead_time_days=3, sigma_daily_error=2.0)
    ss_99 = calculate_safety_stock(service_level=0.99, lead_time_days=3, sigma_daily_error=2.0)

    assert ss_90 < ss_95 < ss_99, "Safety stock must increase with higher target service level"

    ss_lt1 = calculate_safety_stock(service_level=0.95, lead_time_days=1, sigma_daily_error=2.0)
    ss_lt7 = calculate_safety_stock(service_level=0.95, lead_time_days=7, sigma_daily_error=2.0)
    assert ss_lt1 < ss_lt7, "Safety stock must increase with longer lead time"


def test_reorder_parameters_relationships():
    """Verify ROP and Order-up-to level relationships."""
    forecast = np.array([10.0] * 14)
    ss, rop, order_up_to = calculate_reorder_parameters(
        forecast_horizon=forecast,
        lead_time_days=3,
        review_period_days=7,
        service_level=0.95,
        sigma_daily_error=1.5,
    )

    # Lead time demand = 3 * 10 = 30
    assert pytest.approx(rop, abs=0.1) == 30.0 + ss
    # Protection period = 10 days * 10 = 100
    assert order_up_to > rop, "Order-up-to level must exceed reorder point"


def test_inventory_conservation_of_mass():
    """Verify fundamental inventory balance equation holds on every simulation step:

    On-Hand(t) == On-Hand(t-1) + Arrived(t) - Fulfilled(t) - Spoiled(t)
    """
    n_days = 20
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    series_df = pd.DataFrame({
        "id": ["item_test"] * n_days,
        "cat_id": ["FOODS"] * n_days,
        "date": dates,
        "sales": [5, 0, 12, 4, 0, 20, 8, 2, 1, 0, 15, 6, 3, 0, 10, 5, 2, 8, 0, 4],
        "forecast": [6.0] * n_days,
        "sell_price": [3.5] * n_days,
    })

    sim = InventorySimulator(lead_time_days=2, service_level=0.95, review_period_days=1)
    res = sim.simulate_series(series_df, forecast_col="forecast", initial_stock=15.0)

    # Initial stock was 15.0
    prev_on_hand = 15.0
    for t in range(n_days):
        row = res.iloc[t]
        arrived = row["arrived_qty"]
        fulfilled = row["fulfilled_sales"]
        spoiled = row["spoilage_units"]
        expected_on_hand = prev_on_hand + arrived - fulfilled - spoiled

        np.testing.assert_allclose(
            row["on_hand"],
            expected_on_hand,
            atol=0.01,
            err_msg=f"Conservation of mass failed on day {t}",
        )
        prev_on_hand = row["on_hand"]


def test_stockout_and_fulfillment_bounds():
    """Verify fulfilled sales and lost sales never exceed actual demand."""
    n_days = 10
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    series_df = pd.DataFrame({
        "id": ["sku_bounds"] * n_days,
        "cat_id": ["HOUSEHOLD"] * n_days,
        "date": dates,
        "sales": [10, 2, 0, 15, 1, 0, 30, 5, 0, 8],
        "forecast": [5.0] * n_days,
        "sell_price": [12.0] * n_days,
    })

    sim = InventorySimulator(lead_time_days=3, service_level=0.90)
    res = sim.simulate_series(series_df, forecast_col="forecast", initial_stock=5.0)

    for _, row in res.iterrows():
        assert row["fulfilled_sales"] <= row["demand"]
        assert row["lost_sales"] <= row["demand"]
        assert pytest.approx(row["fulfilled_sales"] + row["lost_sales"], abs=1e-4) == row["demand"]
