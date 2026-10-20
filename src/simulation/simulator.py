"""Daily discrete-event inventory replenishment simulation engine."""

from collections import deque

import numpy as np
import pandas as pd

from src.simulation.policy import calculate_reorder_parameters


class InventorySimulator:
    """Simulates daily multi-echelon retail inventory dynamics under forecast-driven policies.

    Explicit Assumptions:
    1. Lost Sales: Unmet demand is permanently lost (walk-aways); no backorders.
    2. Deterministic Lead Time: Orders placed at day t arrive at the start of day t + L.
    3. Morning Delivery: Shipments arrive before retail store doors open for daily traffic.
    4. Perishable Degradation: Perishable goods (FOODS) incur daily shelf spoilage.
    """

    def __init__(
        self,
        lead_time_days: int = 3,
        service_level: float = 0.95,
        review_period_days: int = 1,
        holding_cost_rate: float = 0.02,
        stockout_penalty_multiplier: float = 1.5,
        daily_spoilage_rate_foods: float = 0.015,
        salvage_fraction: float = 0.20,
    ):
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.review_period_days = review_period_days
        self.holding_cost_rate = holding_cost_rate
        self.stockout_penalty_multiplier = stockout_penalty_multiplier
        self.daily_spoilage_rate_foods = daily_spoilage_rate_foods
        self.salvage_fraction = salvage_fraction

    def simulate_series(
        self,
        series_df: pd.DataFrame,
        forecast_col: str,
        initial_stock: float | None = None,
    ) -> pd.DataFrame:
        """Simulate single time-series SKU inventory trajectory across chronological dates."""
        df = series_df.sort_values("date").reset_index(drop=True)
        n_days = len(df)
        if n_days == 0:
            return pd.DataFrame()

        cat_id = str(df["cat_id"].iloc[0]) if "cat_id" in df.columns else "OTHER"
        unit_price = float(df["sell_price"].iloc[0]) if "sell_price" in df.columns else 2.0
        is_perishable = (cat_id == "FOODS")

        # Estimate daily forecast residual error std from data
        residuals = df["sales"].to_numpy() - df[forecast_col].to_numpy()
        sigma_e = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

        # Initial inventory: default to 1.5x lead-time demand or specified
        mean_forecast = float(df[forecast_col].mean())
        if initial_stock is None:
            on_hand = float(max(2.0, (self.lead_time_days + 1) * mean_forecast))
        else:
            on_hand = float(initial_stock)

        # In-transit pipeline: queue of (arrival_day_index, quantity)
        in_transit_pipeline: deque[tuple[int, float]] = deque()

        daily_records = []

        for t in range(n_days):
            row = df.iloc[t]
            cur_date = row["date"]
            actual_demand = float(row["sales"])
            cur_forecast = float(row[forecast_col])

            # 1. Receive Deliveries
            arrived_qty = 0.0
            remaining_pipeline: deque[tuple[int, float]] = deque()
            while in_transit_pipeline:
                arr_day, qty = in_transit_pipeline.popleft()
                if arr_day <= t:
                    arrived_qty += qty
                else:
                    remaining_pipeline.append((arr_day, qty))
            in_transit_pipeline = remaining_pipeline
            on_hand += arrived_qty

            # 2. Demand Fulfillment & Stockout Calculation
            fulfilled_sales = min(on_hand, actual_demand)
            lost_sales = max(0.0, actual_demand - on_hand)
            on_hand -= fulfilled_sales

            # 3. Spoilage for Perishables (FOODS)
            spoilage_units = 0.0
            if is_perishable and on_hand > 0:
                spoilage_units = round(on_hand * self.daily_spoilage_rate_foods, 2)
                on_hand = max(0.0, on_hand - spoilage_units)

            # 4. Inventory Position
            in_transit_qty = sum(q for _, q in in_transit_pipeline)
            inventory_position = on_hand + in_transit_qty

            # 5. Replenishment Review & Reorder Decision
            order_placed_qty = 0.0
            # Review decision triggered on review cycle
            if t % self.review_period_days == 0:
                upcoming_forecasts = df[forecast_col].iloc[t : t + self.lead_time_days + self.review_period_days].to_numpy()
                if len(upcoming_forecasts) == 0:
                    upcoming_forecasts = np.array([cur_forecast])

                _, rop, order_up_to = calculate_reorder_parameters(
                    forecast_horizon=upcoming_forecasts,
                    lead_time_days=self.lead_time_days,
                    review_period_days=self.review_period_days,
                    service_level=self.service_level,
                    sigma_daily_error=sigma_e,
                )

                if inventory_position <= rop:
                    order_placed_qty = float(np.ceil(max(0.0, order_up_to - inventory_position)))
                    if order_placed_qty > 0:
                        arrival_day = t + self.lead_time_days
                        in_transit_pipeline.append((arrival_day, order_placed_qty))

            # 6. Financial Economics
            holding_cost = round(on_hand * unit_price * self.holding_cost_rate, 4)
            stockout_cost = round(lost_sales * unit_price * self.stockout_penalty_multiplier, 4)
            spoilage_cost = round(spoilage_units * unit_price * (1.0 - self.salvage_fraction), 4)
            total_cost = round(holding_cost + stockout_cost + spoilage_cost, 4)

            daily_records.append({
                "id": row["id"],
                "item_id": row.get("item_id", ""),
                "dept_id": row.get("dept_id", ""),
                "cat_id": cat_id,
                "store_id": row.get("store_id", ""),
                "date": cur_date,
                "day_index": t,
                "demand": actual_demand,
                "forecast": cur_forecast,
                "arrived_qty": arrived_qty,
                "fulfilled_sales": fulfilled_sales,
                "lost_sales": lost_sales,
                "spoilage_units": spoilage_units,
                "on_hand": round(on_hand, 2),
                "in_transit": round(in_transit_qty, 2),
                "order_placed": order_placed_qty,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "spoilage_cost": spoilage_cost,
                "total_cost": total_cost,
            })

        return pd.DataFrame(daily_records)

    def simulate_panel(
        self,
        panel_df: pd.DataFrame,
        forecast_col: str,
    ) -> pd.DataFrame:
        """Simulate all SKUs in panel across chronological dates."""
        results = []
        for _, grp in panel_df.groupby("id", observed=False):
            sim_series = self.simulate_series(grp, forecast_col=forecast_col)
            results.append(sim_series)
        return pd.concat(results, ignore_index=True)
