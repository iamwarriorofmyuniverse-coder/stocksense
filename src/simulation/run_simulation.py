"""Head-to-head replenishment simulation: Baseline Policy vs Model Policy."""

import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from src.simulation.simulator import InventorySimulator


def run_replenishment_simulation(
    predictions_path: str | Path = "artifacts/backtest_predictions.parquet",
    output_dir: str | Path = "artifacts",
    lead_time_days: int = 3,
    service_level: float = 0.95,
    review_period_days: int = 1,
) -> dict:
    """Execute side-by-side inventory simulation comparing baseline vs LightGBM policies."""
    preds_file = Path(predictions_path)
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_file}. Train model first.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_preds = pd.read_parquet(preds_file)

    # Simulate over the final held-out 28-day window (Fold 3)
    max_fold = df_preds["fold"].max()
    test_window = df_preds[df_preds["fold"] == max_fold].copy()

    simulator = InventorySimulator(
        lead_time_days=lead_time_days,
        service_level=service_level,
        review_period_days=review_period_days,
    )

    print(f"\n=== Simulating Inventory Replenishment (Lead Time: {lead_time_days}d, Target SL: {service_level*100:.0f}%) ===")
    print("Running Baseline Replenishment Policy (Moving Average 28d)...")
    baseline_sim = simulator.simulate_panel(test_window, forecast_col="pred_moving_average_28d")
    baseline_sim["policy"] = "baseline_ma28"

    print("Running Machine Learning Replenishment Policy (LightGBM)...")
    lgbm_sim = simulator.simulate_panel(test_window, forecast_col="pred_lgbm")
    lgbm_sim["policy"] = "model_lgbm"

    # Combine simulation paths
    combined_sim = pd.concat([baseline_sim, lgbm_sim], ignore_index=True)
    combined_sim.to_parquet(out_path / "simulation_results.parquet", index=False)

    # Compute KPI Summaries
    def _compute_kpis(sim_df: pd.DataFrame) -> dict:
        tot_demand = float(sim_df["demand"].sum())
        tot_fulfilled = float(sim_df["fulfilled_sales"].sum())
        tot_lost = float(sim_df["lost_sales"].sum())
        tot_spoilage = float(sim_df["spoilage_units"].sum())
        avg_on_hand = float(sim_df["on_hand"].mean())
        tot_orders = float(sim_df["order_placed"].sum())

        tot_holding = float(sim_df["holding_cost"].sum())
        tot_stockout = float(sim_df["stockout_cost"].sum())
        tot_spoilage_cost = float(sim_df["spoilage_cost"].sum())
        tot_cost = float(sim_df["total_cost"].sum())

        fill_rate = (tot_fulfilled / (tot_demand + 1e-5)) * 100.0
        stockout_rate = (tot_lost / (tot_demand + 1e-5)) * 100.0

        return {
            "total_demand": tot_demand,
            "fulfilled_units": tot_fulfilled,
            "lost_sales_units": tot_lost,
            "spoilage_units": tot_spoilage,
            "avg_on_hand_units": round(avg_on_hand, 2),
            "fill_rate_pct": round(fill_rate, 2),
            "stockout_rate_pct": round(stockout_rate, 2),
            "total_orders_placed": tot_orders,
            "holding_cost_usd": round(tot_holding, 2),
            "stockout_penalty_usd": round(tot_stockout, 2),
            "spoilage_cost_usd": round(tot_spoilage_cost, 2),
            "total_operational_cost_usd": round(tot_cost, 2),
        }

    kpi_baseline = _compute_kpis(baseline_sim)
    kpi_lgbm = _compute_kpis(lgbm_sim)

    # Compute percentage changes (ML vs Baseline)
    stockout_reduction_pct = (
        (kpi_baseline["lost_sales_units"] - kpi_lgbm["lost_sales_units"])
        / (kpi_baseline["lost_sales_units"] + 1e-5)
    ) * 100.0

    cost_reduction_pct = (
        (kpi_baseline["total_operational_cost_usd"] - kpi_lgbm["total_operational_cost_usd"])
        / (kpi_baseline["total_operational_cost_usd"] + 1e-5)
    ) * 100.0

    inventory_reduction_pct = (
        (kpi_baseline["avg_on_hand_units"] - kpi_lgbm["avg_on_hand_units"])
        / (kpi_baseline["avg_on_hand_units"] + 1e-5)
    ) * 100.0

    summary_report = {
        "parameters": {
            "lead_time_days": lead_time_days,
            "target_service_level": service_level,
            "review_period_days": review_period_days,
            "simulated_days": 28,
            "num_skus": len(test_window["id"].unique()),
        },
        "baseline_policy": kpi_baseline,
        "model_policy": kpi_lgbm,
        "operational_delta": {
            "stockout_reduction_pct": round(stockout_reduction_pct, 2),
            "cost_reduction_pct": round(cost_reduction_pct, 2),
            "avg_inventory_change_pct": round(inventory_reduction_pct, 2),
        },
    }

    # Save summary
    summary_file = out_path / "simulation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Print Executive Comparison Table
    kpi_rows = [
        ["Cycle Service Level (Target)", f"{service_level*100:.0f}%", f"{service_level*100:.0f}%", "-"],
        ["Realized Fill Rate", f"{kpi_baseline['fill_rate_pct']:.2f}%", f"{kpi_lgbm['fill_rate_pct']:.2f}%", f"{kpi_lgbm['fill_rate_pct'] - kpi_baseline['fill_rate_pct']:+.2f}%"],
        ["Stockout Rate (% Demand)", f"{kpi_baseline['stockout_rate_pct']:.2f}%", f"{kpi_lgbm['stockout_rate_pct']:.2f}%", f"{kpi_lgbm['stockout_rate_pct'] - kpi_baseline['stockout_rate_pct']:+.2f}%"],
        ["Lost Sales Units", f"{kpi_baseline['lost_sales_units']:.0f}", f"{kpi_lgbm['lost_sales_units']:.0f}", f"{kpi_lgbm['lost_sales_units'] - kpi_baseline['lost_sales_units']:+.0f}"],
        ["Spoilage Units (Waste)", f"{kpi_baseline['spoilage_units']:.1f}", f"{kpi_lgbm['spoilage_units']:.1f}", f"{kpi_lgbm['spoilage_units'] - kpi_baseline['spoilage_units']:+.1f}"],
        ["Average Daily On-Hand", f"{kpi_baseline['avg_on_hand_units']:.2f} units", f"{kpi_lgbm['avg_on_hand_units']:.2f} units", f"{-inventory_reduction_pct:+.2f}%"],
        ["Total Holding Cost", f"${kpi_baseline['holding_cost_usd']:.2f}", f"${kpi_lgbm['holding_cost_usd']:.2f}", f"${kpi_lgbm['holding_cost_usd'] - kpi_baseline['holding_cost_usd']:+.2f}"],
        ["Total Stockout Penalties", f"${kpi_baseline['stockout_penalty_usd']:.2f}", f"${kpi_lgbm['stockout_penalty_usd']:.2f}", f"${kpi_lgbm['stockout_penalty_usd'] - kpi_baseline['stockout_penalty_usd']:+.2f}"],
        ["Total Operational Cost", f"${kpi_baseline['total_operational_cost_usd']:.2f}", f"${kpi_lgbm['total_operational_cost_usd']:.2f}", f"${kpi_lgbm['total_operational_cost_usd'] - kpi_baseline['total_operational_cost_usd']:+.2f} ({cost_reduction_pct:+.1f}%)"],
    ]
    headers = ["Supply Chain Metric", "Baseline Policy (MA-28)", "Model Policy (LightGBM)", "Impact (Delta)"]
    print("\n=== Replenishment Policy Simulation Results (28-Day Held-Out Window) ===")
    print(tabulate(kpi_rows, headers=headers, tablefmt="github"))
    print(f"\nArtifact saved to: {summary_file}")
    print(f"Trajectory parquet saved to: {out_path / 'simulation_results.parquet'}\n")

    return summary_report


if __name__ == "__main__":
    run_replenishment_simulation()
