"""Evaluate heuristic baselines on a held-out 28-day forecast horizon."""

import json
from pathlib import Path

from tabulate import tabulate

from src.data.loader import DataLoader
from src.evaluation.metrics import evaluate_forecasts
from src.models.baselines import generate_all_baselines


def run_baseline_evaluation(
    data_dir: str | Path = "data/synthetic",
    horizon_days: int = 28,
    output_path: str | Path = "artifacts/baseline_benchmark.json",
) -> dict:
    """Run baseline models on historical panel data and report WAPE, MAE, and Bias."""
    loader = DataLoader(data_dir)
    cal, prices, sales = loader.load_raw()
    panel_df = loader.melt_and_merge(sales, cal, prices)

    # Chronological time-based split
    unique_dates = sorted(panel_df["date"].unique())
    if len(unique_dates) <= horizon_days:
        raise ValueError(f"Dataset has {len(unique_dates)} days, but horizon requires {horizon_days} days.")

    split_date = unique_dates[-horizon_days - 1]
    train_df = panel_df[panel_df["date"] <= split_date].copy()
    test_df = panel_df[panel_df["date"] > split_date].copy()

    # Generate baseline predictions
    predictions = generate_all_baselines(train_df, test_df)

    benchmark_results = {}
    table_rows = []

    for model_name, preds in predictions.items():
        eval_df = test_df.copy()
        eval_df["prediction"] = preds

        # Global metrics
        global_metrics = evaluate_forecasts(eval_df, actual_col="sales", pred_col="prediction")
        mae = float(global_metrics["mae"].iloc[0])
        wape = float(global_metrics["wape"].iloc[0])
        bias = float(global_metrics["bias"].iloc[0])

        # Category breakdown
        cat_metrics = evaluate_forecasts(
            eval_df, actual_col="sales", pred_col="prediction", group_cols=["cat_id"]
        )

        # Store breakdown
        store_metrics = evaluate_forecasts(
            eval_df, actual_col="sales", pred_col="prediction", group_cols=["store_id"]
        )

        benchmark_results[model_name] = {
            "global": {"mae": mae, "wape": wape, "bias": bias},
            "by_category": cat_metrics.to_dict(orient="records"),
            "by_store": store_metrics.to_dict(orient="records"),
        }

        table_rows.append([
            model_name,
            f"{mae:.4f}",
            f"{wape * 100:.2f}%",
            f"{bias * 100:+.2f}%",
        ])

    # Save to artifacts
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    headers = ["Baseline Model", "MAE (Units)", "WAPE (%)", "Bias (%)"]
    try:
        table_str = tabulate(table_rows, headers=headers, tablefmt="github")
    except Exception:
        # Fallback if tabulate is not installed
        header_str = " | ".join(headers)
        sep_str = " | ".join(["---"] * len(headers))
        data_strs = [" | ".join(row) for row in table_rows]
        table_str = "\n".join([header_str, sep_str] + data_strs)

    print("\n=== Baseline Benchmark Results (Held-Out 28-Day Horizon) ===")
    print(table_str)
    print(f"\nArtifact saved to: {out_file}\n")

    return benchmark_results


if __name__ == "__main__":
    run_baseline_evaluation()
