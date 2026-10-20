"""Granular segmented evaluation and failure analysis engine."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.evaluation.classification import classify_demand_patterns
from src.evaluation.metrics import evaluate_forecasts


def run_failure_analysis(
    predictions_path: str | Path = "artifacts/backtest_predictions.parquet",
    output_dir: str | Path = "artifacts",
) -> dict:
    """Analyze multi-fold backtest predictions across segments and document model failure modes."""
    preds_file = Path(predictions_path)
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_file}. Run training first.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_preds = pd.read_parquet(preds_file)

    # 1. Classify demand patterns for each time series
    print("Classifying demand velocity profiles (Syntetos-Boylan)...")
    demand_profiles = classify_demand_patterns(df_preds)

    # Merge demand type into predictions
    merged_preds = pd.merge(
        df_preds,
        demand_profiles[["id", "demand_type", "adi", "cv2", "zero_percentage"]],
        on="id",
        how="left",
    )

    models = ["lgbm", "moving_average_28d", "moving_average_7d", "naive", "seasonal_naive_7d"]
    available_models = [m for m in models if (f"pred_{m}" in merged_preds.columns or f"pred_{m}" == "pred_lgbm")]

    # 2. Segmented Scorecards: Category, Store, Demand Type
    results_dict = {}

    def _score_models_by_group(group_col: str | None = None) -> list[dict]:
        score_rows = []
        for m in available_models:
            pred_col = f"pred_{m}"
            if group_col is None:
                res = evaluate_forecasts(merged_preds, actual_col="sales", pred_col=pred_col)
                for _, r in res.iterrows():
                    score_rows.append({
                        "model": m,
                        "mae": r["mae"],
                        "wape": r["wape"],
                        "bias": r["bias"],
                        "total_actual": r["total_actual"],
                    })
            else:
                res = evaluate_forecasts(merged_preds, actual_col="sales", pred_col=pred_col, group_cols=[group_col])
                for _, r in res.iterrows():
                    score_rows.append({
                        group_col: r[group_col],
                        "model": m,
                        "mae": r["mae"],
                        "wape": r["wape"],
                        "bias": r["bias"],
                        "total_actual": r["total_actual"],
                    })
        return score_rows

    results_dict["global"] = _score_models_by_group(None)
    results_dict["by_category"] = _score_models_by_group("cat_id")
    results_dict["by_store"] = _score_models_by_group("store_id")
    results_dict["by_demand_type"] = _score_models_by_group("demand_type")

    # 3. Item-Level Head-to-Head: LightGBM vs Best Baseline (MA-28)
    print("\nExecuting item-level failure analysis (LightGBM vs Moving Average 28d)...")
    item_comparison = []

    for series_id, grp in merged_preds.groupby("id", observed=False):
        y_act = grp["sales"].to_numpy(dtype=np.float64)
        sum_act = np.sum(y_act)
        if sum_act == 0:
            continue

        lgbm_preds = grp["pred_lgbm"].to_numpy(dtype=np.float64)
        ma_preds = grp["pred_moving_average_28d"].to_numpy(dtype=np.float64)

        wape_lgbm = float(np.sum(np.abs(y_act - lgbm_preds)) / sum_act)
        wape_ma = float(np.sum(np.abs(y_act - ma_preds)) / sum_act)
        mae_lgbm = float(np.mean(np.abs(y_act - lgbm_preds)))
        mae_ma = float(np.mean(np.abs(y_act - ma_preds)))

        meta = grp.iloc[0]
        item_comparison.append({
            "id": series_id,
            "item_id": meta["item_id"],
            "dept_id": meta["dept_id"],
            "cat_id": meta["cat_id"],
            "store_id": meta["store_id"],
            "demand_type": meta["demand_type"],
            "adi": meta["adi"],
            "cv2": meta["cv2"],
            "zero_percentage": meta["zero_percentage"],
            "total_sales": float(sum_act),
            "wape_lgbm": round(wape_lgbm, 4),
            "wape_ma28": round(wape_ma, 4),
            "mae_lgbm": round(mae_lgbm, 4),
            "mae_ma28": round(mae_ma, 4),
            "lgbm_won": bool(wape_lgbm < wape_ma),
            "wape_diff": round(wape_lgbm - wape_ma, 4),  # positive means lgbm is worse
        })

    item_df = pd.DataFrame(item_comparison)
    item_df.to_parquet(out_path / "failure_analysis.parquet", index=False)

    # 4. Synthesize Failure Summary
    total_items = len(item_df)
    lgbm_wins = int(item_df["lgbm_won"].sum())
    ma_wins = total_items - lgbm_wins
    win_rate = (lgbm_wins / total_items) * 100

    # Win rate by demand type
    win_rate_by_type = {}
    for dt, grp in item_df.groupby("demand_type"):
        dt_wins = int(grp["lgbm_won"].sum())
        dt_total = len(grp)
        win_rate_by_type[dt] = {
            "lgbm_wins": dt_wins,
            "total_items": dt_total,
            "win_rate_pct": round((dt_wins / dt_total) * 100, 1),
            "avg_wape_lgbm": round(grp["wape_lgbm"].mean() * 100, 2),
            "avg_wape_ma28": round(grp["wape_ma28"].mean() * 100, 2),
        }

    failure_summary = {
        "total_skus_evaluated": total_items,
        "lgbm_wins": lgbm_wins,
        "ma28_wins": ma_wins,
        "overall_lgbm_win_rate_pct": round(win_rate, 2),
        "performance_by_demand_type": win_rate_by_type,
    }
    results_dict["failure_summary"] = failure_summary

    # Save summary JSON
    summary_file = out_path / "granular_evaluation.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # 5. Print Executive Failure Analysis Tables
    print("\n=== Performance by Demand Velocity Type (Syntetos-Boylan) ===")
    dt_rows = []
    for dt, stats in win_rate_by_type.items():
        dt_rows.append([
            dt,
            f"{stats['total_items']}",
            f"{stats['win_rate_pct']}% ({stats['lgbm_wins']}/{stats['total_items']})",
            f"{stats['avg_wape_lgbm']:.2f}%",
            f"{stats['avg_wape_ma28']:.2f}%",
            "LightGBM" if stats['avg_wape_lgbm'] < stats['avg_wape_ma28'] else "MA-28",
        ])
    print(tabulate(
        dt_rows,
        headers=["Demand Type", "SKU Count", "LGBM Win Rate", "LGBM WAPE", "MA-28 WAPE", "Segment Winner"],
        tablefmt="github",
    ))

    # Print Top 5 SKU Cases where LightGBM Lost the Most
    print("\n=== Top 5 Failure Modes: Where LightGBM Lost to Baseline ===")
    losses = item_df[~item_df["lgbm_won"]].sort_values("wape_diff", ascending=False).head(5)
    loss_rows = []
    for _, r in losses.iterrows():
        loss_rows.append([
            r["id"],
            r["cat_id"],
            r["demand_type"],
            f"{r['zero_percentage']:.1f}%",
            f"{r['wape_lgbm'] * 100:.1f}%",
            f"{r['wape_ma28'] * 100:.1f}%",
            f"+{r['wape_diff'] * 100:.1f}%",
        ])
    print(tabulate(
        loss_rows,
        headers=["Series ID", "Category", "Demand Type", "Zero %", "LGBM WAPE", "MA-28 WAPE", "WAPE Deficit"],
        tablefmt="github",
    ))

    print(f"\nSaved granular evaluation report to: {summary_file}")
    print(f"Saved SKU-level failure parquet to: {out_path / 'failure_analysis.parquet'}")
    return results_dict


if __name__ == "__main__":
    run_failure_analysis()
