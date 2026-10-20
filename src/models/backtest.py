"""Rolling-origin temporal cross-validation (backtesting) engine."""

import pandas as pd
from tabulate import tabulate

from src.evaluation.metrics import evaluate_forecasts
from src.models.baselines import generate_all_baselines
from src.models.lgbm import LightGBMForecaster


class RollingOriginBacktest:
    """Executes multi-fold rolling-origin backtesting with zero temporal leakage.

    Why Rolling-Origin Backtesting:
    A single train/test split can be biased by calendar anomalies (e.g., an early Easter,
    unusual promotional week). Multi-fold walk-forward validation tests model stability
    across distinct temporal regimes.
    """

    def __init__(
        self,
        horizon_days: int = 28,
        num_folds: int = 3,
        step_days: int = 28,
    ):
        self.horizon_days = horizon_days
        self.num_folds = num_folds
        self.step_days = step_days

    def generate_splits(self, df: pd.DataFrame) -> list[dict]:
        """Generate chronologically expanding train/test splits."""
        unique_dates = sorted(df["date"].unique())
        total_days = len(unique_dates)
        required_days = self.num_folds * self.step_days + 30  # at least 30 days for minimal training

        if total_days < required_days:
            raise ValueError(
                f"Dataset has {total_days} days; backtest requires at least {required_days} "
                f"for {self.num_folds} folds of {self.horizon_days}-day horizons."
            )

        splits = []
        # Generate folds backwards from the most recent date
        for fold_idx in range(self.num_folds):
            # Test window end offset
            test_end_offset = fold_idx * self.step_days
            test_end_idx = total_days - 1 - test_end_offset
            test_start_idx = test_end_idx - self.horizon_days + 1
            train_end_idx = test_start_idx - 1

            train_end_date = unique_dates[train_end_idx]
            test_start_date = unique_dates[test_start_idx]
            test_end_date = unique_dates[test_end_idx]

            train_mask = df["date"] <= train_end_date
            test_mask = (df["date"] >= test_start_date) & (df["date"] <= test_end_date)

            splits.append({
                "fold": self.num_folds - fold_idx,  # Fold 1 earliest, Fold N latest
                "train_end_date": train_end_date,
                "test_start_date": test_start_date,
                "test_end_date": test_end_date,
                "train_mask": train_mask,
                "test_mask": test_mask,
            })

        # Order chronologically: Fold 1 -> Fold N
        splits.sort(key=lambda s: s["fold"])
        return splits

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        categorical_cols: list[str] | None = None,
        model_params: dict | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Run backtest across all folds for both LightGBM and Baselines."""
        splits = self.generate_splits(df)
        all_fold_predictions = []
        fold_summaries = []

        print(f"\n=== Executing {self.num_folds}-Fold Rolling-Origin Backtest (Horizon: {self.horizon_days}d) ===")

        for split in splits:
            fold = split["fold"]
            t_train_end = split["train_end_date"].strftime("%Y-%m-%d")
            t_test_start = split["test_start_date"].strftime("%Y-%m-%d")
            t_test_end = split["test_end_date"].strftime("%Y-%m-%d")
            print(f"\n--- Fold {fold}: Train up to {t_train_end} | Test: [{t_test_start} to {t_test_end}] ---")

            train_data = df[split["train_mask"]].copy()
            test_data = df[split["test_mask"]].copy()

            # 1. Evaluate Baselines
            baseline_preds = generate_all_baselines(train_data, test_data)

            # 2. Train and Predict Global LightGBM
            X_train = train_data[feature_cols]
            y_train = train_data["sales"]
            X_test = test_data[feature_cols]

            forecaster = LightGBMForecaster(**(model_params or {}))
            forecaster.fit(
                X_train=X_train,
                y_train=y_train,
                categorical_features=categorical_cols,
            )
            lgbm_preds = forecaster.predict(X_test)

            # 3. Store fold results
            fold_df = test_data[[
                "id", "item_id", "dept_id", "cat_id", "store_id", "date", "sales", "sell_price"
            ]].copy()
            fold_df["fold"] = fold
            fold_df["pred_lgbm"] = lgbm_preds
            for b_name, b_pred in baseline_preds.items():
                fold_df[f"pred_{b_name}"] = b_pred

            all_fold_predictions.append(fold_df)

            # 4. Compute Fold Performance Metrics
            models_to_score = ["lgbm"] + list(baseline_preds.keys())
            for m in models_to_score:
                col = f"pred_{m}" if m != "lgbm" else "pred_lgbm"
                res = evaluate_forecasts(fold_df, actual_col="sales", pred_col=col)
                fold_summaries.append({
                    "fold": fold,
                    "model": m,
                    "mae": float(res["mae"].iloc[0]),
                    "wape": float(res["wape"].iloc[0]),
                    "bias": float(res["bias"].iloc[0]),
                })

        # Concatenate all fold predictions
        combined_preds = pd.concat(all_fold_predictions, ignore_index=True)
        summary_df = pd.DataFrame(fold_summaries)

        # Aggregate across folds (mean and std)
        agg_rows = []
        for model_name, grp in summary_df.groupby("model"):
            agg_rows.append({
                "model": model_name,
                "wape_mean": float(grp["wape"].mean()),
                "wape_std": float(grp["wape"].std()),
                "mae_mean": float(grp["mae"].mean()),
                "mae_std": float(grp["mae"].std()),
                "bias_mean": float(grp["bias"].mean()),
            })
        agg_df = pd.DataFrame(agg_rows).sort_values("wape_mean").reset_index(drop=True)

        # Print Benchmark Comparison Table
        table_rows = []
        for _, row in agg_df.iterrows():
            table_rows.append([
                row["model"],
                f"{row['mae_mean']:.4f} ± {row['mae_std']:.3f}",
                f"{row['wape_mean'] * 100:.2f}% ± {row['wape_std'] * 100:.2f}%",
                f"{row['bias_mean'] * 100:+.2f}%",
            ])
        headers = ["Model", "MAE Mean ± Std", "WAPE Mean ± Std", "Bias Mean"]
        print("\n=== Multi-Fold Backtest Aggregated Results ===")
        print(tabulate(table_rows, headers=headers, tablefmt="github"))

        return combined_preds, {
            "fold_summaries": fold_summaries,
            "aggregated": agg_df.to_dict(orient="records"),
        }
