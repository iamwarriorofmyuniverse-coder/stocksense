"""Model training and rolling-origin backtest execution script."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.data.loader import DataLoader
from src.features.pipeline import FeaturePipeline
from src.models.backtest import RollingOriginBacktest
from src.models.lgbm import LightGBMForecaster


def run_training_and_backtest(
    data_dir: str | Path = "data/synthetic",
    config_dir: str | Path = "configs",
    artifacts_dir: str | Path = "artifacts",
    results_dir: str | Path = "results",
    num_folds: int = 3,
    horizon_days: int = 28,
) -> dict:
    """Execute end-to-end feature pipeline, multi-fold backtest, and artifact generation."""
    art_path = Path(artifacts_dir)
    res_path = Path(results_dir)
    art_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading raw panel dataset...")
    loader = DataLoader(data_dir)
    cal, prices, sales = loader.load_raw()
    panel_df = loader.melt_and_merge(sales, cal, prices)

    # 2. Extract Features
    print("Executing feature pipeline with strict horizon leakage guards...")
    feat_cfg_path = Path(config_dir) / "features.yaml"
    pipeline = FeaturePipeline(feat_cfg_path)
    processed_df, feature_cols = pipeline.fit_transform(panel_df, drop_warmup=True)
    print(f"Processed dataset ready: {len(processed_df)} rows, {len(feature_cols)} features.")

    # 3. Load Model Config
    model_cfg_path = Path(config_dir) / "model.yaml"
    model_params = {}
    if model_cfg_path.exists():
        with open(model_cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            model_params = cfg.get("lightgbm", {})

    # 4. Execute Rolling-Origin Backtest
    backtester = RollingOriginBacktest(
        horizon_days=horizon_days,
        num_folds=num_folds,
        step_days=horizon_days,
    )
    predictions_df, summary = backtester.run(
        df=processed_df,
        feature_cols=feature_cols,
        categorical_cols=pipeline.categorical_cols,
        model_params=model_params,
    )

    # 5. Train Final Global Model on Complete Processed Dataset
    print("\nTraining final global production model on full historical dataset...")
    final_model = LightGBMForecaster(**model_params)
    final_model.fit(
        X_train=processed_df[feature_cols],
        y_train=processed_df["sales"],
        categorical_features=pipeline.categorical_cols,
    )

    # 6. Save Artifacts
    model_save_path = art_path / "lgbm_model.joblib"
    final_model.save(model_save_path)
    print(f"Saved production model to: {model_save_path}")

    # Feature importances
    feat_imp = final_model.get_feature_importance()
    feat_imp_path = art_path / "feature_importance.json"
    feat_imp.to_json(feat_imp_path, orient="records", indent=2)
    print(f"Top 5 Most Important Features:\n{feat_imp.head(5).to_string(index=False)}")

    # Predictions and summaries
    preds_save_path = art_path / "backtest_predictions.parquet"
    predictions_df.to_parquet(preds_save_path, index=False)
    print(f"Saved backtest predictions to: {preds_save_path}")

    summary_save_path = art_path / "backtest_summary.json"
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 7. Append to Immutable Experiment Log
    log_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "num_folds": num_folds,
        "horizon_days": horizon_days,
        "num_series": len(processed_df["id"].unique()),
        "features_count": len(feature_cols),
        "metrics": summary["aggregated"],
    }
    log_file = res_path / "experiment_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Logged experiment run to: {log_file}")

    return summary


if __name__ == "__main__":
    run_training_and_backtest()
