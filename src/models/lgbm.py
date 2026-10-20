"""Global LightGBM forecaster for multi-store retail panel data."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


class LightGBMForecaster:
    """Global LightGBM model utilizing Tweedie loss for zero-inflated demand.

    Why Global LightGBM + Tweedie:
    1. Cross-Series Learning: A single model learns shared price elasticity and seasonal
       patterns across thousands of SKUs, outperforming isolated per-SKU models.
    2. Tweedie Distribution (p=1.2): Compound Poisson-Gamma distribution ideal for retail
       sales where ~50-70% of days are 0 (no demand) with skewed positive quantities.
    """

    def __init__(self, config_path: str | Path | None = None, **kwargs):
        self.params = self._default_params()
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "lightgbm" in cfg:
                    self.params.update(cfg["lightgbm"])
                if "early_stopping_rounds" in cfg:
                    self.early_stopping_rounds = cfg["early_stopping_rounds"]
        else:
            self.early_stopping_rounds = 30

        # Override with kwargs if provided
        self.params.update(kwargs)
        self.model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []
        self.categorical_features: list[str] = []

    @staticmethod
    def _default_params() -> dict:
        return {
            "objective": "tweedie",
            "tweedie_variance_power": 1.2,
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 45,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        categorical_features: list[str] | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "LightGBMForecaster":
        """Train LightGBM model with optional early stopping on validation split."""
        self.feature_names = list(X_train.columns)

        # Identify categorical features
        if categorical_features is None:
            self.categorical_features = [
                c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)
            ]
        else:
            self.categorical_features = [c for c in categorical_features if c in X_train.columns]

        # Convert categoricals to 'category' dtype if not already
        X_tr = X_train.copy()
        for cat in self.categorical_features:
            X_tr[cat] = X_tr[cat].astype("category")

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.copy()
            for cat in self.categorical_features:
                X_v[cat] = X_v[cat].astype("category")
            eval_set = [(X_v, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        # Filter out keys not accepted by LGBMRegressor constructor
        lgbm_kwargs = {k: v for k, v in self.params.items() if k not in ("early_stopping_rounds",)}
        self.model = lgb.LGBMRegressor(**lgbm_kwargs)

        self.model.fit(
            X_tr,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sales for feature matrix X, clipped at zero (no negative retail demand)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_eval = X[self.feature_names].copy()
        for cat in self.categorical_features:
            if cat in X_eval.columns:
                X_eval[cat] = X_eval[cat].astype("category")

        preds = self.model.predict(X_eval)
        # Demand cannot be negative in retail operations
        return np.clip(preds, 0.0, None).astype(np.float32)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Extract sorted feature importance dataframe."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: str | Path) -> None:
        """Persist model and metadata to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
        }, path)

    @classmethod
    def load(cls, filepath: str | Path) -> "LightGBMForecaster":
        """Load persisted model artifact from disk."""
        data = joblib.load(filepath)
        forecaster = cls()
        forecaster.model = data["model"]
        forecaster.params = data["params"]
        forecaster.feature_names = data["feature_names"]
        forecaster.categorical_features = data["categorical_features"]
        return forecaster
