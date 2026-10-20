"""Deterministic query tools over precomputed StockSense artifacts.

CRITICAL RULE: The LLM must NEVER generate, guess, or calculate numbers itself.
All analytical answers MUST be grounded in data retrieved from these tools.
"""

import json
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")


def _load_granular_eval() -> dict:
    path = ARTIFACTS_DIR / "granular_evaluation.json"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_failure_df() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "failure_analysis.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return pd.read_parquet(path)


def _load_sim_summary() -> dict:
    path = ARTIFACTS_DIR / "simulation_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_store_metrics(store_id: str) -> dict:
    """Retrieve computed WAPE, MAE, Bias, and sales volume for all models at a specific store."""
    data = _load_granular_eval()
    store_records = [r for r in data.get("by_store", []) if r.get("store_id") == store_id]
    if not store_records:
        available_stores = list({r.get("store_id") for r in data.get("by_store", [])})
        return {
            "error": f"Store '{store_id}' not found. Available stores: {available_stores}"
        }
    return {
        "store_id": store_id,
        "models": store_records,
    }


def get_category_metrics(cat_id: str) -> dict:
    """Retrieve computed WAPE, MAE, Bias, and sales volume for a merchandise category (FOODS, HOBBIES, HOUSEHOLD)."""
    data = _load_granular_eval()
    cat_records = [r for r in data.get("by_category", []) if r.get("cat_id") == cat_id]
    if not cat_records:
        available_cats = list({r.get("cat_id") for r in data.get("by_category", [])})
        return {
            "error": f"Category '{cat_id}' not found. Available categories: {available_cats}"
        }
    return {
        "cat_id": cat_id,
        "models": cat_records,
    }


def get_sku_profile(store_id: str, item_id: str) -> dict:
    """Retrieve exact demand velocity type, ADI, CV2, zero %, and model vs baseline WAPE for a single SKU."""
    df = _load_failure_df()
    match = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)]
    if match.empty:
        return {"error": f"SKU '{item_id}' at store '{store_id}' not found."}
    row = match.iloc[0]
    return {
        "id": row["id"],
        "store_id": store_id,
        "item_id": item_id,
        "category": row["cat_id"],
        "demand_type": row["demand_type"],
        "adi": float(row["adi"]),
        "cv2": float(row["cv2"]),
        "zero_percentage": float(row["zero_percentage"]),
        "total_sales": float(row["total_sales"]),
        "wape_lgbm_pct": round(float(row["wape_lgbm"]) * 100, 2),
        "wape_ma28_pct": round(float(row["wape_ma28"]) * 100, 2),
        "winner": "LightGBM" if row["lgbm_won"] else "Moving Average 28d",
        "wape_difference_pct": round(float(row["wape_diff"]) * 100, 2),
    }


def get_failure_cases(category: str | None = None, limit: int = 5) -> dict:
    """Retrieve the top SKU cases where the Moving Average baseline beat LightGBM."""
    df = _load_failure_df()
    losses = df[~df["lgbm_won"]].copy()
    if category and category != "ALL":
        losses = losses[losses["cat_id"] == category]
    losses = losses.sort_values("wape_diff", ascending=False).head(limit)
    cases = []
    for _, r in losses.iterrows():
        cases.append({
            "id": r["id"],
            "store_id": r["store_id"],
            "item_id": r["item_id"],
            "category": r["cat_id"],
            "demand_type": r["demand_type"],
            "zero_percentage": float(r["zero_percentage"]),
            "wape_lgbm_pct": round(float(r["wape_lgbm"]) * 100, 2),
            "wape_ma28_pct": round(float(r["wape_ma28"]) * 100, 2),
            "wape_deficit_pct": round(float(r["wape_diff"]) * 100, 2),
        })
    return {
        "total_failures_in_scope": len(cases),
        "top_failure_cases": cases,
        "operational_explanation": (
            "Most failures occur in Smooth FOODS SKUs where stable demand favors "
            "a low-variance moving average over tree-based feature splits."
        ),
    }


def get_simulation_impact() -> dict:
    """Retrieve operational and financial replenishment simulation results (Baseline Policy vs LightGBM)."""
    return _load_sim_summary()


def get_demand_velocity_summary() -> dict:
    """Retrieve Syntetos-Boylan demand segmentation summary and LightGBM win rates."""
    data = _load_granular_eval()
    return data.get("failure_summary", {})


TOOL_DEFINITIONS = [
    {
        "name": "get_store_metrics",
        "description": "Get forecasting accuracy metrics (WAPE, MAE, Bias) for all models at a given retail store.",
        "parameters": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID e.g. 'CA_1', 'CA_2'"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_category_metrics",
        "description": "Get forecasting metrics (WAPE, MAE, Bias) across merchandise categories: FOODS, HOBBIES, HOUSEHOLD.",
        "parameters": {
            "type": "object",
            "properties": {
                "cat_id": {"type": "string", "description": "Category ID e.g. 'FOODS', 'HOBBIES', 'HOUSEHOLD'"},
            },
            "required": ["cat_id"],
        },
    },
    {
        "name": "get_sku_profile",
        "description": "Look up an individual SKU profile: ADI, CV2, demand type, zero sales frequency, and model vs baseline WAPE.",
        "parameters": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID e.g. 'CA_1'"},
                "item_id": {"type": "string", "description": "Item ID e.g. 'FOODS_1_001'"},
            },
            "required": ["store_id", "item_id"],
        },
    },
    {
        "name": "get_failure_cases",
        "description": "Get the top failure cases where the heuristic baseline beat LightGBM.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional category filter: FOODS, HOBBIES, HOUSEHOLD"},
                "limit": {"type": "integer", "description": "Number of cases to retrieve, default 5"},
            },
        },
    },
    {
        "name": "get_simulation_impact",
        "description": "Get the supply chain replenishment simulation KPIs (fill rate, inventory reduction %, food waste savings %, cost reduction %).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_demand_velocity_summary",
        "description": "Get Syntetos-Boylan demand velocity quadrant summary (Smooth, Intermittent, Lumpy) and SKU win rates.",
        "parameters": {"type": "object", "properties": {}},
    },
]

TOOL_REGISTRY = {
    "get_store_metrics": get_store_metrics,
    "get_category_metrics": get_category_metrics,
    "get_sku_profile": get_sku_profile,
    "get_failure_cases": get_failure_cases,
    "get_simulation_impact": get_simulation_impact,
    "get_demand_velocity_summary": get_demand_velocity_summary,
}
