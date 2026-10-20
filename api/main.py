"""FastAPI backend service for StockSense demand forecasting and replenishment engine."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalystRequest,
    AnalystResponse,
    ForecastDataPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReorderRequest,
    ReorderResponse,
    SKUItem,
    SKUListResponse,
    ToolCallRecord,
)
from src.assistant.agent import AnalystAssistant
from src.simulation.policy import calculate_reorder_parameters

# Global state for loaded artifacts
ARTIFACTS_DIR = Path("artifacts")
STATE = {
    "predictions_df": None,
    "failure_df": None,
    "metrics_json": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load precomputed model artifacts and evaluation summaries on server startup."""
    preds_file = ARTIFACTS_DIR / "backtest_predictions.parquet"
    failure_file = ARTIFACTS_DIR / "failure_analysis.parquet"
    metrics_file = ARTIFACTS_DIR / "granular_evaluation.json"

    if preds_file.exists():
        STATE["predictions_df"] = pd.read_parquet(preds_file)
        print(f"Loaded backtest predictions: {len(STATE['predictions_df'])} rows.")
    if failure_file.exists():
        STATE["failure_df"] = pd.read_parquet(failure_file)
        print(f"Loaded failure analysis metadata: {len(STATE['failure_df'])} SKUs.")
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            STATE["metrics_json"] = json.load(f)
        print("Loaded granular evaluation metrics summary.")

    yield
    STATE.clear()


app = FastAPI(
    title="StockSense Replenishment API",
    description="Operational demand forecasting and dynamic inventory replenishment service.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for web apps and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health status and metadata endpoint."""
    num_skus = len(STATE["failure_df"]) if STATE["failure_df"] is not None else 0
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        num_skus_available=num_skus,
        models_available=["lgbm", "moving_average_28d", "moving_average_7d", "naive", "seasonal_naive_7d"],
    )


@app.get("/skus", response_model=SKUListResponse)
def list_skus(
    store_id: str | None = Query(None, description="Optional store filter e.g. CA_1"),
    cat_id: str | None = Query(None, description="Optional category filter e.g. FOODS"),
):
    """List available SKUs and their demand velocity classifications."""
    if STATE["failure_df"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded.")

    df = STATE["failure_df"].copy()
    if store_id:
        df = df[df["store_id"] == store_id]
    if cat_id:
        df = df[df["cat_id"] == cat_id]

    items = [
        SKUItem(
            id=row["id"],
            item_id=row["item_id"],
            store_id=row["store_id"],
            cat_id=row["cat_id"],
            demand_type=row.get("demand_type", "Unknown"),
        )
        for _, row in df.iterrows()
    ]
    return SKUListResponse(total_count=len(items), skus=items)


@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(req: ForecastRequest):
    """Retrieve 28-day forward forecast and error uncertainty band for a store/item SKU."""
    if STATE["predictions_df"] is None:
        raise HTTPException(status_code=503, detail="Prediction artifacts not loaded.")

    df = STATE["predictions_df"]
    # Filter by store and item in the final held-out fold
    max_fold = df["fold"].max()
    sku_data = df[(df["store_id"] == req.store_id) & (df["item_id"] == req.item_id) & (df["fold"] == max_fold)].sort_values("date")

    if sku_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{req.item_id}' at store '{req.store_id}' not found.",
        )

    pred_col = f"pred_{req.model_name}"
    if pred_col not in sku_data.columns:
        pred_col = "pred_lgbm"

    # Compute empirical standard error for uncertainty bands
    residuals = sku_data["sales"].to_numpy() - sku_data[pred_col].to_numpy()
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

    # Demand type lookup
    demand_type = "Smooth"
    if STATE["failure_df"] is not None:
        f_row = STATE["failure_df"][(STATE["failure_df"]["store_id"] == req.store_id) & (STATE["failure_df"]["item_id"] == req.item_id)]
        if not f_row.empty:
            demand_type = f_row.iloc[0].get("demand_type", "Smooth")

    points = []
    for _, row in sku_data.iterrows():
        f_val = float(row[pred_col])
        points.append(ForecastDataPoint(
            date=row["date"].strftime("%Y-%m-%d"),
            actual=float(row["sales"]),
            forecast=round(f_val, 2),
            lower_bound=round(max(0.0, f_val - 1.96 * sigma), 2),
            upper_bound=round(f_val + 1.96 * sigma, 2),
        ))

    series_id = str(sku_data["id"].iloc[0])
    return ForecastResponse(
        id=series_id,
        store_id=req.store_id,
        item_id=req.item_id,
        model_name=req.model_name,
        demand_type=demand_type,
        forecast_points=points,
    )


@app.post("/reorder", response_model=ReorderResponse)
def calculate_replenishment_order(req: ReorderRequest):
    """Compute dynamic safety stock, reorder point (ROP), and recommended purchase order quantity."""
    if STATE["predictions_df"] is None:
        raise HTTPException(status_code=503, detail="Prediction artifacts not loaded.")

    df = STATE["predictions_df"]
    max_fold = df["fold"].max()
    sku_data = df[(df["store_id"] == req.store_id) & (df["item_id"] == req.item_id) & (df["fold"] == max_fold)].sort_values("date")

    if sku_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{req.item_id}' at store '{req.store_id}' not found.",
        )

    # Use LightGBM predictions for replenishment decision
    f_arr = sku_data["pred_lgbm"].to_numpy()
    residuals = sku_data["sales"].to_numpy() - f_arr
    sigma_e = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

    # Calculate policy parameters
    ss, rop, order_up_to = calculate_reorder_parameters(
        forecast_horizon=f_arr,
        lead_time_days=req.lead_time_days,
        review_period_days=req.review_period_days,
        service_level=req.service_level,
        sigma_daily_error=sigma_e,
    )

    inventory_position = req.current_on_hand + req.in_transit_qty
    ltd = float(np.sum(f_arr[:req.lead_time_days]))

    reorder_recommended = inventory_position <= rop
    recommended_qty = float(np.ceil(max(0.0, order_up_to - inventory_position))) if reorder_recommended else 0.0

    return ReorderResponse(
        id=str(sku_data["id"].iloc[0]),
        store_id=req.store_id,
        item_id=req.item_id,
        lead_time_days=req.lead_time_days,
        service_level=req.service_level,
        current_on_hand=req.current_on_hand,
        in_transit_qty=req.in_transit_qty,
        inventory_position=round(inventory_position, 2),
        lead_time_demand=round(ltd, 2),
        safety_stock=round(ss, 2),
        reorder_point=round(rop, 2),
        target_stock_level=round(order_up_to, 2),
        reorder_recommended=reorder_recommended,
        recommended_order_qty=recommended_qty,
    )


@app.get("/metrics")
def get_metrics_summary():
    """Retrieve global and segment-wise model vs baseline benchmark metrics."""
    if STATE["metrics_json"] is None:
        raise HTTPException(status_code=503, detail="Metrics summary artifact not found.")
    return STATE["metrics_json"]


@app.post("/analyst", response_model=AnalystResponse)
def query_analyst_assistant(req: AnalystRequest):
    """Natural-language retail analyst querying precomputed metrics via tool calling."""
    assistant = AnalystAssistant(api_key=req.api_key)
    res = assistant.ask(req.question)

    tool_records = [
        ToolCallRecord(
            tool=tc.get("tool", "unknown"),
            args=tc.get("args", {}),
            result=tc.get("result", {}),
        )
        for tc in res.get("tools_called", [])
    ]

    return AnalystResponse(
        question=req.question,
        answer=res.get("answer", ""),
        tools_called=tool_records,
        mode=res.get("mode", "deterministic"),
    )

