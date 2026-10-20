"""Pydantic schemas for FastAPI endpoints."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    num_skus_available: int
    models_available: list[str]


class SKUItem(BaseModel):
    id: str
    item_id: str
    store_id: str
    cat_id: str
    demand_type: str


class SKUListResponse(BaseModel):
    total_count: int
    skus: list[SKUItem]


class ForecastRequest(BaseModel):
    store_id: str = Field(..., json_schema_extra={"example": "CA_1"})
    item_id: str = Field(..., json_schema_extra={"example": "FOODS_1_001"})
    model_name: str = Field("lgbm", description="lgbm or moving_average_28d", json_schema_extra={"example": "lgbm"})


class ForecastDataPoint(BaseModel):
    date: str
    actual: float | None = None
    forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    id: str
    store_id: str
    item_id: str
    model_name: str
    demand_type: str
    forecast_points: list[ForecastDataPoint]


class ReorderRequest(BaseModel):
    store_id: str = Field(..., json_schema_extra={"example": "CA_1"})
    item_id: str = Field(..., json_schema_extra={"example": "FOODS_1_001"})
    current_on_hand: float = Field(..., ge=0, json_schema_extra={"example": 8.0})
    in_transit_qty: float = Field(0.0, ge=0, json_schema_extra={"example": 0.0})
    lead_time_days: int = Field(3, ge=1, le=30, json_schema_extra={"example": 3})
    service_level: float = Field(0.95, gt=0.5, lt=1.0, json_schema_extra={"example": 0.95})
    review_period_days: int = Field(1, ge=1, le=14, json_schema_extra={"example": 1})


class ReorderResponse(BaseModel):
    id: str
    store_id: str
    item_id: str
    lead_time_days: int
    service_level: float
    current_on_hand: float
    in_transit_qty: float
    inventory_position: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    target_stock_level: float
    reorder_recommended: bool
    recommended_order_qty: float


class AnalystRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "Why did forecast error spike for store CA_1?"})
    api_key: str | None = Field(None, json_schema_extra={"example": None})


class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    result: dict


class AnalystResponse(BaseModel):
    question: str
    answer: str
    tools_called: list[ToolCallRecord]
    mode: str

