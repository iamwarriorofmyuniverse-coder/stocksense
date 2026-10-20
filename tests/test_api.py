"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client with lifespan context to ensure artifacts load."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health returns 200 and valid status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_available" in data


def test_skus_endpoint(client):
    """Verify /skus returns list of available retail items."""
    res = client.get("/skus")
    assert res.status_code == 200
    data = res.json()
    assert "total_count" in data
    assert data["total_count"] > 0
    assert len(data["skus"]) > 0


def test_forecast_endpoint(client):
    """Verify /forecast returns predictions and error uncertainty bands."""
    # First get a valid store and item
    skus_res = client.get("/skus")
    sample_sku = skus_res.json()["skus"][0]

    payload = {
        "store_id": sample_sku["store_id"],
        "item_id": sample_sku["item_id"],
        "model_name": "lgbm",
    }
    res = client.post("/forecast", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["store_id"] == sample_sku["store_id"]
    assert data["item_id"] == sample_sku["item_id"]
    assert len(data["forecast_points"]) == 28

    # Check that upper bound is >= lower bound
    for pt in data["forecast_points"]:
        assert pt["upper_bound"] >= pt["lower_bound"]
        assert pt["lower_bound"] >= 0.0


def test_reorder_endpoint(client):
    """Verify /reorder computes safety stock and replenishment recommendations."""
    skus_res = client.get("/skus")
    sample_sku = skus_res.json()["skus"][0]

    # Test case 1: Very low on-hand stock (should trigger reorder)
    low_stock_req = {
        "store_id": sample_sku["store_id"],
        "item_id": sample_sku["item_id"],
        "current_on_hand": 1.0,
        "in_transit_qty": 0.0,
        "lead_time_days": 3,
        "service_level": 0.95,
        "review_period_days": 1,
    }
    res_low = client.post("/reorder", json=low_stock_req)
    assert res_low.status_code == 200
    data_low = res_low.json()
    assert data_low["reorder_recommended"] is True
    assert data_low["recommended_order_qty"] > 0
    assert data_low["safety_stock"] > 0
    assert data_low["reorder_point"] > 0

    # Test case 2: Massive on-hand stock (should NOT trigger reorder)
    high_stock_req = {
        "store_id": sample_sku["store_id"],
        "item_id": sample_sku["item_id"],
        "current_on_hand": 5000.0,
        "in_transit_qty": 0.0,
        "lead_time_days": 3,
        "service_level": 0.95,
        "review_period_days": 1,
    }
    res_high = client.post("/reorder", json=high_stock_req)
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert data_high["reorder_recommended"] is False
    assert data_high["recommended_order_qty"] == 0.0


def test_metrics_endpoint(client):
    """Verify /metrics returns segmented model vs baseline benchmark."""
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "global" in data
    assert "by_category" in data
    assert "by_demand_type" in data
    assert "failure_summary" in data
