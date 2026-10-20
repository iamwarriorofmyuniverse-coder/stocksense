"""Unit tests for Analyst Assistant tools and agent querying."""

from fastapi.testclient import TestClient

from api.main import app
from src.assistant.agent import AnalystAssistant
from src.assistant.tools import (
    get_category_metrics,
    get_demand_velocity_summary,
    get_failure_cases,
    get_simulation_impact,
    get_store_metrics,
)


def test_tool_get_store_metrics():
    """Verify get_store_metrics queries valid store performance records."""
    res = get_store_metrics("CA_1")
    assert "store_id" in res
    assert res["store_id"] == "CA_1"
    models = {m["model"]: m for m in res["models"]}
    assert "lgbm" in models
    assert "moving_average_28d" in models
    assert 0.0 < models["lgbm"]["wape"] < 2.0


def test_tool_get_category_metrics():
    """Verify get_category_metrics queries category performance."""
    res = get_category_metrics("FOODS")
    assert res["cat_id"] == "FOODS"
    assert len(res["models"]) > 0


def test_tool_get_failure_cases():
    """Verify get_failure_cases returns documented failure cases."""
    res = get_failure_cases(limit=3)
    assert "top_failure_cases" in res
    assert len(res["top_failure_cases"]) > 0
    first_case = res["top_failure_cases"][0]
    assert first_case["wape_deficit_pct"] > 0


def test_tool_get_simulation_impact():
    """Verify get_simulation_impact returns valid replenishment KPIs."""
    res = get_simulation_impact()
    assert "baseline_policy" in res
    assert "model_policy" in res
    assert "operational_delta" in res
    assert res["operational_delta"]["cost_reduction_pct"] > 0


def test_tool_get_demand_velocity_summary():
    """Verify get_demand_velocity_summary returns Syntetos-Boylan statistics."""
    res = get_demand_velocity_summary()
    assert "overall_lgbm_win_rate_pct" in res
    assert res["overall_lgbm_win_rate_pct"] > 50.0


def test_analyst_assistant_deterministic_execution():
    """Verify AnalystAssistant calls tools and produces grounded answers."""
    assistant = AnalystAssistant()

    # Store query
    res_store = assistant.ask("Why did forecast error spike for store CA_1?")
    assert len(res_store["tools_called"]) > 0
    assert res_store["tools_called"][0]["tool"] == "get_store_metrics"
    assert "CA_1" in res_store["answer"]

    # Failure query
    res_fail = assistant.ask("Where does the model lose to the baseline?")
    assert len(res_fail["tools_called"]) > 0
    assert res_fail["tools_called"][0]["tool"] == "get_failure_cases"
    assert "Failure Analysis" in res_fail["answer"]


def test_fastapi_analyst_endpoint():
    """Verify /analyst endpoint processes natural language queries."""
    with TestClient(app) as client:
        payload = {"question": "What was the bottom-line financial impact of our replenishment policy?"}
        res = client.post("/analyst", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert len(data["tools_called"]) > 0
        assert data["tools_called"][0]["tool"] == "get_simulation_impact"
