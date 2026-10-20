"""Page 3: Replenishment Simulator (The Centerpiece) - Live Interactive Policy Sandbox."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import load_backtest_predictions
from src.simulation.simulator import InventorySimulator

st.set_page_config(page_title="Replenishment Simulator | StockSense", page_icon="⚙️", layout="wide")

st.title("⚙️ Replenishment Simulator: Dynamic Supply Chain Sandbox")
st.markdown("""
**The Centerpiece of StockSense**: Adjust operational levers (Lead Time & Target Service Level)
and observe how machine learning forecasts alter inventory holding, stock-outs, and food waste in real time.
""")

df_preds = load_backtest_predictions()
max_fold = df_preds["fold"].max()
test_window = df_preds[df_preds["fold"] == max_fold].copy()

# Sidebar Interactive Controls
st.sidebar.header("🎛️ Operational Levers")

lead_time = st.sidebar.slider(
    "Supplier Lead Time (Days)",
    min_value=1,
    max_value=14,
    value=3,
    step=1,
    help="Days between purchase order placement and delivery arrival at store.",
)

service_level = st.sidebar.slider(
    "Target Cycle Service Level (SLA %)",
    min_value=80,
    max_value=99,
    value=95,
    step=1,
    help="Target probability of not stocking out during lead time.",
) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Inventory Scope")
stores = ["ALL"] + sorted(test_window["store_id"].unique())
sel_store = st.sidebar.selectbox("Store Filter", stores, index=0)

categories = ["ALL"] + sorted(test_window["cat_id"].unique())
sel_cat = st.sidebar.selectbox("Category Filter", categories, index=0)

# Apply filters
sim_input = test_window.copy()
if sel_store != "ALL":
    sim_input = sim_input[sim_input["store_id"] == sel_store]
if sel_cat != "ALL":
    sim_input = sim_input[sim_input["cat_id"] == sel_cat]

if sim_input.empty:
    st.warning("No SKUs match the selected filters.")
    st.stop()

# Live simulation execution
with st.spinner("Simulating daily discrete inventory paths..."):
    sim_engine = InventorySimulator(
        lead_time_days=lead_time,
        service_level=service_level,
        review_period_days=1,
    )

    baseline_sim = sim_engine.simulate_panel(sim_input, forecast_col="pred_moving_average_28d")
    lgbm_sim = sim_engine.simulate_panel(sim_input, forecast_col="pred_lgbm")

# Compute KPIs
tot_demand = float(baseline_sim["demand"].sum())

def _get_metrics(df):
    tot_fulfilled = float(df["fulfilled_sales"].sum())
    tot_lost = float(df["lost_sales"].sum())
    tot_spoil = float(df["spoilage_units"].sum())
    avg_stock = float(df["on_hand"].mean())
    tot_holding = float(df["holding_cost"].sum())
    tot_stockout = float(df["stockout_cost"].sum())
    tot_spoil_cost = float(df["spoilage_cost"].sum())
    tot_cost = float(df["total_cost"].sum())
    fill_rate = (tot_fulfilled / (tot_demand + 1e-5)) * 100.0
    stockout_rate = (tot_lost / (tot_demand + 1e-5)) * 100.0
    return {
        "fulfilled": tot_fulfilled,
        "lost": tot_lost,
        "spoilage": tot_spoil,
        "avg_stock": avg_stock,
        "holding_cost": tot_holding,
        "stockout_cost": tot_stockout,
        "spoilage_cost": tot_spoil_cost,
        "total_cost": tot_cost,
        "fill_rate": fill_rate,
        "stockout_rate": stockout_rate,
    }

m_base = _get_metrics(baseline_sim)
m_lgbm = _get_metrics(lgbm_sim)

# Top KPI comparison row
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

inv_delta_pct = ((m_base["avg_stock"] - m_lgbm["avg_stock"]) / (m_base["avg_stock"] + 1e-5)) * 100.0
cost_delta_pct = ((m_base["total_cost"] - m_lgbm["total_cost"]) / (m_base["total_cost"] + 1e-5)) * 100.0
waste_delta_pct = ((m_base["spoilage"] - m_lgbm["spoilage"]) / (m_base["spoilage"] + 1e-5)) * 100.0

with kpi_col1:
    st.metric(
        label="Average On-Hand Inventory",
        value=f"{m_lgbm['avg_stock']:.1f} units",
        delta=f"-{inv_delta_pct:.1f}% vs Baseline" if inv_delta_pct > 0 else f"+{-inv_delta_pct:.1f}% vs Baseline",
        delta_color="normal",
    )
with kpi_col2:
    st.metric(
        label="Realized Customer Fill Rate",
        value=f"{m_lgbm['fill_rate']:.1f}%",
        delta=f"{m_lgbm['fill_rate'] - m_base['fill_rate']:+.2f}% vs Baseline",
    )
with kpi_col3:
    st.metric(
        label="Food Waste / Spoilage",
        value=f"{m_lgbm['spoilage']:.1f} units",
        delta=f"-{waste_delta_pct:.1f}% vs Baseline" if waste_delta_pct > 0 else f"+{-waste_delta_pct:.1f}% vs Baseline",
        delta_color="normal",
    )
with kpi_col4:
    st.metric(
        label="Total Operational Cost",
        value=f"${m_lgbm['total_cost']:,.2f}",
        delta=f"-{cost_delta_pct:.1f}% vs Baseline" if cost_delta_pct > 0 else f"+{-cost_delta_pct:.1f}% vs Baseline",
        delta_color="normal",
    )

st.markdown("---")

# Aggregate Daily Inventory Trajectory Chart
st.subheader("📦 Daily Aggregate On-Hand Inventory Trajectory")
st.markdown("Watch how LightGBM tracks demand closely while avoiding excessive capital lockup.")

daily_base = baseline_sim.groupby("date")["on_hand"].sum().reset_index()
daily_lgbm = lgbm_sim.groupby("date")["on_hand"].sum().reset_index()
daily_demand = baseline_sim.groupby("date")["demand"].sum().reset_index()

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=daily_base["date"],
    y=daily_base["on_hand"],
    name="Baseline Policy (MA-28)",
    line={"color": "#10B981", "width": 2.5, "dash": "dash"},
))
fig_sim.add_trace(go.Scatter(
    x=daily_lgbm["date"],
    y=daily_lgbm["on_hand"],
    name="Model Policy (LightGBM)",
    line={"color": "#2563EB", "width": 3},
))
fig_sim.add_trace(go.Bar(
    x=daily_demand["date"],
    y=daily_demand["demand"],
    name="Daily Realized Demand",
    marker_color="rgba(107, 114, 128, 0.25)",
))
fig_sim.update_layout(
    title=f"Inventory Path (L={lead_time} days, SLA={int(service_level*100)}%)",
    xaxis_title="Simulation Date",
    yaxis_title="Total On-Hand Inventory (Units)",
    hovermode="x unified",
    template="plotly_white",
    height=450,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
)
st.plotly_chart(fig_sim, use_container_width=True)

# Detailed Economics Breakdown Table
st.subheader("💰 Supply Chain Financial Breakdown")
fin_table = [
    ["Holding Costs (Capital & Warehousing)", f"${m_base['holding_cost']:,.2f}", f"${m_lgbm['holding_cost']:,.2f}", f"${m_lgbm['holding_cost'] - m_base['holding_cost']:+,.2f}"],
    ["Stockout Penalties (Unmet Demand)", f"${m_base['stockout_cost']:,.2f}", f"${m_lgbm['stockout_cost']:,.2f}", f"${m_lgbm['stockout_cost'] - m_base['stockout_cost']:+,.2f}"],
    ["Spoilage / Waste Write-Offs", f"${m_base['spoilage_cost']:,.2f}", f"${m_lgbm['spoilage_cost']:,.2f}", f"${m_lgbm['spoilage_cost'] - m_base['spoilage_cost']:+,.2f}"],
    ["Total Operational Cost", f"${m_base['total_cost']:,.2f}", f"${m_lgbm['total_cost']:,.2f}", f"${m_lgbm['total_cost'] - m_base['total_cost']:+,.2f} ({cost_delta_pct:+.1f}%)"],
]
df_fin = pd.DataFrame(fin_table, columns=["Financial Cost Component", "Baseline Policy", "LightGBM Policy", "Difference"])
st.dataframe(df_fin, use_container_width=True, hide_index=True)
