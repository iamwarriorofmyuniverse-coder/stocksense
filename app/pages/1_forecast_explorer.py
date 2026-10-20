"""Page 1: Forecast Explorer - SKU-Level Forecast vs Actual with Uncertainty Bands."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils import load_backtest_predictions, load_failure_analysis

st.set_page_config(page_title="Forecast Explorer | StockSense", page_icon="📈", layout="wide")

st.title("📈 Forecast Explorer: 28-Day SKU Deep Dive")
st.markdown("Inspect forward forecasts, actual retail sales, and empirical uncertainty bands across stores and categories.")

df_preds = load_backtest_predictions()
df_failure = load_failure_analysis()

# Select final held-out backtest fold
max_fold = df_preds["fold"].max()
held_out = df_preds[df_preds["fold"] == max_fold].copy()

# Filters in sidebar
st.sidebar.header("🔍 Filters")
stores = sorted(held_out["store_id"].unique())
selected_store = st.sidebar.selectbox("Select Store", stores, index=0)

store_skus = held_out[held_out["store_id"] == selected_store]
categories = sorted(store_skus["cat_id"].unique())
selected_cat = st.sidebar.selectbox("Select Category", categories, index=0)

cat_skus = store_skus[store_skus["cat_id"] == selected_cat]
item_ids = sorted(cat_skus["item_id"].unique())
selected_item = st.sidebar.selectbox("Select Item ID", item_ids, index=0)

# Filter SKU data
sku_df = cat_skus[cat_skus["item_id"] == selected_item].sort_values("date").reset_index(drop=True)

# Lookup metadata
meta_match = df_failure[(df_failure["store_id"] == selected_store) & (df_failure["item_id"] == selected_item)]
demand_type = meta_match["demand_type"].iloc[0] if not meta_match.empty else "Unknown"
adi = meta_match["adi"].iloc[0] if not meta_match.empty else 1.0
cv2 = meta_match["cv2"].iloc[0] if not meta_match.empty else 0.5
zero_pct = meta_match["zero_percentage"].iloc[0] if not meta_match.empty else 0.0

# SKU Information header
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Demand Velocity Quadrant", demand_type)
with col2:
    st.metric("Average Demand Interval (ADI)", f"{adi:.2f} days")
with col3:
    st.metric("Coeff. of Variation (CV²)", f"{cv2:.2f}")
with col4:
    st.metric("Zero Demand Frequency", f"{zero_pct:.1f}%")

st.markdown("---")

# Model comparison checkboxes
st.sidebar.markdown("---")
st.sidebar.subheader("Models to Display")
show_lgbm = st.sidebar.checkbox("LightGBM (Global Model)", value=True)
show_ma28 = st.sidebar.checkbox("Moving Average (28-day Baseline)", value=True)
show_snaive = st.sidebar.checkbox("Seasonal Naive (7-day Baseline)", value=False)
show_bands = st.sidebar.checkbox("Show 95% Confidence Band (±1.96σ)", value=True)

# Build interactive Plotly chart
fig = go.Figure()

# 1. Ground Truth Actual Sales
fig.add_trace(go.Scatter(
    x=sku_df["date"],
    y=sku_df["sales"],
    mode="lines+markers",
    name="Actual Sales",
    line={"color": "#111827", "width": 2.5},
    marker={"size": 6, "color": "#111827"},
))

# 2. LightGBM Forecast and Uncertainty Band
if show_lgbm and "pred_lgbm" in sku_df.columns:
    preds = sku_df["pred_lgbm"].to_numpy()
    residuals = sku_df["sales"].to_numpy() - preds
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0
    upper_bound = preds + 1.96 * sigma
    lower_bound = np.maximum(0.0, preds - 1.96 * sigma)

    if show_bands:
        fig.add_trace(go.Scatter(
            x=sku_df["date"].tolist() + sku_df["date"].tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            name="LightGBM 95% Error Band",
        ))

    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=preds,
        mode="lines+markers",
        name="LightGBM Forecast",
        line={"color": "#2563EB", "width": 3, "dash": "solid"},
        marker={"size": 5, "color": "#2563EB"},
    ))

# 3. Moving Average 28d
if show_ma28 and "pred_moving_average_28d" in sku_df.columns:
    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=sku_df["pred_moving_average_28d"],
        mode="lines",
        name="MA-28 Baseline",
        line={"color": "#10B981", "width": 2, "dash": "dash"},
    ))

# 4. Seasonal Naive 7d
if show_snaive and "pred_seasonal_naive_7d" in sku_df.columns:
    fig.add_trace(go.Scatter(
        x=sku_df["date"],
        y=sku_df["pred_seasonal_naive_7d"],
        mode="lines",
        name="Seasonal Naive (s=7)",
        line={"color": "#F59E0B", "width": 1.5, "dash": "dot"},
    ))

fig.update_layout(
    title=f"28-Day Forward Trajectory: {selected_store} | {selected_item} ({demand_type} Demand)",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode="x unified",
    template="plotly_white",
    height=500,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
)

st.plotly_chart(fig, use_container_width=True)

# Performance scorecard on this specific SKU
st.subheader(f"📊 Accuracy Scorecard for {selected_item}")
y_t = sku_df["sales"].to_numpy()
sum_t = float(np.sum(y_t))
if sum_t > 0:
    wape_lgbm = float(np.sum(np.abs(y_t - sku_df["pred_lgbm"].to_numpy())) / sum_t) * 100
    wape_ma28 = float(np.sum(np.abs(y_t - sku_df["pred_moving_average_28d"].to_numpy())) / sum_t) * 100
    bias_lgbm = float(np.sum(sku_df["pred_lgbm"].to_numpy() - y_t) / sum_t) * 100
    bias_ma28 = float(np.sum(sku_df["pred_moving_average_28d"].to_numpy() - y_t) / sum_t) * 100

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("LightGBM WAPE", f"{wape_lgbm:.1f}%")
    with sc2:
        st.metric("MA-28 Baseline WAPE", f"{wape_ma28:.1f}%")
    with sc3:
        st.metric("LightGBM Bias", f"{bias_lgbm:+.1f}%")
    with sc4:
        st.metric("Winner on this SKU", "LightGBM 🏆" if wape_lgbm < wape_ma28 else "MA-28 Heuristic 🏆")
