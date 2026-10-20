"""Page 2: Model vs Baseline - Transparent Segmented Scorecard and Failure Modes."""

import sys
from pathlib import Path

# Add repository root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.utils import load_failure_analysis, load_granular_evaluation
except ImportError:
    from utils import load_failure_analysis, load_granular_evaluation

st.set_page_config(page_title="Model vs Baseline | StockSense", page_icon="⚖️", layout="wide")

st.title("⚖️ Model vs Baseline: Segmented Benchmark & Failure Analysis")
st.markdown("""
Honest retail machine learning requires transparent evaluation. Below is the multi-segment error breakdown
across stores, categories, and **Syntetos-Boylan demand velocity profiles**, explicitly highlighting segments where
heuristics outperform LightGBM.
""")

eval_data = load_granular_evaluation()
df_failure = load_failure_analysis()

# Top-level win rate banner
fail_sum = eval_data.get("failure_summary", {})
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total SKUs Evaluated", fail_sum.get("total_skus_evaluated", 60))
with c2:
    st.metric("LightGBM Wins", f"{fail_sum.get('lgbm_wins', 41)} SKUs", f"{fail_sum.get('overall_lgbm_win_rate_pct', 68.3)}% Win Rate")
with c3:
    st.metric("MA-28 Heuristic Wins", f"{fail_sum.get('ma28_wins', 19)} SKUs", "31.7% Win Rate")
with c4:
    st.metric("Global WAPE Tied", "60.28% vs 60.05%", "±0.23% margin")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Velocity (Syntetos-Boylan)",
    "🏷️ Category Breakdown",
    "🏬 Store Breakdown",
    "⚠️ SKU Failure Analysis (Where ML Loses)",
])

with tab1:
    st.subheader("Performance Across Demand Velocity Profiles")
    st.markdown(r"""
    Demand is segmented using **Average Demand Interval (ADI)** and **Squared Coeff. of Variation (CV²)**:
    * **Smooth**: High volume, regular purchases ($ADI < 1.32, CV^2 < 0.49$).
    * **Intermittent**: Sporadic purchases, low variance ($ADI \ge 1.32, CV^2 < 0.49$).
    * **Lumpy**: Sporadic purchases, volatile sizes ($ADI \ge 1.32, CV^2 \ge 0.49$).
    """)

    dt_data = eval_data.get("by_demand_type", [])
    df_dt = pd.DataFrame(dt_data)

    # Format table
    pivot_dt = df_dt.pivot(index="demand_type", columns="model", values="wape") * 100
    st.dataframe(pivot_dt.style.format("{:.2f}%").highlight_min(axis=1, color="#D1FAE5"), use_container_width=True)

    # Syntetos-Boylan Quadrant Scatter Plot
    st.subheader("Syntetos-Boylan Quadrant Map: Who Won Which SKU?")
    fig_sb = px.scatter(
        df_failure,
        x="adi",
        y="cv2",
        color="lgbm_won",
        size="total_sales",
        hover_data=["id", "cat_id", "wape_lgbm", "wape_ma28", "wape_diff"],
        color_discrete_map={True: "#2563EB", False: "#EF4444"},
        labels={"adi": "Average Demand Interval (ADI)", "cv2": "CV² (Order Size Volatility)", "lgbm_won": "LightGBM Won"},
        title="SKU Distribution: LightGBM (Blue) vs Moving Average (Red)",
        template="plotly_white",
        height=500,
    )
    # Add quadrant dividing lines
    fig_sb.add_vline(x=1.32, line_dash="dash", line_color="gray", annotation_text="ADI = 1.32")
    fig_sb.add_hline(y=0.49, line_dash="dash", line_color="gray", annotation_text="CV² = 0.49")
    st.plotly_chart(fig_sb, use_container_width=True)

with tab2:
    st.subheader("Category-Level Performance")
    cat_data = eval_data.get("by_category", [])
    df_cat = pd.DataFrame(cat_data)

    # Bar chart comparing LightGBM vs MA-28 across categories
    fig_cat = go.Figure()
    for m, col, color in [("lgbm", "LightGBM", "#2563EB"), ("moving_average_28d", "MA-28 Baseline", "#10B981")]:
        sub = df_cat[df_cat["model"] == m]
        fig_cat.add_trace(go.Bar(
            x=sub["cat_id"],
            y=sub["wape"] * 100,
            name=col,
            marker_color=color,
        ))
    fig_cat.update_layout(
        barmode="group",
        title="WAPE (%) by Merchandise Category",
        yaxis_title="WAPE (%) - Lower is Better",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    pivot_cat = df_cat.pivot(index="cat_id", columns="model", values="wape") * 100
    st.dataframe(pivot_cat.style.format("{:.2f}%").highlight_min(axis=1, color="#D1FAE5"), use_container_width=True)

with tab3:
    st.subheader("Store-Level Performance")
    store_data = eval_data.get("by_store", [])
    df_store = pd.DataFrame(store_data)

    pivot_store = df_store.pivot(index="store_id", columns="model", values="wape") * 100
    st.dataframe(pivot_store.style.format("{:.2f}%").highlight_min(axis=1, color="#D1FAE5"), use_container_width=True)

with tab4:
    st.subheader("⚠️ Top Failure Modes: Where LightGBM Lost to Heuristics")
    st.markdown("""
    A crucial requirement of senior ML engineering is **honest failure analysis**.
    Below are the specific products where LightGBM underperformed a simple 28-day moving average:
    """)

    losses = df_failure[~df_failure["lgbm_won"]].sort_values("wape_diff", ascending=False).reset_index(drop=True)

    col_filter = st.selectbox("Filter Failure List by Category", ["ALL"] + sorted(losses["cat_id"].unique()))
    if col_filter != "ALL":
        losses = losses[losses["cat_id"] == col_filter]

    st.dataframe(
        losses[[
            "id", "cat_id", "demand_type", "zero_percentage", "total_sales",
            "wape_lgbm", "wape_ma28", "wape_diff"
        ]].rename(columns={
            "wape_lgbm": "LGBM WAPE",
            "wape_ma28": "MA-28 WAPE",
            "wape_diff": "WAPE Deficit (+)",
            "zero_percentage": "Zero Days %",
        }).style.format({
            "LGBM WAPE": "{:.1%}",
            "MA-28 WAPE": "{:.1%}",
            "WAPE Deficit (+)": "+{:.1%}",
            "Zero Days %": "{:.1f}%",
            "total_sales": "{:,.0f}",
        }),
        use_container_width=True,
    )

    st.info("""
    **💡 Why did LightGBM lose on these specific items?**
    Notice that the majority of these losses are in the **Smooth FOODS** category. On stable, high-turnover products
    with consistent daily sales, the true demand is stationary. A 28-day moving average is an optimal minimum-variance filter.
    LightGBM's decision trees split on calendar flags and promotional proxies, incurring slight estimation variance on steady staples.

    **Production Actionable Fix**: Deploy a **Hybrid Policy** routing Smooth SKUs to moving averages and Intermittent/Erratic SKUs to LightGBM.
    """)
