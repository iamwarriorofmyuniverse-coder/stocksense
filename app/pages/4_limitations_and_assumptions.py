"""Page 4: Limitations and Operational Assumptions."""

import streamlit as st

st.set_page_config(page_title="Methodology & Limitations | StockSense", page_icon="📋", layout="wide")

st.title("📋 Simulation Assumptions, Limitations & Production Roadmap")
st.markdown("""
Honest engineering demands stating all model assumptions and simulation constraints explicitly.
Below are the mathematical and operational boundaries of the StockSense system.
""")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("1. Core Operational Assumptions")
    st.markdown(r"""
    * **Lost Sales (No Backorders)**:
      * *Assumption*: Any unmet customer demand is permanently lost on the day it occurs.
      * *Retail Justification*: In brick-and-mortar grocery and supermarket retail (e.g. Reliance Smart / Fresh), shoppers do not wait for backordered inventory; they walk away or purchase from a competing store.
    * **Deterministic Supplier Lead Times**:
      * *Assumption*: Supplier purchase orders arrive exactly $L$ days after ordering ($L \in [1, 14]$ days).
      * *Limitation*: In real supply chains, port congestion, vendor stockouts, and logistics delays cause stochastic lead-time variance.
    * **Morning Batch Deliveries**:
      * *Assumption*: New inventory arrives at the start of the business day prior to customer store traffic.
      * *Limitation*: Intraday cross-docking or multi-wave replenishment is not modeled in this release.
    * **FIFO Spoilage Rate on Perishables**:
      * *Assumption*: Food items (category `FOODS`) degrade at a fixed daily rate on shelf (1.5%/day) with 20% clearance salvage recovery.
    """)

with col_b:
    st.subheader("2. Modeling & Demand Limitations")
    st.markdown(r"""
    * **Censored Sales vs Unconstrained Demand**:
      * *Assumption*: Recorded historical transactions are treated as actual consumer demand.
      * *Limitation*: If an item was previously stocked out on day $t-20$, historical sales recorded 0. This artificially depresses rolling demand estimates (censorship bias). In enterprise production, Kaplan-Meier or Tobit regression unbiasing is applied.
    * **The 28-Day Horizon Direct Lag Trade-Off**:
      * *Assumption*: All lag features look back $\ge 28$ days to allow non-recursive, single-pass forward inference.
      * *Trade-Off*: The model cannot observe sales from days $t-1$ through $t-27$. For day $t+1$, a 1-day lag would have been more accurate, but for day $t+28$, it is unobserved. We intentionally traded short-term accuracy for multi-step operational consistency.
    * **Tweedie Loss Negative Bias**:
      * *Limitation*: The Tweedie distribution ($p=1.2$) slightly under-forecasts demand peaks (-14.5% bias) on sporadic items to avoid large likelihood penalties on zeros. This is compensated by safety stock in the replenishment policy.
    """)

st.markdown("---")

st.subheader("🚀 Production Roadmap for Enterprise Retail (Reliance Retail Scale)")
st.markdown(r"""
To transition this engine from flagship portfolio to enterprise deployment at multi-thousand store scale:
1. **Hybrid Policy Routing**:
   * Automatically route **Smooth, high-velocity SKUs** to Exponential Smoothing / Moving Average (which won 65% of smooth items).
   * Route **Intermittent, Lumpy, and Promotional SKUs** to the Global LightGBM model (which won 83–100% of sporadic items).
2. **Stochastic Lead Time Modeling**:
   * Replace deterministic $L$ with empirical lead-time distributions: $SS = z \cdot \sqrt{L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2}$.
3. **Multi-Echelon DC-to-Store Optimization**:
   * Expand from single-store replenishment to multi-echelon inventory: Regional Distribution Centers (RDCs) buffering Mother Warehouses and feeding local retail stores via consolidated truckloads (Full Truckload / FTL optimization).
4. **Demand Cannibalization & Halo Effects**:
   * Incorporate cross-item price elasticity: discounting Brand A soda reduces demand for Brand B soda while increasing demand for salty snacks.
""")
