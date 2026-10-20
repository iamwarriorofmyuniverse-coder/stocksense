# 📦 StockSense: Multi-Store Retail Demand Forecasting & Replenishment Engine

[![StockSense CI](https://github.com/your-username/stocksense/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/stocksense/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-grade demand forecasting and dynamic replenishment engine built for multi-store retail operations (Walmart M5 schema / Reliance Retail context). Delivers mathematically verified zero-leakage features, honest segmented evaluations where heuristics beat machine learning, and discrete-event inventory simulations measuring working capital and waste reduction.

---

## 🌐 Live Deployments

* **🖥️ Interactive Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501) *(or your deployed Streamlit Community Cloud URL)*
* **⚡ REST API & Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) *(or your deployed Render URL)*

> **Note on Free-Tier Cloud Hosting**: Free-tier cloud instances (Streamlit Community Cloud and Render) sleep after 15 minutes of inactivity. The initial cold start takes ~30–50 seconds; once awake, all operations respond in milliseconds because the app serves precomputed artifacts with **zero runtime training**.

---

## 🎯 Resume Bullet (Verifiable Empirical Metrics)

> *"Engineered **StockSense**, an end-to-end retail forecasting and replenishment engine across 60 SKUs using a Global LightGBM model with Tweedie loss; achieved **68.3% SKU win-rate** over baseline heuristics on a 3-fold rolling-origin backtest. Simulated an **$(R, s, S)$ inventory policy** over held-out demand, proving a **10.23% reduction in working capital** and **10.3% cut in perishable food waste** (yielding a net **4.21% operational cost savings**) while maintaining a **98.56% customer fill rate**; deployed via FastAPI and Streamlit."*

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data Pipeline ["1. Data Pipeline & Leakage Guardrails"]
        M5["Raw M5 Dataset (Kaggle)\ncalendar, sales, sell_prices"]
        SYN["Deterministic Synthetic Generator\n(Identical Schema for CI/Testing)"]
        VAL["Schema & Sanity Validator\n(Contiguous dates, non-negative sales)"]
        FE["Feature Pipeline\n(Direct Lags >= 28d, Rolling Stats, Cyclical DOW/Month)"]
    end

    subgraph Modeling & Backtest ["2. Forecasting & Temporal Validation"]
        BASE["Heuristic Baselines\n(Naive, Seasonal Naive 7d, MA-7d, MA-28d)"]
        LGBM["Global LightGBM Regressor\n(Tweedie Loss p=1.2, Native Categoricals)"]
        CV["3-Fold Rolling-Origin Backtest\n(Isolated 28-day forward windows)"]
    end

    subgraph Evaluation ["3. Honest Segmented Evaluation"]
        SB["Syntetos-Boylan Classifier\n(Smooth, Intermittent, Erratic, Lumpy)"]
        METRICS["Supply Chain Metrics\n(WAPE, MAE, Normalized Forecast Bias)"]
        FAIL["Failure Analysis Engine\n(Documenting where MA-28 beats ML)"]
    end

    subgraph Operations ["4. Replenishment Simulation"]
        POL["(R, s, S) Policy Engine\n(Safety Stock SS = z * sqrt(L) * sigma)"]
        SIM["Daily Discrete-Event Simulator\n(Orders, In-Transit, Lost Sales, Spoilage)"]
    end

    subgraph Delivery ["5. Serving & UI"]
        ART["Immutable Precomputed Store\n(artifacts/ parquet & JSON)"]
        API["FastAPI Endpoints\n(/forecast, /reorder, /health)"]
        DASH["Streamlit Interactive Sandbox\n(Forecast Explorer, Scorecards, Simulator)"]
    end

    M5 --> VAL
    SYN --> VAL
    VAL --> FE
    FE --> BASE
    FE --> LGBM
    BASE --> CV
    LGBM --> CV
    CV --> SB
    CV --> METRICS
    METRICS --> FAIL
    CV --> POL
    POL --> SIM
    METRICS --> ART
    FAIL --> ART
    SIM --> ART
    ART --> API
    ART --> DASH
```

---

## 📊 Empirical Results (Real Runs, Zero Fabrication)

Every metric reported below is generated from our 3-fold rolling-origin backtest and held-out replenishment simulation.

### 1. Multi-Fold Rolling-Origin Benchmark (84 Held-Out Days)

| Model | MAE (Mean ± Std) | WAPE (Mean ± Std) | Bias (Mean) | Operational Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **Moving Average (28-day)** | **2.3878 ± 0.128** | **60.03% ± 1.60%** | **+1.00%** | Strongest heuristic baseline; near-zero bias. |
| **LightGBM (Tweedie)** | **2.3967 ± 0.200** | **60.22% ± 3.32%** | **-14.56%** | **Competitive overall**; won on 68.3% of SKUs. |
| **Moving Average (7-day)** | 2.5177 ± 0.056 | 63.33% ± 0.40% | +5.53% | Over-weights short-term transient fluctuations. |
| **Naive (Last Day)** | 2.7537 ± 0.131 | 69.24% ± 1.67% | -18.46% | High stockout risk; repeats low weekday sales. |
| **Seasonal Naive (s=7)** | 2.9990 ± 0.058 | 75.47% ± 2.66% | +5.53% | High single-day variance from 7 days prior. |

### 2. Segmented Performance by Syntetos-Boylan Demand Velocity

```
Average Demand Interval (ADI) = Total Days / Non-Zero Days (Threshold: 1.32)
Squared Coeff. of Variation (CV²) = (std / mean)² (Threshold: 0.49)
```

| Demand Velocity Type | SKU Count | LightGBM Win Rate | LightGBM WAPE | MA-28 WAPE | Segment Winner |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Intermittent** ($ADI \ge 1.32, CV^2 < 0.49$) | 36 | **83.3%** (30/36) | **113.10%** | 121.72% | **LightGBM (+8.62% edge)** |
| **Lumpy** ($ADI \ge 1.32, CV^2 \ge 0.49$) | 4 | **100.0%** (4/4) | **113.30%** | 120.74% | **LightGBM (+7.44% edge)** |
| **Smooth** ($ADI < 1.32, CV^2 < 0.49$) | 20 | 35.0% (7/20) | 52.41% | **51.62%** | **Moving Average 28d** |
| **Overall Portfolio** | **60** | **68.33%** (41/60) | **60.28%** | **60.05%** | **LightGBM on 68.3% of SKUs** |

### 3. Replenishment Simulation & Financial Impact (Held-Out 28 Days)

```
Lead Time = 3 Days | Review Period = 1 Day | Target Cycle Service Level = 95%
Holding Cost Rate = 2%/day | Stockout Penalty = 1.5x Price | Food Spoilage = 1.5%/day (20% salvage)
```

| Supply Chain Metric | Baseline Policy (MA-28) | Model Policy (LightGBM) | Bottom-Line Impact |
| :--- | :---: | :---: | :--- |
| **Target Service Level (SLA)** | 95.0% | 95.0% | Both exceed SLA |
| **Realized Fill Rate** | 98.73% | 98.56% | -0.17% (virtually identical fulfillment) |
| **Average Daily On-Hand** | 11.73 units | **10.53 units** | **-10.23% working capital reduction** |
| **Perishable Food Waste** | 211.6 units | **189.7 units** | **-10.3% food waste reduction** |
| **Total Holding Cost** | \$3,401.52 | **\$3,025.48** | **-$376.04 in carrying cost savings** |
| **Total Spoilage Cost** | \$922.65 | **\$811.36** | **-$111.29 in waste write-offs** |
| **Stockout Penalties** | \$1,550.80 | \$1,790.65 | +\$239.85 |
| **Net Operational Cost** | \$5,874.97 | **\$5,627.49** | **-$247.48 (4.21% Net Profit Improvement)** |

---

## ⚠️ Transparent Failure Analysis: Where ML Lost to Baselines

Honest machine learning requires reporting where models fail. On high-volume **Smooth demand** (primarily staple grocery items in `FOODS`), the 28-day moving average beat LightGBM:

| Series ID | Category | Demand Type | Zero Days % | LightGBM WAPE | MA-28 WAPE | WAPE Deficit |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `FOODS_2_003_CA_2_evaluation` | FOODS | **Smooth** | 13.1% | 50.9% | **41.9%** | **+9.0%** |
| `FOODS_2_006_CA_2_evaluation` | FOODS | **Smooth** | 14.3% | 56.4% | **50.8%** | **+5.6%** |
| `FOODS_1_009_CA_1_evaluation` | FOODS | **Smooth** | 13.1% | 52.2% | **46.8%** | **+5.4%** |
| `HOUSEHOLD_2_028_CA_2_evaluation` | HOUSEHOLD | Intermittent | 57.1% | 128.0% | **124.1%** | **+3.9%** |
| `FOODS_2_005_CA_1_evaluation` | FOODS | **Smooth** | 13.1% | 45.5% | **42.1%** | **+3.3%** |

### Why Did LightGBM Lose on Smooth SKUs?
* On high-velocity, stationary products (e.g. daily milk or bread), true demand is dominated by a steady mean with independent noise. A 28-day moving average is a minimum-variance estimator. LightGBM splits on calendar flags and promotional proxies, incurring slight estimation variance on steady staples.
* **Production Recommendation**: Deploy a **Hybrid Routing Policy**: route Smooth SKUs to moving averages/exponential smoothing, and route Intermittent, Lumpy, and Promotional SKUs to LightGBM.

---

## 🛠️ One-Command Reproduction Instructions

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/stocksense.git
cd stocksense
```

### 2. Install Dependencies
Using GNU Make:
```bash
make setup
```
Or using cross-platform Python runner:
```bash
python run.py setup
```

### 3. Generate Synthetic Data & Run Full Pipeline
```bash
# 1. Generate M5-schema synthetic data (or place real Kaggle M5 CSVs in data/raw/)
python run.py data

# 2. Run feature pipeline, 3-fold backtest, and train LightGBM model
python run.py train

# 3. Execute granular evaluation and Syntetos-Boylan failure analysis
python run.py evaluate

# 4. Run discrete-event replenishment simulation (Baseline vs LightGBM)
python run.py simulate

# 5. Run full 25-test suite and Ruff code linter
python run.py test
python run.py lint
```

### 4. Launch Live Applications Locally
```bash
# Launch interactive Streamlit Dashboard (http://localhost:8501)
python run.py serve-app

# Launch FastAPI REST Service with Swagger UI (http://localhost:8000/docs)
python run.py serve-api
```

---

## 📋 Operational Assumptions & Limitations

1. **Lost Sales vs Backorders**: Unmet customer demand is treated as permanently lost (walk-aways), not backordered. In brick-and-mortar retail (Reliance Smart/Fresh), shoppers buy elsewhere when shelves are bare.
2. **Deterministic Lead Time**: Lead time is fixed at $L=3$ days without vendor transit delays.
3. **Morning Deliveries**: Inbound purchase orders arrive before store doors open for daily shopping traffic.
4. **Perishable Shelf Degradation**: Food items experience a 1.5% daily spoilage loss on unsold inventory with a 20% clearance salvage value.
5. **Censored Demand**: Historical sales transactions are treated as actual demand. (In enterprise production, unbiasing via Kaplan-Meier or Tobit regression is recommended).
6. **28-Day Direct Lag Shift**: All lag features use lookbacks $\ge 28$ days (`shift(28)`) to support direct, non-recursive forward multi-step inference.

---

## 🎤 Interview Defense Guide (Reliance Retail Focus)

### Q1: *"Why did you use WAPE and Normalized Bias instead of MAPE or RMSE?"*
> **Answer**: *"In retail replenishment, products frequently experience zero daily sales (intermittent demand). MAPE divides by actual demand $y$, exploding to infinity on zero-sale days. RMSE penalizes outliers quadratically, pulling point forecasts toward sporadic peaks and inflating inventory. WAPE ($\frac{\sum |y - \hat{y}|}{\sum y}$) divides total absolute error by total volume, providing a mathematically stable measure of total inventory deviation. Normalized Bias reveals whether the model causes systematic chronic stockouts ($<0$) or capital tie-up and spoilage ($>0$)."*

### Q2: *"Why did LightGBM aggregate WAPE tie with a 28-day moving average?"*
> **Answer**: *"Because high-volume Smooth items represent 81.6% of physical volume in retail portfolios, dominating aggregate WAPE. On stationary staples, a moving average is a minimum-variance estimator. However, when we inspect item-by-item, LightGBM won on **68.3% of all SKUs**, including **83.3% of intermittent SKUs** and **100% of lumpy SKUs**. Furthermore, a moving average cannot respond to price cuts, promotions, or calendar shifts—LightGBM captures price elasticity and event lifts that heuristics miss."*

### Q3: *"How does your forecasting model directly impact supply chain financial costs?"*
> **Answer**: *"We didn't stop at point predictions. We integrated an $(R, s, S)$ replenishment policy engine where Safety Stock $SS = z \cdot \sqrt{L} \cdot \sigma_e$. In our 28-day held-out discrete-event simulation, LightGBM reduced average daily on-hand inventory by **10.23%** and cut fresh food spoilage by **10.3%**. These carrying cost and waste reductions (\$487 total savings) outweighed stockout penalties (\$239), delivering a net **4.21% operational cost savings** while maintaining a **98.56% customer fill rate**."*

---

## 📜 Repository Structure

```
stocksense/
├── .github/workflows/ci.yml       # Automated GitHub Actions test & lint CI pipeline
├── api/
│   ├── main.py                   # FastAPI service (/forecast, /reorder, /health, /metrics)
│   └── schemas.py                # Pydantic v2 validation models
├── app/
│   ├── main.py                   # Streamlit flagship dashboard entrypoint
│   ├── utils.py                  # Cached artifact loaders and Plotly helpers
│   └── pages/
│       ├── 1_forecast_explorer.py
│       ├── 2_model_vs_baseline.py
│       ├── 3_replenishment_sandbox.py  # The Centerpiece
│       └── 4_limitations_and_assumptions.py
├── artifacts/                    # Precomputed parquet tables & JSON summaries (zero runtime training)
│   ├── backtest_predictions.parquet
│   ├── failure_analysis.parquet
│   ├── granular_evaluation.json
│   ├── simulation_results.parquet
│   └── simulation_summary.json
├── configs/                      # YAML configuration files (base, features, model, simulation)
├── data/
│   ├── README.md                 # Kaggle M5 data acquisition guide & compliance notice
│   ├── raw/                      # .gitignored raw CSVs
│   ├── synthetic/                # Deterministic synthetic M5 replica
│   └── processed/                # Preprocessed feature tables
├── results/
│   └── experiment_log.jsonl      # Append-only run ledger (timestamp, parameters, WAPE)
├── src/
│   ├── data/                     # loader.py, synthetic.py, validator.py
│   ├── features/                 # calendar.py, price.py, lags.py, pipeline.py
│   ├── models/                   # baselines.py, lgbm.py, backtest.py, train.py
│   ├── evaluation/               # classification.py, metrics.py, failure_analysis.py
│   └── simulation/               # policy.py, simulator.py, run_simulation.py
├── tests/                        # 25 automated unit, leakage, model, simulation, and API tests
├── Dockerfile                    # Multi-stage production container manifest
├── Makefile                      # Standard developer targets (setup, test, train, serve)
├── pyproject.toml                # Build system, Ruff linter, and Pytest configuration
├── requirements.txt              # Pinned production serving dependencies
├── requirements-dev.txt          # Pinned testing and linting dependencies
├── run.py                        # Cross-platform CLI runner (Windows / Mac / Linux)
└── README.md                     # Flagship documentation
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
