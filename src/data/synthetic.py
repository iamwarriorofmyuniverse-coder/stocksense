"""Deterministic Synthetic M5 Dataset Generator.

Generates an exact structural replica of the Kaggle M5 Forecasting dataset:
1. calendar.csv
2. sell_prices.csv
3. sales_train_evaluation.csv

Demand dynamics include:
- Base volume variation across categories (FOODS > HOUSEHOLD > HOBBIES)
- Day-of-week seasonality (weekend shopping lifts)
- Event effects (promotions and holidays)
- Price sensitivity (price cuts boost sales)
- Intermittent demand / zero inflation for realistic retail simulation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_m5(
    output_dir: str | Path = "data/synthetic",
    num_days: int = 365,
    num_stores: int = 2,
    num_items: int = 50,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Generate deterministic synthetic M5 dataset matching the Kaggle schema."""
    rng = np.random.default_rng(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Store and State hierarchies
    # M5 naming convention: CA_1, CA_2, TX_1, WI_1
    all_stores = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1", "WI_2"]
    stores = all_stores[:num_stores]
    store_state_map = {s: s.split("_")[0] for s in stores}

    # 2. Categories and Departments
    # Categories: FOODS, HOBBIES, HOUSEHOLD
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    depts = {
        "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
        "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
        "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    }

    # Generate item definitions
    items = []
    item_metadata = []
    items_per_cat = max(1, num_items // len(categories))

    item_idx = 1
    for cat in categories:
        cat_depts = depts[cat]
        for _ in range(items_per_cat):
            dept = rng.choice(cat_depts)
            item_id = f"{dept}_{item_idx:03d}"
            base_price = (
                rng.uniform(1.5, 8.0) if cat == "FOODS"
                else rng.uniform(5.0, 25.0) if cat == "HOUSEHOLD"
                else rng.uniform(8.0, 45.0)
            )
            base_demand = (
                rng.uniform(3.0, 15.0) if cat == "FOODS"
                else rng.uniform(0.5, 4.0) if cat == "HOUSEHOLD"
                else rng.uniform(0.1, 2.0)
            )
            items.append(item_id)
            item_metadata.append({
                "item_id": item_id,
                "dept_id": dept,
                "cat_id": cat,
                "base_price": round(base_price, 2),
                "base_demand": base_demand,
            })
            item_idx += 1

    items_df = pd.DataFrame(item_metadata)

    # 3. Calendar generation
    start_date = pd.Timestamp("2015-01-01")
    dates = pd.date_range(start_date, periods=num_days, freq="D")

    # M5: wday 1 is Saturday, 2 is Sunday, ..., 7 is Friday
    # pandas dayofweek: Monday=0, Sunday=6
    # Mapping to M5: Saturday is 1, Sunday is 2, Monday is 3, ..., Friday is 7
    dow_to_m5 = {5: 1, 6: 2, 0: 3, 1: 4, 2: 5, 3: 6, 4: 7}
    wday_series = [dow_to_m5[d.dayofweek] for d in dates]
    weekday_series = [d.strftime("%A") for d in dates]

    # Walmart fiscal week: wm_yr_wk starts on Saturday
    # Construct synthetic wm_yr_wk: e.g., 11501 (year 2015, week 1)
    base_wm_wk = 11501
    wm_yr_wk_list = []
    curr_wk = base_wm_wk
    for i, d in enumerate(dates):
        if i > 0 and wday_series[i] == 1:  # New Saturday
            curr_wk += 1
            if curr_wk % 100 > 52:
                curr_wk = (curr_wk // 100 + 1) * 100 + 1
        wm_yr_wk_list.append(curr_wk)

    # Events and SNAP
    events = [
        "", "", "SuperBowl", "", "ValentinesDay", "", "StPatricksDay",
        "Easter", "", "CincoDeMayo", "MemorialDay", "", "IndependenceDay",
        "", "LaborDay", "", "Halloween", "VeteransDay", "Thanksgiving",
        "Christmas", "NewYear",
    ]
    event_types = {
        "SuperBowl": "Sporting",
        "ValentinesDay": "Cultural",
        "StPatricksDay": "Cultural",
        "Easter": "Cultural",
        "CincoDeMayo": "Cultural",
        "MemorialDay": "National",
        "IndependenceDay": "National",
        "LaborDay": "National",
        "Halloween": "Cultural",
        "VeteransDay": "National",
        "Thanksgiving": "National",
        "Christmas": "Religious",
        "NewYear": "National",
    }

    p_event = np.full(len(events), 0.15 / (len(events) - 1))
    p_event[0] = 0.85
    p_event = p_event / p_event.sum()

    event_name_1 = [rng.choice(events, p=p_event) for _ in range(num_days)]
    event_type_1 = [event_types.get(e, "") for e in event_name_1]

    calendar_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": wm_yr_wk_list,
        "weekday": weekday_series,
        "wday": wday_series,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i+1}" for i in range(num_days)],
        "event_name_1": event_name_1,
        "event_type_1": event_type_1,
        "event_name_2": "",
        "event_type_2": "",
        "snap_CA": rng.choice([0, 1], size=num_days, p=[0.7, 0.3]),
        "snap_TX": rng.choice([0, 1], size=num_days, p=[0.7, 0.3]),
        "snap_WI": rng.choice([0, 1], size=num_days, p=[0.7, 0.3]),
    })

    # 4. Sell Prices generation
    unique_weeks = calendar_df[["wm_yr_wk"]].drop_duplicates()
    price_records = []

    for store in stores:
        for _, item in items_df.iterrows():
            base_p = item["base_price"]
            for wk in unique_weeks["wm_yr_wk"]:
                # 10% chance of a discount promotion
                discount = rng.choice([0.0, 0.10, 0.20, 0.30], p=[0.85, 0.08, 0.05, 0.02])
                price = max(0.50, round(base_p * (1.0 - discount), 2))
                price_records.append({
                    "store_id": store,
                    "item_id": item["item_id"],
                    "wm_yr_wk": wk,
                    "sell_price": price,
                })

    sell_prices_df = pd.DataFrame(price_records)

    # 5. Sales Generation (Wide format: d_1 ... d_N)
    sales_rows = []
    day_columns = [f"d_{i+1}" for i in range(num_days)]

    # Pre-index prices for fast lookup: (store, item, wm_yr_wk) -> price
    price_lookup = sell_prices_df.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"].to_dict()

    for store in stores:
        state = store_state_map[store]
        snap_col = f"snap_{state}"

        for _, item in items_df.iterrows():
            series_id = f"{item['item_id']}_{store}_evaluation"
            base_lambda = item["base_demand"]

            # Series daily simulation
            daily_sales = np.zeros(num_days, dtype=np.int16)
            for d_idx, day_str in enumerate(day_columns):
                cal_row = calendar_df.iloc[d_idx]
                wday = cal_row["wday"]
                wk = cal_row["wm_yr_wk"]
                has_event = 1 if cal_row["event_name_1"] else 0
                is_snap = cal_row[snap_col]

                # Weekend lift (wday 1, 2 = Saturday, Sunday)
                dow_multiplier = 1.35 if wday in (1, 2) else 0.90
                event_multiplier = 1.40 if has_event else 1.0
                snap_multiplier = 1.15 if is_snap else 1.0

                # Price elasticity
                current_price = price_lookup.get((store, item["item_id"], wk), item["base_price"])
                price_ratio = current_price / item["base_price"]
                price_multiplier = max(0.5, 1.0 - 1.2 * (price_ratio - 1.0))

                lam = base_lambda * dow_multiplier * event_multiplier * snap_multiplier * price_multiplier

                # Intermittent zero-inflation (simulate stockout or slow demand)
                is_active = rng.random() > (0.15 if item["cat_id"] == "FOODS" else 0.45)
                if is_active:
                    daily_sales[d_idx] = rng.poisson(lam=max(0.1, lam))
                else:
                    daily_sales[d_idx] = 0

            row_dict = {
                "id": series_id,
                "item_id": item["item_id"],
                "dept_id": item["dept_id"],
                "cat_id": item["cat_id"],
                "store_id": store,
                "state_id": state,
            }
            for d_idx, day_col in enumerate(day_columns):
                row_dict[day_col] = daily_sales[d_idx]

            sales_rows.append(row_dict)

    sales_df = pd.DataFrame(sales_rows)

    # Save to CSV files matching Kaggle M5
    cal_path = output_path / "calendar.csv"
    prices_path = output_path / "sell_prices.csv"
    sales_path = output_path / "sales_train_evaluation.csv"

    calendar_df.to_csv(cal_path, index=False)
    sell_prices_df.to_csv(prices_path, index=False)
    sales_df.to_csv(sales_path, index=False)

    return {
        "calendar": cal_path,
        "sell_prices": prices_path,
        "sales": sales_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic M5 retail dataset.")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--num-days", type=int, default=365, help="Number of historical days")
    parser.add_argument("--num-stores", type=int, default=2, help="Number of retail stores")
    parser.add_argument("--num-items", type=int, default=50, help="Number of product items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    files = generate_synthetic_m5(
        output_dir=args.output_dir,
        num_days=args.num_days,
        num_stores=args.num_stores,
        num_items=args.num_items,
        random_seed=args.seed,
    )
    print(f"Synthetic M5 dataset successfully created at '{args.output_dir}':")
    for k, v in files.items():
        print(f"  - {k}: {v}")
