# Data Directory Guide: Kaggle M5 Forecasting Dataset

This project is built for multi-store retail demand forecasting and replenishment using the schema of the **Kaggle M5 Forecasting - Accuracy** competition (Walmart US dataset).

## Competition Rules & Data Redistribution
Per Kaggle's competition rules and terms of use, the raw M5 competition dataset cannot be redistributed or bundled directly inside this repository.

## How to Obtain the Raw Data
1. Visit the competition page: [Kaggle M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
2. Accept the competition rules and download the following files:
   - `calendar.csv`
   - `sales_train_evaluation.csv` (or `sales_train_validation.csv`)
   - `sell_prices.csv`
3. Place these files inside `data/raw/`:
   ```
   data/raw/
   ├── calendar.csv
   ├── sales_train_evaluation.csv
   └── sell_prices.csv
   ```

## Development Without Kaggle Data (Synthetic Mode)
If you do not have Kaggle credentials or wish to test the pipeline rapidly in CI or local development, StockSense includes a deterministic synthetic data generator that creates an exact structural replica of the M5 schema:

```bash
make data-synthetic
```
or run directly:
```bash
python -m src.data.synthetic --num-stores 2 --num-items 50 --num-days 365
```
All schema validators, feature pipelines, baseline models, LightGBM models, and replenishment simulations execute against both synthetic data and real M5 data interchangeably.
