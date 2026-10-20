.PHONY: help setup data train evaluate simulate serve test lint format clean

PYTHON ?= python

help:
	@echo "StockSense Development Makefile"
	@echo "================================"
	@echo "Targets:"
	@echo "  setup     : Install dependencies"
	@echo "  data      : Generate synthetic M5 dataset"
	@echo "  train     : Train LightGBM model and execute rolling-origin backtest"
	@echo "  evaluate  : Run granular evaluation and Syntetos-Boylan failure analysis"
	@echo "  simulate  : Run replenishment simulation (Baseline vs Model Policy)"
	@echo "  serve-api : Launch FastAPI service"
	@echo "  serve-app : Launch Streamlit dashboard"
	@echo "  serve     : Launch Streamlit dashboard (default app)"
	@echo "  test      : Run test suite with pytest"
	@echo "  lint      : Check code style with ruff"
	@echo "  clean     : Remove temporary caches and build artifacts"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PYTHON) -m src.data.synthetic --output-dir data/synthetic --num-days 180 --num-stores 2 --num-items 30

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.evaluation.failure_analysis

simulate:
	$(PYTHON) -m src.simulation.run_simulation

serve-api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-app:
	$(PYTHON) -m streamlit run app/main.py

serve: serve-app

test:
	$(PYTHON) -m pytest tests/ -v
	$(PYTHON) -m ruff check src tests

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/**/__pycache__ tests/__pycache__
