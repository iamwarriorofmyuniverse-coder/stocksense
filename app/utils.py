"""Shared utility functions and artifact loaders for the Streamlit dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")


@st.cache_data
def load_backtest_predictions() -> pd.DataFrame:
    """Load precomputed 3-fold backtest predictions."""
    file_path = ARTIFACTS_DIR / "backtest_predictions.parquet"
    if not file_path.exists():
        st.error(f"Artifact not found: {file_path}. Please run `make train` first.")
        st.stop()
    return pd.read_parquet(file_path)


@st.cache_data
def load_failure_analysis() -> pd.DataFrame:
    """Load SKU-level failure analysis and demand classifications."""
    file_path = ARTIFACTS_DIR / "failure_analysis.parquet"
    if not file_path.exists():
        st.error(f"Artifact not found: {file_path}. Please run `make evaluate` first.")
        st.stop()
    return pd.read_parquet(file_path)


@st.cache_data
def load_granular_evaluation() -> dict:
    """Load segment-wise performance metrics JSON."""
    file_path = ARTIFACTS_DIR / "granular_evaluation.json"
    if not file_path.exists():
        st.error(f"Artifact not found: {file_path}. Please run `make evaluate` first.")
        st.stop()
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_simulation_summary() -> dict:
    """Load precomputed replenishment simulation summary."""
    file_path = ARTIFACTS_DIR / "simulation_summary.json"
    if not file_path.exists():
        st.error(f"Artifact not found: {file_path}. Please run `make simulate` first.")
        st.stop()
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_simulation_results() -> pd.DataFrame:
    """Load precomputed daily simulation trajectory."""
    file_path = ARTIFACTS_DIR / "simulation_results.parquet"
    if not file_path.exists():
        st.error(f"Artifact not found: {file_path}. Please run `make simulate` first.")
        st.stop()
    return pd.read_parquet(file_path)
