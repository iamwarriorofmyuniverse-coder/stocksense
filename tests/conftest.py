"""Pytest fixtures for StockSense testing suite."""

import pytest

from src.data.synthetic import generate_synthetic_m5


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Generate a fast, small synthetic M5 dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("m5_synthetic")
    generate_synthetic_m5(
        output_dir=tmp_dir,
        num_days=90,
        num_stores=2,
        num_items=12,
        random_seed=42,
    )
    return tmp_dir


@pytest.fixture
def sample_panel_df(synthetic_data_dir):
    """Load and format sample long panel DataFrame."""
    from src.data.loader import DataLoader
    loader = DataLoader(synthetic_data_dir)
    cal, prices, sales = loader.load_raw()
    return loader.melt_and_merge(sales, cal, prices)
