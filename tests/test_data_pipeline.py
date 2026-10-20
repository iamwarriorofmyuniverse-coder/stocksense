"""Unit tests for M5 data generation, validation, and loading."""

import pytest

from src.data.validator import DataValidationError, validate_processed_data, validate_raw_data
from src.features.pipeline import FeaturePipeline


def test_synthetic_data_generation(synthetic_data_dir):
    """Verify synthetic files exist and pass raw validation."""
    report = validate_raw_data(synthetic_data_dir)
    assert report["calendar_valid"] is True
    assert report["sell_prices_valid"] is True
    assert report["sales_valid"] is True
    assert report["total_days"] == 90


def test_validator_detects_missing_file(tmp_path):
    """Verify DataValidationError when required CSV files are missing."""
    with pytest.raises(DataValidationError, match="Missing calendar file"):
        validate_raw_data(tmp_path)


def test_data_loader_downcasting(sample_panel_df):
    """Verify DataLoader downcasts types to conserve memory."""
    assert sample_panel_df["sales"].dtype == "int16"
    assert sample_panel_df["sell_price"].dtype == "float32"
    assert str(sample_panel_df["id"].dtype) == "category"
    assert "sales" in sample_panel_df.columns
    assert "date" in sample_panel_df.columns


def test_feature_pipeline_output(sample_panel_df):
    """Verify FeaturePipeline transforms panel data and extracts expected features."""
    pipeline = FeaturePipeline()
    feat_df, feature_names = pipeline.fit_transform(sample_panel_df, drop_warmup=True)

    assert len(feat_df) > 0
    assert len(feature_names) >= 20
    assert "lag_28" in feature_names
    assert "rolling_mean_7_lag_28" in feature_names
    assert "price_discount" in feature_names
    assert "day_of_week" in feature_names

    # Check that processed data validates cleanly
    assert validate_processed_data(feat_df, feature_names) is True
