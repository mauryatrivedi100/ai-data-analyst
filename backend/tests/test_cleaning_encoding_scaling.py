"""Unit tests for encoding and scaling functions in cleaning module."""

import pytest
import pandas as pd
import numpy as np

from cleaning import (
    label_encode,
    one_hot_encode,
    standard_scale,
    min_max_scale,
)


class TestLabelEncode:
    """Tests for label_encode function."""

    def test_encodes_alphabetically(self):
        df = pd.DataFrame({"color": ["red", "blue", "green", "blue"]})
        result = label_encode(df, "color")
        # Alphabetical: blue=0, green=1, red=2
        assert result["color"].tolist() == [2, 0, 1, 0]

    def test_handles_nan_values(self):
        df = pd.DataFrame({"color": ["red", None, "blue", "red"]})
        result = label_encode(df, "color")
        # blue=0, red=1; NaN stays NaN
        assert result["color"].iloc[0] == 1
        assert pd.isna(result["color"].iloc[1])
        assert result["color"].iloc[2] == 0
        assert result["color"].iloc[3] == 1

    def test_single_category(self):
        df = pd.DataFrame({"status": ["active", "active", "active"]})
        result = label_encode(df, "status")
        assert result["status"].tolist() == [0, 0, 0]

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="does not exist"):
            label_encode(df, "nonexistent")

    def test_raises_for_numerical_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="numerical"):
            label_encode(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [None, None, None]})
        # Force object dtype
        df["a"] = df["a"].astype(object)
        with pytest.raises(ValueError, match="only missing values"):
            label_encode(df, "a")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        original_values = df["color"].tolist()
        label_encode(df, "color")
        assert df["color"].tolist() == original_values

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"color": ["red", "blue"], "value": [10, 20]})
        result = label_encode(df, "color")
        assert result["value"].tolist() == [10, 20]


class TestOneHotEncode:
    """Tests for one_hot_encode function."""

    def test_creates_binary_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        result = one_hot_encode(df, "color")
        assert "color" not in result.columns
        assert "color_red" in result.columns
        assert "color_blue" in result.columns
        assert "color_green" in result.columns

    def test_binary_values_correct(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"]})
        result = one_hot_encode(df, "color")
        # Row 0: red
        assert result["color_red"].iloc[0] == 1
        assert result["color_blue"].iloc[0] == 0
        # Row 1: blue
        assert result["color_red"].iloc[1] == 0
        assert result["color_blue"].iloc[1] == 1

    def test_exactly_one_hot_per_row(self):
        df = pd.DataFrame({"animal": ["cat", "dog", "bird", "cat"]})
        result = one_hot_encode(df, "animal")
        dummy_cols = [c for c in result.columns if c.startswith("animal_")]
        for _, row in result[dummy_cols].iterrows():
            assert row.sum() == 1

    def test_raises_for_more_than_50_categories(self):
        categories = [f"cat_{i}" for i in range(51)]
        df = pd.DataFrame({"col": categories})
        with pytest.raises(ValueError, match="exceeds the maximum of 50"):
            one_hot_encode(df, "col")

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="does not exist"):
            one_hot_encode(df, "nonexistent")

    def test_raises_for_numerical_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="numerical"):
            one_hot_encode(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [None, None, None]})
        df["a"] = df["a"].astype(object)
        with pytest.raises(ValueError, match="only missing values"):
            one_hot_encode(df, "a")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        original_cols = list(df.columns)
        one_hot_encode(df, "color")
        assert list(df.columns) == original_cols

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"color": ["red", "blue"], "value": [10, 20]})
        result = one_hot_encode(df, "color")
        assert result["value"].tolist() == [10, 20]

    def test_exactly_50_categories_allowed(self):
        categories = [f"cat_{i}" for i in range(50)]
        df = pd.DataFrame({"col": categories})
        result = one_hot_encode(df, "col")
        assert "col" not in result.columns
        dummy_cols = [c for c in result.columns if c.startswith("col_")]
        assert len(dummy_cols) == 50


class TestStandardScale:
    """Tests for standard_scale function."""

    def test_zero_mean_unit_variance(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = standard_scale(df, "value")
        assert abs(result["value"].mean()) < 1e-10
        assert abs(result["value"].std(ddof=0) - 1.0) < 1e-10

    def test_known_values(self):
        df = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
        result = standard_scale(df, "value")
        # mean=20, std=8.165 (population)
        mean = 20.0
        std = np.std([10.0, 20.0, 30.0])
        expected = [(10 - mean) / std, (20 - mean) / std, (30 - mean) / std]
        for i, exp in enumerate(expected):
            assert abs(result["value"].iloc[i] - exp) < 1e-10

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="does not exist"):
            standard_scale(df, "nonexistent")

    def test_raises_for_categorical_column(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            standard_scale(df, "a")

    def test_raises_for_constant_column(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        with pytest.raises(ValueError, match="zero standard deviation"):
            standard_scale(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            standard_scale(df, "a")

    def test_handles_nan_values(self):
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, 5.0]})
        result = standard_scale(df, "value")
        # NaN should remain NaN
        assert pd.isna(result["value"].iloc[1])
        # Non-NaN values should be scaled
        non_nan = result["value"].dropna()
        assert abs(non_nan.mean()) < 1e-10

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        original = df["value"].tolist()
        standard_scale(df, "value")
        assert df["value"].tolist() == original

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "other": ["a", "b", "c"]})
        result = standard_scale(df, "value")
        assert result["other"].tolist() == ["a", "b", "c"]


class TestMinMaxScale:
    """Tests for min_max_scale function."""

    def test_scales_to_zero_one(self):
        df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = min_max_scale(df, "value")
        assert result["value"].min() == 0.0
        assert result["value"].max() == 1.0

    def test_known_values(self):
        df = pd.DataFrame({"value": [0.0, 50.0, 100.0]})
        result = min_max_scale(df, "value")
        assert result["value"].tolist() == [0.0, 0.5, 1.0]

    def test_all_values_in_range(self):
        df = pd.DataFrame({"value": [3.0, 7.0, 1.0, 9.0, 5.0]})
        result = min_max_scale(df, "value")
        assert all(0.0 <= v <= 1.0 for v in result["value"])

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="does not exist"):
            min_max_scale(df, "nonexistent")

    def test_raises_for_categorical_column(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            min_max_scale(df, "a")

    def test_raises_for_constant_column(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        with pytest.raises(ValueError, match="identical min and max"):
            min_max_scale(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            min_max_scale(df, "a")

    def test_handles_nan_values(self):
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, 5.0]})
        result = min_max_scale(df, "value")
        # NaN should remain NaN
        assert pd.isna(result["value"].iloc[1])
        # Non-NaN values should be in [0, 1]
        non_nan = result["value"].dropna()
        assert non_nan.min() == 0.0
        assert non_nan.max() == 1.0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        original = df["value"].tolist()
        min_max_scale(df, "value")
        assert df["value"].tolist() == original

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "other": ["a", "b", "c"]})
        result = min_max_scale(df, "value")
        assert result["other"].tolist() == ["a", "b", "c"]

    def test_two_distinct_values(self):
        df = pd.DataFrame({"value": [10.0, 20.0]})
        result = min_max_scale(df, "value")
        assert result["value"].iloc[0] == 0.0
        assert result["value"].iloc[1] == 1.0
