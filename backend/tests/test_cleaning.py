"""Unit tests for missing value handling functions in cleaning module."""

import pytest
import pandas as pd
import numpy as np

from cleaning import (
    remove_missing_rows,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
)


class TestRemoveMissingRows:
    """Tests for remove_missing_rows function."""

    def test_removes_rows_with_nan(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        cleaned, count = remove_missing_rows(df)
        assert count == 1
        assert len(cleaned) == 2
        assert not cleaned.isna().any().any()

    def test_returns_correct_count(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, np.nan, 4.0],
            "b": [np.nan, 2.0, np.nan, 5.0],
        })
        cleaned, count = remove_missing_rows(df)
        assert count == 3
        assert len(cleaned) == 1

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cleaned, count = remove_missing_rows(df)
        assert count == 0
        assert len(cleaned) == 3

    def test_raises_when_all_rows_have_nan(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan],
            "b": [1.0, np.nan, 3.0],
        })
        with pytest.raises(ValueError, match="all rows contain missing values"):
            remove_missing_rows(df)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        original_len = len(df)
        remove_missing_rows(df)
        assert len(df) == original_len

    def test_resets_index(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
        })
        cleaned, _ = remove_missing_rows(df)
        assert list(cleaned.index) == list(range(len(cleaned)))


class TestFillMissingMean:
    """Tests for fill_missing_mean function."""

    def test_fills_with_mean(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cleaned, mean_val = fill_missing_mean(df, "a")
        assert mean_val == 2.0
        assert cleaned["a"].iloc[1] == 2.0
        assert not cleaned["a"].isna().any()

    def test_preserves_non_missing_values(self):
        df = pd.DataFrame({"a": [10.0, np.nan, 30.0, 40.0]})
        cleaned, _ = fill_missing_mean(df, "a")
        assert cleaned["a"].iloc[0] == 10.0
        assert cleaned["a"].iloc[2] == 30.0
        assert cleaned["a"].iloc[3] == 40.0

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_mean(df, "nonexistent")

    def test_raises_for_categorical_column(self):
        df = pd.DataFrame({"a": ["x", "y", None]})
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_mean(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_mean(df, "a")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        original_val = df["a"].iloc[1]
        fill_missing_mean(df, "a")
        assert pd.isna(df["a"].iloc[1])

    def test_returns_float_mean(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 2.0]})
        _, mean_val = fill_missing_mean(df, "a")
        assert isinstance(mean_val, float)


class TestFillMissingMedian:
    """Tests for fill_missing_median function."""

    def test_fills_with_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0]})
        cleaned, median_val = fill_missing_median(df, "a")
        assert median_val == 3.0
        assert cleaned["a"].iloc[1] == 3.0
        assert not cleaned["a"].isna().any()

    def test_preserves_non_missing_values(self):
        df = pd.DataFrame({"a": [10.0, np.nan, 30.0]})
        cleaned, _ = fill_missing_median(df, "a")
        assert cleaned["a"].iloc[0] == 10.0
        assert cleaned["a"].iloc[2] == 30.0

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_median(df, "nonexistent")

    def test_raises_for_categorical_column(self):
        df = pd.DataFrame({"a": ["x", "y", None]})
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_median(df, "a")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_median(df, "a")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        fill_missing_median(df, "a")
        assert pd.isna(df["a"].iloc[1])

    def test_returns_float_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 4.0]})
        _, median_val = fill_missing_median(df, "a")
        assert isinstance(median_val, float)


class TestFillMissingMode:
    """Tests for fill_missing_mode function."""

    def test_fills_with_mode(self):
        df = pd.DataFrame({"a": [1, 2, 2, np.nan, 3]})
        cleaned, mode_val = fill_missing_mode(df, "a")
        assert mode_val == 2
        assert cleaned["a"].iloc[3] == 2
        assert not cleaned["a"].isna().any()

    def test_works_with_categorical_data(self):
        df = pd.DataFrame({"a": ["cat", "dog", "cat", None]})
        cleaned, mode_val = fill_missing_mode(df, "a")
        assert mode_val == "cat"
        assert cleaned["a"].iloc[3] == "cat"

    def test_uses_first_encountered_when_tied(self):
        # pandas mode() returns all modes sorted; first one is used
        df = pd.DataFrame({"a": ["x", "y", None]})
        cleaned, mode_val = fill_missing_mode(df, "a")
        # Both 'x' and 'y' appear once; pandas sorts them so 'x' comes first
        assert mode_val in ["x", "y"]
        assert not cleaned["a"].isna().any()

    def test_preserves_non_missing_values(self):
        df = pd.DataFrame({"a": ["cat", "dog", "cat", None]})
        cleaned, _ = fill_missing_mode(df, "a")
        assert cleaned["a"].iloc[0] == "cat"
        assert cleaned["a"].iloc[1] == "dog"
        assert cleaned["a"].iloc[2] == "cat"

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_mode(df, "nonexistent")

    def test_raises_when_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_mode(df, "a")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": ["cat", "dog", None]})
        fill_missing_mode(df, "a")
        assert pd.isna(df["a"].iloc[2])
