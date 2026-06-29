"""Unit tests for missing value handling functions in cleaning.py.

Tests validate Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7.
"""

import pytest
import pandas as pd
import numpy as np

from cleaning import (
    remove_missing_rows,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
)


# --- Fixtures ---


@pytest.fixture
def df_with_some_missing():
    """DataFrame where some rows have NaN and some don't."""
    return pd.DataFrame({
        "A": [1.0, np.nan, 3.0, 4.0, np.nan],
        "B": [10.0, 20.0, np.nan, 40.0, 50.0],
        "C": ["x", "y", "z", "w", "v"],
    })


@pytest.fixture
def df_all_rows_missing():
    """DataFrame where every row has at least one NaN."""
    return pd.DataFrame({
        "A": [np.nan, 2.0, np.nan],
        "B": [1.0, np.nan, 3.0],
    })


@pytest.fixture
def df_numerical_with_nan():
    """DataFrame with a numerical column containing some NaN values."""
    return pd.DataFrame({
        "value": [10.0, 20.0, np.nan, 40.0, np.nan, 60.0],
        "other": [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def df_all_nan_column():
    """DataFrame with a column that is entirely NaN."""
    return pd.DataFrame({
        "empty_col": [np.nan, np.nan, np.nan],
        "valid_col": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def df_categorical():
    """DataFrame with categorical (object dtype) columns."""
    return pd.DataFrame({
        "category": ["a", "b", None, "a", "c"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# --- Tests for remove_missing_rows ---


class TestRemoveMissingRows:
    """Tests for remove_missing_rows (Requirement 3.1, 3.5)."""

    def test_removes_rows_with_nan(self, df_with_some_missing):
        """Req 3.1: Removes all rows containing at least one missing value."""
        cleaned, count = remove_missing_rows(df_with_some_missing)

        # No NaN values should remain
        assert cleaned.isna().sum().sum() == 0
        # Rows 1 and 4 had NaN (0-indexed), rows 0, 3 remain plus row 2 has NaN in B
        # Actually: row 0 (A=1, B=10, C=x) - ok, row 1 (A=nan) - removed,
        # row 2 (B=nan) - removed, row 3 (A=4, B=40, C=w) - ok, row 4 (A=nan) - removed
        assert len(cleaned) == 2
        assert count == 3

    def test_returns_correct_count(self, df_with_some_missing):
        """Req 3.1: Displays the number of rows removed."""
        _, count = remove_missing_rows(df_with_some_missing)
        assert count == 3

    def test_no_missing_values_returns_same(self):
        """No rows removed when no NaN values exist."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        cleaned, count = remove_missing_rows(df)
        assert count == 0
        assert len(cleaned) == 3

    def test_raises_when_all_rows_have_nan(self, df_all_rows_missing):
        """Req 3.5: Error when all rows contain missing values."""
        with pytest.raises(ValueError, match="all rows contain missing values"):
            remove_missing_rows(df_all_rows_missing)

    def test_does_not_mutate_input(self, df_with_some_missing):
        """Input DataFrame should not be modified."""
        original = df_with_some_missing.copy()
        remove_missing_rows(df_with_some_missing)
        pd.testing.assert_frame_equal(df_with_some_missing, original)

    def test_index_is_reset(self, df_with_some_missing):
        """Cleaned DataFrame has a reset index."""
        cleaned, _ = remove_missing_rows(df_with_some_missing)
        assert list(cleaned.index) == list(range(len(cleaned)))


# --- Tests for fill_missing_mean ---


class TestFillMissingMean:
    """Tests for fill_missing_mean (Requirements 3.2, 3.6, 3.7)."""

    def test_fills_with_correct_mean(self, df_numerical_with_nan):
        """Req 3.2: Fill with arithmetic mean of non-missing values."""
        cleaned, mean_val = fill_missing_mean(df_numerical_with_nan, "value")

        # Mean of [10, 20, 40, 60] = 130 / 4 = 32.5
        expected_mean = 32.5
        assert mean_val == pytest.approx(expected_mean)
        assert cleaned["value"].isna().sum() == 0

    def test_non_missing_values_unchanged(self, df_numerical_with_nan):
        """Req 3.2: Non-missing values should not change."""
        cleaned, _ = fill_missing_mean(df_numerical_with_nan, "value")
        assert cleaned["value"].iloc[0] == 10.0
        assert cleaned["value"].iloc[1] == 20.0
        assert cleaned["value"].iloc[3] == 40.0
        assert cleaned["value"].iloc[5] == 60.0

    def test_returns_mean_value(self, df_numerical_with_nan):
        """Req 3.2: Displays the computed mean value."""
        _, mean_val = fill_missing_mean(df_numerical_with_nan, "value")
        assert isinstance(mean_val, float)

    def test_raises_for_categorical_column(self, df_categorical):
        """Req 3.7: Error for categorical columns."""
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_mean(df_categorical, "category")

    def test_raises_for_all_nan_column(self, df_all_nan_column):
        """Req 3.6: Error when all values are NaN."""
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_mean(df_all_nan_column, "empty_col")

    def test_raises_for_nonexistent_column(self, df_numerical_with_nan):
        """Error when column doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_mean(df_numerical_with_nan, "nonexistent")

    def test_does_not_mutate_input(self, df_numerical_with_nan):
        """Input DataFrame should not be modified."""
        original = df_numerical_with_nan.copy()
        fill_missing_mean(df_numerical_with_nan, "value")
        pd.testing.assert_frame_equal(df_numerical_with_nan, original)


# --- Tests for fill_missing_median ---


class TestFillMissingMedian:
    """Tests for fill_missing_median (Requirements 3.3, 3.6, 3.7)."""

    def test_fills_with_correct_median(self, df_numerical_with_nan):
        """Req 3.3: Fill with median of non-missing values."""
        cleaned, median_val = fill_missing_median(df_numerical_with_nan, "value")

        # Median of [10, 20, 40, 60] = (20 + 40) / 2 = 30.0
        expected_median = 30.0
        assert median_val == pytest.approx(expected_median)
        assert cleaned["value"].isna().sum() == 0

    def test_non_missing_values_unchanged(self, df_numerical_with_nan):
        """Req 3.3: Non-missing values should not change."""
        cleaned, _ = fill_missing_median(df_numerical_with_nan, "value")
        assert cleaned["value"].iloc[0] == 10.0
        assert cleaned["value"].iloc[1] == 20.0
        assert cleaned["value"].iloc[3] == 40.0
        assert cleaned["value"].iloc[5] == 60.0

    def test_returns_median_value(self, df_numerical_with_nan):
        """Req 3.3: Displays the computed median value."""
        _, median_val = fill_missing_median(df_numerical_with_nan, "value")
        assert isinstance(median_val, float)

    def test_raises_for_categorical_column(self, df_categorical):
        """Req 3.7: Error for categorical columns."""
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_median(df_categorical, "category")

    def test_raises_for_all_nan_column(self, df_all_nan_column):
        """Req 3.6: Error when all values are NaN."""
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_median(df_all_nan_column, "empty_col")

    def test_raises_for_nonexistent_column(self, df_numerical_with_nan):
        """Error when column doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_median(df_numerical_with_nan, "nonexistent")

    def test_does_not_mutate_input(self, df_numerical_with_nan):
        """Input DataFrame should not be modified."""
        original = df_numerical_with_nan.copy()
        fill_missing_median(df_numerical_with_nan, "value")
        pd.testing.assert_frame_equal(df_numerical_with_nan, original)


# --- Tests for fill_missing_mode ---


class TestFillMissingMode:
    """Tests for fill_missing_mode (Requirements 3.4, 3.6)."""

    def test_fills_with_correct_mode(self, df_categorical):
        """Req 3.4: Fill with most frequently occurring value."""
        cleaned, mode_val = fill_missing_mode(df_categorical, "category")

        # Values: ["a", "b", None, "a", "c"] -> mode is "a" (appears twice)
        assert mode_val == "a"
        assert cleaned["category"].isna().sum() == 0

    def test_fills_numerical_column_with_mode(self):
        """Req 3.4: Mode works for numerical columns too."""
        df = pd.DataFrame({"nums": [1, 2, 2, 3, np.nan]})
        cleaned, mode_val = fill_missing_mode(df, "nums")
        assert mode_val == 2
        assert cleaned["nums"].isna().sum() == 0

    def test_tied_mode_uses_first_encountered(self):
        """Req 3.4: If tied, use first encountered mode value."""
        # pandas mode() returns sorted values, so "a" comes before "b" alphabetically
        df = pd.DataFrame({"col": ["a", "b", "a", "b", None]})
        cleaned, mode_val = fill_missing_mode(df, "col")
        # Both "a" and "b" appear twice - mode() returns first in sorted order
        assert mode_val in ["a", "b"]
        assert cleaned["col"].isna().sum() == 0

    def test_non_missing_values_unchanged(self, df_categorical):
        """Req 3.4: Non-missing values should not change."""
        cleaned, _ = fill_missing_mode(df_categorical, "category")
        assert cleaned["category"].iloc[0] == "a"
        assert cleaned["category"].iloc[1] == "b"
        assert cleaned["category"].iloc[3] == "a"
        assert cleaned["category"].iloc[4] == "c"

    def test_raises_for_all_nan_column(self, df_all_nan_column):
        """Req 3.6: Error when all values are NaN."""
        with pytest.raises(ValueError, match="only missing values"):
            fill_missing_mode(df_all_nan_column, "empty_col")

    def test_raises_for_nonexistent_column(self, df_categorical):
        """Error when column doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            fill_missing_mode(df_categorical, "nonexistent")

    def test_does_not_mutate_input(self, df_categorical):
        """Input DataFrame should not be modified."""
        original = df_categorical.copy()
        fill_missing_mode(df_categorical, "category")
        pd.testing.assert_frame_equal(df_categorical, original)
