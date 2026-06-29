"""Unit tests for duplicate and outlier handling functions in cleaning module."""

import pytest
import pandas as pd
import numpy as np

from cleaning import remove_duplicates, detect_outliers, remove_outliers


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""

    def test_removes_duplicate_rows(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": [10, 20, 10, 30],
        })
        cleaned, count = remove_duplicates(df)
        assert count == 1
        assert len(cleaned) == 3

    def test_keeps_first_occurrence(self):
        df = pd.DataFrame({
            "a": [1, 2, 1],
            "b": [10, 20, 10],
        })
        cleaned, _ = remove_duplicates(df)
        # First occurrence at index 0 should be kept
        assert cleaned.iloc[0]["a"] == 1
        assert cleaned.iloc[0]["b"] == 10

    def test_no_duplicates_returns_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cleaned, count = remove_duplicates(df)
        assert count == 0
        assert len(cleaned) == 3

    def test_all_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 1, 1],
            "b": [2, 2, 2],
        })
        cleaned, count = remove_duplicates(df)
        assert count == 2
        assert len(cleaned) == 1

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        original_len = len(df)
        remove_duplicates(df)
        assert len(df) == original_len

    def test_resets_index(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 3, 2],
            "b": [10, 20, 10, 30, 20],
        })
        cleaned, _ = remove_duplicates(df)
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        cleaned, count = remove_duplicates(df)
        assert count == 0
        assert len(cleaned) == 0

    def test_with_nan_values(self):
        # NaN != NaN in pandas duplicated by default
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [2.0, 3.0, 3.0]})
        cleaned, count = remove_duplicates(df)
        # pandas treats NaN as equal in duplicated()
        assert count == 1
        assert len(cleaned) == 2


class TestDetectOutliers:
    """Tests for detect_outliers function."""

    def test_detects_outliers_correctly(self):
        # IQR method: Q1=2, Q3=4, IQR=2, lower=-1, upper=7
        data = [1, 2, 3, 4, 5, 100]
        df = pd.DataFrame({"val": data})
        count, lower, upper, indices = detect_outliers(df, "val")
        assert count == 1
        assert 100 in [df["val"].iloc[i] for i in indices]

    def test_no_outliers(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        count, lower, upper, indices = detect_outliers(df, "val")
        assert count == 0
        assert indices == []

    def test_returns_correct_bounds(self):
        # Known dataset: [1, 2, 3, 4, 5]
        # Q1=2, Q3=4, IQR=2, lower=2-3=-1, upper=4+3=7
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        count, lower, upper, indices = detect_outliers(df, "val")
        assert lower == pytest.approx(-1.0)
        assert upper == pytest.approx(7.0)

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="does not exist"):
            detect_outliers(df, "nonexistent")

    def test_raises_for_non_numerical_column(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            detect_outliers(df, "a")

    def test_raises_for_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            detect_outliers(df, "a")

    def test_ignores_nan_in_computation(self):
        # NaN values should not be counted as outliers
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0]})
        count, lower, upper, indices = detect_outliers(df, "val")
        # No values are outside bounds so no outliers
        assert count == 0

    def test_both_lower_and_upper_outliers(self):
        df = pd.DataFrame({"val": [-100, 1, 2, 3, 4, 5, 100]})
        count, lower, upper, indices = detect_outliers(df, "val")
        assert count == 2

    def test_constant_column_no_outliers(self):
        # If all values are the same, IQR=0, bounds = [val, val], no outliers
        df = pd.DataFrame({"val": [5, 5, 5, 5, 5]})
        count, lower, upper, indices = detect_outliers(df, "val")
        assert count == 0
        assert lower == 5.0
        assert upper == 5.0


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_removes_outlier_rows(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        cleaned, count = remove_outliers(df, "val")
        assert 100 not in cleaned["val"].values
        assert count >= 1

    def test_no_outliers_returns_zero(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        cleaned, count = remove_outliers(df, "val")
        assert count == 0
        assert len(cleaned) == 5

    def test_preserves_nan_rows(self):
        # NaN rows should NOT be removed as outliers
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0]})
        cleaned, count = remove_outliers(df, "val")
        assert count == 0
        # NaN row is preserved
        assert len(cleaned) == 6

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        original_len = len(df)
        remove_outliers(df, "val")
        assert len(df) == original_len

    def test_resets_index(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100]})
        cleaned, _ = remove_outliers(df, "val")
        assert list(cleaned.index) == list(range(len(cleaned)))

    def test_raises_for_nonexistent_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="does not exist"):
            remove_outliers(df, "nonexistent")

    def test_raises_for_non_numerical_column(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            remove_outliers(df, "a")

    def test_raises_for_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only missing values"):
            remove_outliers(df, "a")

    def test_remaining_values_within_bounds(self):
        df = pd.DataFrame({"val": [-100, 1, 2, 3, 4, 5, 100]})
        cleaned, count = remove_outliers(df, "val")
        _, lower, upper, _ = detect_outliers(df, "val")
        # All non-NaN remaining values should be within bounds
        non_nan = cleaned["val"].dropna()
        assert (non_nan >= lower).all()
        assert (non_nan <= upper).all()
