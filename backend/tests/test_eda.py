"""Unit tests for EDA computation functions."""

import numpy as np
import pandas as pd
import pytest

from backend.eda import (
    compute_bar,
    compute_box_plot,
    compute_correlation_heatmap,
    compute_histogram,
    compute_line,
    compute_pie,
    compute_scatter,
)


class TestComputeHistogram:
    """Tests for compute_histogram."""

    def test_basic_histogram(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = compute_histogram(df, "a", bins=5)
        assert len(result) == 5
        assert all("bin_start" in r and "bin_end" in r and "count" in r for r in result)
        assert sum(r["count"] for r in result) == 10

    def test_bins_are_contiguous(self):
        df = pd.DataFrame({"a": range(100)})
        result = compute_histogram(df, "a", bins=10)
        for i in range(1, len(result)):
            assert result[i]["bin_start"] == pytest.approx(result[i - 1]["bin_end"])

    def test_nonexistent_column_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_histogram(df, "nonexistent")

    def test_categorical_column_raises(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_histogram(df, "a")

    def test_all_nan_returns_empty(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        result = compute_histogram(df, "a")
        assert result == []

    def test_nan_values_dropped(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = compute_histogram(df, "a", bins=4)
        assert sum(r["count"] for r in result) == 4

    def test_single_value(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5]})
        result = compute_histogram(df, "a", bins=5)
        assert sum(r["count"] for r in result) == 4


class TestComputeScatter:
    """Tests for compute_scatter."""

    def test_basic_scatter(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = compute_scatter(df, "x", "y")
        assert len(result) == 3
        assert result[0] == {"x": 1.0, "y": 4.0}
        assert result[2] == {"x": 3.0, "y": 6.0}

    def test_nan_dropped(self):
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [4, 5, np.nan]})
        result = compute_scatter(df, "x", "y")
        assert len(result) == 1
        assert result[0] == {"x": 1.0, "y": 4.0}

    def test_nonexistent_x_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_scatter(df, "bad", "y")

    def test_nonexistent_y_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_scatter(df, "x", "bad")

    def test_non_numeric_x_raises(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_scatter(df, "x", "y")

    def test_non_numeric_y_raises(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_scatter(df, "x", "y")

    def test_empty_after_dropna(self):
        df = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
        result = compute_scatter(df, "x", "y")
        assert result == []


class TestComputeLine:
    """Tests for compute_line."""

    def test_sorted_by_x(self):
        df = pd.DataFrame({"x": [3, 1, 2], "y": [30, 10, 20]})
        result = compute_line(df, "x", "y")
        assert result == [
            {"x": 1.0, "y": 10.0},
            {"x": 2.0, "y": 20.0},
            {"x": 3.0, "y": 30.0},
        ]

    def test_nan_dropped(self):
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [10, 20, np.nan]})
        result = compute_line(df, "x", "y")
        assert len(result) == 1

    def test_nonexistent_column_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_line(df, "bad", "y")

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a"], "y": [1]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_line(df, "x", "y")


class TestComputeBar:
    """Tests for compute_bar."""

    def test_basic_bar(self):
        df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [10, 20, 30, 40]})
        result = compute_bar(df, "cat", "val")
        result_dict = {r["category"]: r["value"] for r in result}
        assert result_dict["A"] == pytest.approx(15.0)
        assert result_dict["B"] == pytest.approx(35.0)

    def test_nonexistent_x_raises(self):
        df = pd.DataFrame({"cat": ["A"], "val": [1]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_bar(df, "bad", "val")

    def test_nonexistent_y_raises(self):
        df = pd.DataFrame({"cat": ["A"], "val": [1]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_bar(df, "cat", "bad")

    def test_non_numeric_y_raises(self):
        df = pd.DataFrame({"cat": ["A"], "val": ["x"]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_bar(df, "cat", "val")

    def test_nan_dropped(self):
        df = pd.DataFrame({"cat": ["A", "A", None], "val": [10, np.nan, 30]})
        result = compute_bar(df, "cat", "val")
        # Only row 0 survives (A, 10) since row 1 has NaN val and row 2 has None cat
        assert len(result) == 1
        assert result[0]["value"] == pytest.approx(10.0)

    def test_empty_after_dropna(self):
        df = pd.DataFrame({"cat": [None, None], "val": [np.nan, np.nan]})
        result = compute_bar(df, "cat", "val")
        assert result == []


class TestComputePie:
    """Tests for compute_pie."""

    def test_basic_pie(self):
        df = pd.DataFrame({"cat": ["A", "A", "B", "C", "C", "C"]})
        result = compute_pie(df, "cat")
        result_dict = {r["name"]: r["value"] for r in result}
        assert result_dict["A"] == 2
        assert result_dict["B"] == 1
        assert result_dict["C"] == 3

    def test_sum_equals_total_non_null(self):
        df = pd.DataFrame({"cat": ["A", "B", "B", None, "C"]})
        result = compute_pie(df, "cat")
        assert sum(r["value"] for r in result) == 4  # Excludes None

    def test_nonexistent_column_raises(self):
        df = pd.DataFrame({"cat": ["A"]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_pie(df, "bad")

    def test_all_nan_returns_empty(self):
        df = pd.DataFrame({"cat": [None, None, None]})
        result = compute_pie(df, "cat")
        assert result == []

    def test_numerical_column_works(self):
        df = pd.DataFrame({"val": [1, 1, 2, 3, 3, 3]})
        result = compute_pie(df, "val")
        result_dict = {r["name"]: r["value"] for r in result}
        assert result_dict["3"] == 3
        assert result_dict["1"] == 2
        assert result_dict["2"] == 1


class TestComputeBoxPlot:
    """Tests for compute_box_plot."""

    def test_basic_box_plot(self):
        df = pd.DataFrame({"a": list(range(1, 11))})
        result = compute_box_plot(df, "a")
        assert "min" in result
        assert "q1" in result
        assert "median" in result
        assert "q3" in result
        assert "max" in result
        assert "outliers" in result
        assert result["median"] == pytest.approx(5.5)

    def test_outliers_detected(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
        result = compute_box_plot(df, "a")
        assert 100.0 in result["outliers"]

    def test_no_outliers(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = compute_box_plot(df, "a")
        assert result["outliers"] == []

    def test_nonexistent_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="does not exist"):
            compute_box_plot(df, "bad")

    def test_categorical_column_raises(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="not numerical"):
            compute_box_plot(df, "a")

    def test_all_nan_returns_none_values(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        result = compute_box_plot(df, "a")
        assert result["min"] is None
        assert result["q1"] is None
        assert result["outliers"] == []

    def test_all_same_values(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        result = compute_box_plot(df, "a")
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["median"] == 5.0
        assert result["outliers"] == []


class TestComputeCorrelationHeatmap:
    """Tests for compute_correlation_heatmap."""

    def test_basic_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        result = compute_correlation_heatmap(df)
        assert result["columns"] == ["a", "b"]
        assert len(result["matrix"]) == 2
        # Perfect negative correlation
        assert result["matrix"][0][1] == pytest.approx(-1.0)
        # Diagonal is 1
        assert result["matrix"][0][0] == pytest.approx(1.0)

    def test_symmetric_matrix(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = compute_correlation_heatmap(df)
        matrix = result["matrix"]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                assert matrix[i][j] == pytest.approx(matrix[j][i])

    def test_fewer_than_2_numerical_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "cat": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="at least 2 numerical columns"):
            compute_correlation_heatmap(df)

    def test_only_categorical_raises(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["m", "n"]})
        with pytest.raises(ValueError, match="at least 2 numerical columns"):
            compute_correlation_heatmap(df)

    def test_ignores_categorical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "cat": ["x", "y", "z"]})
        result = compute_correlation_heatmap(df)
        assert "cat" not in result["columns"]
        assert len(result["columns"]) == 2

    def test_values_in_range(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {"a": np.random.randn(50), "b": np.random.randn(50), "c": np.random.randn(50)}
        )
        result = compute_correlation_heatmap(df)
        for row in result["matrix"]:
            for val in row:
                assert -1.0 <= val <= 1.0
