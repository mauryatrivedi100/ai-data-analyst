"""Tests for the /summary and /statistics routes."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the backend module is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


@pytest.fixture
def app():
    """Create a Flask app configured for testing."""
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Create a Flask test client."""
    return app.test_client()


@pytest.fixture
def upload_folder(app):
    """Return the upload folder path."""
    return app.config["UPLOAD_FOLDER"]


@pytest.fixture
def sample_csv(upload_folder):
    """Create a sample CSV file in uploads/ for testing."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 25],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 50000.0],
        "city": ["New York", "London", "Paris", "London", "New York"],
        "department": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
    })
    filepath = os.path.join(upload_folder, "test_data.csv")
    df.to_csv(filepath, index=False)
    yield "test_data.csv"
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


@pytest.fixture
def csv_with_missing(upload_folder):
    """Create a CSV file with missing values."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [np.nan, np.nan, np.nan, np.nan],  # All NaN column
        "c": ["x", "y", None, "x"],
    })
    filepath = os.path.join(upload_folder, "missing_data.csv")
    df.to_csv(filepath, index=False)
    yield "missing_data.csv"
    if os.path.exists(filepath):
        os.remove(filepath)


class TestSummaryRoute:
    """Tests for the GET /summary endpoint."""

    def test_summary_returns_correct_structure(self, client, sample_csv):
        """Summary should return all expected fields."""
        resp = client.get(f"/summary?filename={sample_csv}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "row_count" in data
        assert "column_count" in data
        assert "columns" in data
        assert "dtypes" in data
        assert "missing_values" in data
        assert "duplicate_rows" in data
        assert "memory_usage" in data

    def test_summary_correct_counts(self, client, sample_csv):
        """Summary should report correct row and column counts."""
        resp = client.get(f"/summary?filename={sample_csv}")
        data = resp.get_json()
        assert data["row_count"] == 5
        assert data["column_count"] == 4

    def test_summary_columns_list(self, client, sample_csv):
        """Summary should return column names."""
        resp = client.get(f"/summary?filename={sample_csv}")
        data = resp.get_json()
        assert data["columns"] == ["age", "salary", "city", "department"]

    def test_summary_dtypes(self, client, sample_csv):
        """Summary should return dtype for each column."""
        resp = client.get(f"/summary?filename={sample_csv}")
        data = resp.get_json()
        assert "int64" in data["dtypes"]["age"]
        assert "float64" in data["dtypes"]["salary"]
        assert "object" in data["dtypes"]["city"]

    def test_summary_missing_values(self, client, csv_with_missing):
        """Summary should report missing value counts per column."""
        resp = client.get(f"/summary?filename={csv_with_missing}")
        data = resp.get_json()
        assert data["missing_values"]["a"] == 1
        assert data["missing_values"]["b"] == 4
        assert data["missing_values"]["c"] == 1

    def test_summary_duplicate_rows(self, client, sample_csv):
        """Summary should report duplicate row count."""
        resp = client.get(f"/summary?filename={sample_csv}")
        data = resp.get_json()
        # row 0 and row 4 have same age and salary but different city/dept? Let's check
        # age=[25,30,35,40,25], city=["NY","London","Paris","London","NY"], dept=["Sales","Eng","Sales","HR","Eng"]
        # Row 0: 25, 50000, NY, Sales; Row 4: 25, 50000, NY, Engineering — not duplicate
        assert data["duplicate_rows"] == 0

    def test_summary_memory_usage_format(self, client, sample_csv):
        """Memory usage should be a human-readable string."""
        resp = client.get(f"/summary?filename={sample_csv}")
        data = resp.get_json()
        # Should contain a unit like bytes, KB, or MB
        assert any(unit in data["memory_usage"] for unit in ["bytes", "KB", "MB"])

    def test_summary_missing_filename_returns_400(self, client):
        """Should return 400 if filename parameter is missing."""
        resp = client.get("/summary")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_summary_nonexistent_file_returns_404(self, client):
        """Should return 404 for a file that doesn't exist."""
        resp = client.get("/summary?filename=nonexistent.csv")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data


class TestStatisticsRoute:
    """Tests for the GET /statistics endpoint."""

    def test_statistics_returns_numerical_and_categorical(self, client, sample_csv):
        """Statistics should return both numerical and categorical sections."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "numerical" in data
        assert "categorical" in data

    def test_statistics_numerical_fields(self, client, sample_csv):
        """Each numerical column should have mean, median, std, min, max, q1, q2, q3."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        age_stats = data["numerical"]["age"]
        expected_keys = {"mean", "median", "std", "min", "max", "q1", "q2", "q3"}
        assert set(age_stats.keys()) == expected_keys

    def test_statistics_numerical_values_rounded(self, client, sample_csv):
        """Numerical stats should be rounded to 2 decimal places."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        salary_stats = data["numerical"]["salary"]
        # Mean of [50000, 60000, 70000, 80000, 50000] = 62000.0
        assert salary_stats["mean"] == 62000.0
        # Min = 50000, Max = 80000
        assert salary_stats["min"] == 50000.0
        assert salary_stats["max"] == 80000.0

    def test_statistics_categorical_fields(self, client, sample_csv):
        """Each categorical column should have unique_count and top_5."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        city_stats = data["categorical"]["city"]
        assert "unique_count" in city_stats
        assert "top_5" in city_stats

    def test_statistics_categorical_unique_count(self, client, sample_csv):
        """Unique count should match the number of distinct values."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        # city values: NY, London, Paris, London, NY -> 3 unique
        assert data["categorical"]["city"]["unique_count"] == 3

    def test_statistics_categorical_top_5_structure(self, client, sample_csv):
        """Top 5 entries should have value and count fields."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        top_5 = data["categorical"]["city"]["top_5"]
        assert len(top_5) <= 5
        for entry in top_5:
            assert "value" in entry
            assert "count" in entry

    def test_statistics_categorical_top_5_sorted_descending(self, client, sample_csv):
        """Top 5 should be sorted by count descending."""
        resp = client.get(f"/statistics?filename={sample_csv}")
        data = resp.get_json()
        top_5 = data["categorical"]["city"]["top_5"]
        counts = [entry["count"] for entry in top_5]
        assert counts == sorted(counts, reverse=True)

    def test_statistics_all_nan_column_returns_none(self, client, csv_with_missing):
        """Column with all NaN should return None for all numerical stats."""
        resp = client.get(f"/statistics?filename={csv_with_missing}")
        data = resp.get_json()
        b_stats = data["numerical"]["b"]
        for key in ["mean", "median", "std", "min", "max", "q1", "q2", "q3"]:
            assert b_stats[key] is None

    def test_statistics_missing_filename_returns_400(self, client):
        """Should return 400 if filename parameter is missing."""
        resp = client.get("/statistics")
        assert resp.status_code == 400

    def test_statistics_nonexistent_file_returns_404(self, client):
        """Should return 404 for a file that doesn't exist."""
        resp = client.get("/statistics?filename=nonexistent.csv")
        assert resp.status_code == 404
