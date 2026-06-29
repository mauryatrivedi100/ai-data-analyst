"""Tests for the GET /visualizations route handler."""

import io
import os

import pytest

from app import create_app


@pytest.fixture
def app():
    """Create a test Flask application."""
    app = create_app()
    app.config["TESTING"] = True
    yield app
    # Clean up any uploaded files after tests
    upload_folder = app.config["UPLOAD_FOLDER"]
    for f in os.listdir(upload_folder):
        if f != ".gitkeep":
            os.remove(os.path.join(upload_folder, f))


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def uploaded_dataset(client):
    """Upload a test dataset and return its filename."""
    csv_content = (
        "name,age,salary,city,score\n"
        "Alice,30,50000,NYC,85.5\n"
        "Bob,25,60000,LA,90.2\n"
        "Charlie,35,70000,Chicago,78.1\n"
        "Diana,28,55000,NYC,92.0\n"
        "Eve,32,65000,LA,88.7\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode()), "viz_test.csv")}
    response = client.post("/upload", content_type="multipart/form-data", data=data)
    assert response.status_code == 200
    return response.get_json()["filename"]


class TestVisualizationRouteValidation:
    """Tests for parameter validation on GET /visualizations."""

    def test_missing_filename(self, client):
        """Returns 400 when filename is not provided."""
        response = client.get("/visualizations?type=histogram&x=age")
        assert response.status_code == 400
        json_data = response.get_json()
        assert "filename" in json_data["error"].lower()

    def test_missing_type(self, client, uploaded_dataset):
        """Returns 400 when chart type is not provided."""
        response = client.get(f"/visualizations?filename={uploaded_dataset}&x=age")
        assert response.status_code == 400
        json_data = response.get_json()
        assert "type" in json_data["error"].lower()

    def test_unknown_chart_type(self, client, uploaded_dataset):
        """Returns 400 for an unsupported chart type."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=unknown&x=age"
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert "unknown" in json_data["error"].lower()

    def test_file_not_found(self, client):
        """Returns 404 when the dataset file does not exist."""
        response = client.get(
            "/visualizations?filename=nonexistent.csv&type=histogram&x=age"
        )
        assert response.status_code == 404

    def test_missing_x_for_histogram(self, client, uploaded_dataset):
        """Returns 400 when x is missing for histogram."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=histogram"
        )
        assert response.status_code == 400
        assert "'x'" in response.get_json()["error"]

    def test_missing_y_for_scatter(self, client, uploaded_dataset):
        """Returns 400 when y is missing for scatter."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=scatter&x=age"
        )
        assert response.status_code == 400
        assert "'y'" in response.get_json()["error"]

    def test_missing_y_for_line(self, client, uploaded_dataset):
        """Returns 400 when y is missing for line chart."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=line&x=age"
        )
        assert response.status_code == 400
        assert "'y'" in response.get_json()["error"]

    def test_missing_y_for_bar(self, client, uploaded_dataset):
        """Returns 400 when y is missing for bar chart."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=bar&x=city"
        )
        assert response.status_code == 400
        assert "'y'" in response.get_json()["error"]


class TestVisualizationRouteSuccess:
    """Tests for successful chart data generation."""

    def test_histogram_returns_data(self, client, uploaded_dataset):
        """Histogram returns bin data for a numerical column."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=histogram&x=age"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "histogram"
        assert isinstance(json_data["data"], list)
        assert len(json_data["data"]) > 0
        # Each bin has bin_start, bin_end, count
        assert "bin_start" in json_data["data"][0]
        assert "bin_end" in json_data["data"][0]
        assert "count" in json_data["data"][0]

    def test_scatter_returns_data(self, client, uploaded_dataset):
        """Scatter plot returns x/y data points."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=scatter&x=age&y=salary"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "scatter"
        assert isinstance(json_data["data"], list)
        assert len(json_data["data"]) == 5
        assert "x" in json_data["data"][0]
        assert "y" in json_data["data"][0]

    def test_line_returns_sorted_data(self, client, uploaded_dataset):
        """Line chart returns sorted x/y data points."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=line&x=age&y=score"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "line"
        assert isinstance(json_data["data"], list)
        # Verify sorted by x
        x_values = [p["x"] for p in json_data["data"]]
        assert x_values == sorted(x_values)

    def test_bar_returns_data(self, client, uploaded_dataset):
        """Bar chart returns category/value data."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=bar&x=city&y=salary"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "bar"
        assert isinstance(json_data["data"], list)
        assert "category" in json_data["data"][0]
        assert "value" in json_data["data"][0]

    def test_pie_returns_data(self, client, uploaded_dataset):
        """Pie chart returns name/value data."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=pie&x=city"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "pie"
        assert isinstance(json_data["data"], list)
        assert "name" in json_data["data"][0]
        assert "value" in json_data["data"][0]

    def test_box_returns_data(self, client, uploaded_dataset):
        """Box plot returns statistics dict."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=box&x=age"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "box"
        data = json_data["data"]
        assert "min" in data
        assert "q1" in data
        assert "median" in data
        assert "q3" in data
        assert "max" in data
        assert "outliers" in data

    def test_heatmap_returns_data(self, client, uploaded_dataset):
        """Heatmap returns correlation matrix without requiring x/y."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=heatmap"
        )
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "heatmap"
        data = json_data["data"]
        assert "columns" in data
        assert "matrix" in data
        assert isinstance(data["matrix"], list)


class TestVisualizationRouteColumnErrors:
    """Tests for column type incompatibility errors."""

    def test_histogram_non_numerical_column(self, client, uploaded_dataset):
        """Returns 400 when histogram is requested for a categorical column."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=histogram&x=name"
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert "not numerical" in json_data["error"].lower()

    def test_scatter_non_numerical_column(self, client, uploaded_dataset):
        """Returns 400 when scatter uses a non-numerical column."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=scatter&x=name&y=age"
        )
        assert response.status_code == 400

    def test_box_non_numerical_column(self, client, uploaded_dataset):
        """Returns 400 when box plot is requested for a categorical column."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=box&x=city"
        )
        assert response.status_code == 400

    def test_nonexistent_column(self, client, uploaded_dataset):
        """Returns 400 when a non-existent column is specified."""
        response = client.get(
            f"/visualizations?filename={uploaded_dataset}&type=histogram&x=nonexistent"
        )
        assert response.status_code == 400
        assert "does not exist" in response.get_json()["error"].lower()
