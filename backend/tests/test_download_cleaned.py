"""Tests for the /download-cleaned route handler."""

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


def _upload_csv(client, csv_content, filename="test_data.csv"):
    """Helper to upload a CSV file and return the response."""
    data = {"file": (io.BytesIO(csv_content.encode()), filename)}
    return client.post("/upload", content_type="multipart/form-data", data=data)


class TestDownloadCleanedRoute:
    """Tests for GET /download-cleaned endpoint."""

    def test_download_cleaned_returns_csv(self, client):
        """Successful download returns CSV content with correct headers."""
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        _upload_csv(client, csv_content, "mydata.csv")

        response = client.get("/download-cleaned?filename=mydata.csv")

        assert response.status_code == 200
        assert response.content_type == "text/csv; charset=utf-8"
        assert 'attachment; filename="cleaned_mydata.csv"' in response.headers["Content-Disposition"]

    def test_download_cleaned_csv_content_has_headers_and_data(self, client):
        """Downloaded CSV has column headers and data rows."""
        csv_content = "name,age\nAlice,30\nBob,25\n"
        _upload_csv(client, csv_content, "sample.csv")

        response = client.get("/download-cleaned?filename=sample.csv")

        csv_text = response.data.decode("utf-8")
        lines = csv_text.strip().splitlines()
        assert lines[0] == "name,age"
        assert len(lines) == 3  # header + 2 data rows

    def test_download_cleaned_missing_filename_returns_400(self, client):
        """Returns 400 when filename parameter is missing."""
        response = client.get("/download-cleaned")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data
        assert json_data["code"] == "VALIDATION_ERROR"

    def test_download_cleaned_file_not_found_returns_404(self, client):
        """Returns 404 when the dataset file does not exist."""
        response = client.get("/download-cleaned?filename=nonexistent.csv")

        assert response.status_code == 404
        json_data = response.get_json()
        assert "error" in json_data
        assert json_data["code"] == "NOT_FOUND"

    def test_download_cleaned_filename_format(self, client):
        """Download filename is cleaned_<original_name>.csv (without original extension)."""
        csv_content = "x,y\n1,2\n"
        _upload_csv(client, csv_content, "my_dataset.csv")

        response = client.get("/download-cleaned?filename=my_dataset.csv")

        assert response.status_code == 200
        assert 'filename="cleaned_my_dataset.csv"' in response.headers["Content-Disposition"]

    def test_download_cleaned_empty_dataframe_returns_headers_only(self, client, app):
        """Empty DataFrame (0 rows) still returns CSV with column headers."""
        import pandas as pd
        from utils import save_dataset

        # Manually create an empty DataFrame with headers and save it
        df = pd.DataFrame(columns=["col_a", "col_b", "col_c"])
        with app.app_context():
            save_dataset(df, "empty_data.csv")

        response = client.get("/download-cleaned?filename=empty_data.csv")

        assert response.status_code == 200
        csv_text = response.data.decode("utf-8")
        lines = csv_text.strip().splitlines()
        assert lines[0] == "col_a,col_b,col_c"
        assert len(lines) == 1  # Only headers, no data rows

    def test_download_cleaned_utf8_encoding(self, client):
        """Downloaded CSV is UTF-8 encoded."""
        csv_content = "name,city\nAlice,München\nBob,São Paulo\n"
        _upload_csv(client, csv_content, "unicode_data.csv")

        response = client.get("/download-cleaned?filename=unicode_data.csv")

        assert response.status_code == 200
        csv_text = response.data.decode("utf-8")
        assert "München" in csv_text
        assert "São Paulo" in csv_text
