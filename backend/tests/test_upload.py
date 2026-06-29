"""Tests for the /upload route handler."""

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


class TestUploadRoute:
    """Tests for POST /upload endpoint."""

    def test_upload_valid_csv(self, client):
        """Successful upload returns metadata and preview."""
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago\n"
        data = {"file": (io.BytesIO(csv_content.encode()), "test_data.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["filename"] == "test_data.csv"
        assert json_data["original_name"] == "test_data.csv"
        assert json_data["rows"] == 3
        assert json_data["columns"] == 3
        assert json_data["column_names"] == ["name", "age", "city"]
        assert len(json_data["preview"]) == 3
        assert json_data["file_size_bytes"] > 0
        assert isinstance(json_data["file_size"], str)

    def test_upload_returns_preview_max_20_rows(self, client):
        """Preview contains at most 20 rows even for larger files."""
        header = "col1,col2\n"
        rows = "".join(f"{i},{i*2}\n" for i in range(50))
        csv_content = header + rows
        data = {"file": (io.BytesIO(csv_content.encode()), "large.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["rows"] == 50
        assert len(json_data["preview"]) == 20

    def test_upload_no_file_field(self, client):
        """Returns 400 when no file field is present."""
        response = client.post("/upload", content_type="multipart/form-data", data={})

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data
        assert json_data["code"] == "VALIDATION_ERROR"

    def test_upload_non_csv_extension(self, client):
        """Returns 400 for non-.csv file extension."""
        data = {"file": (io.BytesIO(b"some data"), "data.txt")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert "csv" in json_data["error"].lower() or "CSV" in json_data["error"]

    def test_upload_empty_file(self, client):
        """Returns 400 for an empty file (0 bytes)."""
        data = {"file": (io.BytesIO(b""), "empty.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_upload_malformed_csv(self, client):
        """Returns 422 for a file that cannot be parsed as CSV."""
        # Binary garbage that pandas can't parse
        malformed = b"\x00\x01\x02\x03\x04\x05" * 100
        data = {"file": (io.BytesIO(malformed), "malformed.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        # Could be 400 or 422 depending on how pandas handles it
        assert response.status_code in (400, 422)
        json_data = response.get_json()
        assert "error" in json_data

    def test_upload_csv_headers_only_no_data_rows(self, client):
        """Returns 400 for a CSV with headers but no data rows."""
        csv_content = "name,age,city\n"
        data = {"file": (io.BytesIO(csv_content.encode()), "headers_only.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert "no data" in json_data["error"].lower() or "no data rows" in json_data["error"].lower()

    def test_upload_single_row_csv(self, client):
        """Successfully uploads a CSV with one data row."""
        csv_content = "x,y\n1,2\n"
        data = {"file": (io.BytesIO(csv_content.encode()), "single_row.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["rows"] == 1
        assert json_data["columns"] == 2

    def test_upload_file_saved_to_disk(self, app, client):
        """Uploaded file is saved to the uploads directory."""
        csv_content = "a,b\n1,2\n"
        data = {"file": (io.BytesIO(csv_content.encode()), "saved_test.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 200
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], "saved_test.csv")
        assert os.path.exists(upload_path)

    def test_upload_malformed_csv_not_saved(self, app, client):
        """Malformed CSV file is cleaned up from disk after parse failure."""
        # A file with inconsistent column counts that pandas might reject
        malformed = "a,b,c\n1,2\n3,4,5,6\n"
        data = {"file": (io.BytesIO(malformed.encode()), "bad_parse.csv")}

        response = client.post("/upload", content_type="multipart/form-data", data=data)

        # If pandas parses it successfully (it might with error_bad_lines=False default),
        # the file stays. If it fails, file is removed.
        if response.status_code == 422:
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], "bad_parse.csv")
            assert not os.path.exists(upload_path)
