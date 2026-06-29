"""Integration tests for the full API flow.

Tests the end-to-end pipeline:
  upload → summary → clean → visualize → train → feature-importance → report → download-cleaned

Also tests error conditions at each step and CORS configuration.
"""

import io
import os

import pytest

from app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a test Flask application."""
    application = create_app()
    application.config["TESTING"] = True
    yield application
    # Clean up any uploaded files after tests
    upload_folder = application.config["UPLOAD_FOLDER"]
    for f in os.listdir(upload_folder):
        if f != ".gitkeep":
            filepath = os.path.join(upload_folder, f)
            if os.path.isfile(filepath):
                os.remove(filepath)


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


def _build_test_csv():
    """Build a test CSV with ~20 rows, mixed numerical/categorical columns.

    Columns: id, name, age, salary, department, score, city
    """
    header = "id,name,age,salary,department,score,city\n"
    rows = [
        "1,Alice,30,55000,Engineering,85.5,New York",
        "2,Bob,25,48000,Marketing,72.3,Los Angeles",
        "3,Charlie,35,62000,Engineering,91.0,Chicago",
        "4,Diana,28,51000,Sales,78.2,New York",
        "5,Eve,32,59000,Engineering,88.7,San Francisco",
        "6,Frank,27,47000,Marketing,69.1,Los Angeles",
        "7,Grace,40,71000,Management,93.4,Chicago",
        "8,Hank,33,58000,Sales,76.8,New York",
        "9,Ivy,29,52000,Engineering,84.2,San Francisco",
        "10,Jack,31,56000,Marketing,75.5,Los Angeles",
        "11,Karen,38,67000,Management,90.1,Chicago",
        "12,Leo,26,49000,Sales,71.9,New York",
        "13,Mona,34,60000,Engineering,87.3,San Francisco",
        "14,Nate,30,54000,Marketing,73.8,Los Angeles",
        "15,Olivia,36,63000,Management,92.5,Chicago",
        "16,Paul,29,51000,Sales,77.4,New York",
        "17,Quinn,32,57000,Engineering,86.1,San Francisco",
        "18,Rita,27,48000,Marketing,70.6,Los Angeles",
        "19,Sam,41,72000,Management,94.2,Chicago",
        "20,Tina,28,50000,Sales,74.9,New York",
    ]
    return header + "\n".join(rows) + "\n"


def _upload_test_csv(client, csv_content=None, filename="integration_test.csv"):
    """Helper to upload the test CSV and return the response JSON."""
    if csv_content is None:
        csv_content = _build_test_csv()
    data = {"file": (io.BytesIO(csv_content.encode()), filename)}
    response = client.post("/upload", content_type="multipart/form-data", data=data)
    return response


# ---------------------------------------------------------------------------
# Full API Flow Tests
# ---------------------------------------------------------------------------

class TestFullAPIFlow:
    """Test the complete upload → summary → clean → visualize → train → report flow."""

    def test_upload_step(self, client):
        """Step 1: Upload a CSV and verify 200 with filename."""
        response = _upload_test_csv(client)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["filename"] == "integration_test.csv"
        assert json_data["rows"] == 20
        assert json_data["columns"] == 7
        assert "id" in json_data["column_names"]
        assert "salary" in json_data["column_names"]
        assert "department" in json_data["column_names"]

    def test_summary_step(self, client):
        """Step 2: GET /summary and verify row/column counts."""
        _upload_test_csv(client)

        response = client.get("/summary?filename=integration_test.csv")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["row_count"] == 20
        assert json_data["column_count"] == 7
        assert "id" in json_data["columns"]
        assert "department" in json_data["dtypes"]
        assert json_data["duplicate_rows"] == 0

    def test_clean_step(self, client):
        """Step 3: POST /clean (remove_duplicates) and verify success."""
        _upload_test_csv(client)

        response = client.post("/clean", json={
            "filename": "integration_test.csv",
            "operation": "remove_duplicates",
            "params": {},
        })

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["success"] is True
        assert "dataset_info" in json_data
        assert json_data["dataset_info"]["rows"] == 20  # No duplicates in test data

    def test_visualization_step(self, client):
        """Step 4: GET /visualizations (histogram) and verify chart data."""
        _upload_test_csv(client)

        response = client.get(
            "/visualizations?filename=integration_test.csv&type=histogram&x=age"
        )

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["chart_type"] == "histogram"
        assert "data" in json_data
        assert len(json_data["data"]) > 0

    def test_train_step(self, client):
        """Step 5: POST /train and verify metrics returned."""
        _upload_test_csv(client)

        response = client.post("/train", json={
            "filename": "integration_test.csv",
            "target": "department",
            "algorithm": "random_forest",
            "task_type": "classification",
        })

        assert response.status_code == 200
        json_data = response.get_json()
        assert "accuracy" in json_data
        assert "precision" in json_data
        assert "recall" in json_data
        assert "f1_score" in json_data
        assert "confusion_matrix" in json_data
        assert "predictions" in json_data
        assert "feature_importance" in json_data
        # Verify metrics are in valid range
        assert 0.0 <= json_data["accuracy"] <= 1.0
        assert 0.0 <= json_data["precision"] <= 1.0

    def test_feature_importance_step(self, client):
        """Step 6: GET /feature-importance and verify feature scores."""
        _upload_test_csv(client)

        # First train a model
        client.post("/train", json={
            "filename": "integration_test.csv",
            "target": "department",
            "algorithm": "random_forest",
            "task_type": "classification",
        })

        response = client.get("/feature-importance?filename=integration_test.csv")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["available"] is True
        assert "features" in json_data
        assert len(json_data["features"]) > 0
        # Each feature should have name and importance
        for feat in json_data["features"]:
            assert "name" in feat
            assert "importance" in feat
            assert 0.0 <= feat["importance"] <= 1.0

    def test_download_report_step(self, client):
        """Step 7: GET /download-report and verify PDF response."""
        _upload_test_csv(client)

        response = client.get("/download-report?filename=integration_test.csv")

        assert response.status_code == 200
        assert response.content_type == "application/pdf"
        # PDF files start with %PDF
        assert response.data[:4] == b"%PDF"

    def test_download_cleaned_step(self, client):
        """Step 8: GET /download-cleaned and verify CSV response."""
        _upload_test_csv(client)

        response = client.get("/download-cleaned?filename=integration_test.csv")

        assert response.status_code == 200
        assert "text/csv" in response.content_type
        # Check Content-Disposition header
        assert "attachment" in response.headers.get("Content-Disposition", "")
        assert "cleaned_integration_test.csv" in response.headers.get("Content-Disposition", "")
        # Verify content is valid CSV text
        csv_text = response.data.decode("utf-8")
        lines = csv_text.strip().split("\n")
        assert len(lines) == 21  # Header + 20 rows

    def test_full_pipeline_end_to_end(self, client):
        """Run the entire pipeline sequentially: upload → summary → clean → viz → train → report."""
        # 1. Upload
        resp = _upload_test_csv(client)
        assert resp.status_code == 200
        filename = resp.get_json()["filename"]

        # 2. Summary
        resp = client.get(f"/summary?filename={filename}")
        assert resp.status_code == 200
        assert resp.get_json()["row_count"] == 20

        # 3. Clean (remove duplicates)
        resp = client.post("/clean", json={
            "filename": filename,
            "operation": "remove_duplicates",
            "params": {},
        })
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        # 4. Visualization (scatter)
        resp = client.get(f"/visualizations?filename={filename}&type=scatter&x=age&y=salary")
        assert resp.status_code == 200
        assert resp.get_json()["chart_type"] == "scatter"
        assert len(resp.get_json()["data"]) == 20

        # 5. Train model (regression)
        resp = client.post("/train", json={
            "filename": filename,
            "target": "salary",
            "algorithm": "random_forest",
            "task_type": "regression",
        })
        assert resp.status_code == 200
        metrics = resp.get_json()
        assert "r2_score" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics

        # 6. Feature importance
        resp = client.get(f"/feature-importance?filename={filename}")
        assert resp.status_code == 200
        fi = resp.get_json()
        assert fi["available"] is True
        assert len(fi["features"]) > 0

        # 7. Download report (PDF)
        resp = client.get(f"/download-report?filename={filename}")
        assert resp.status_code == 200
        assert resp.data[:4] == b"%PDF"

        # 8. Download cleaned CSV
        resp = client.get(f"/download-cleaned?filename={filename}")
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type


# ---------------------------------------------------------------------------
# Error Condition Tests
# ---------------------------------------------------------------------------

class TestErrorConditions:
    """Test error responses at each step of the API flow."""

    # Upload errors
    def test_upload_without_file(self, client):
        """Upload without file field returns 400."""
        response = client.post("/upload", content_type="multipart/form-data", data={})

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_upload_non_csv(self, client):
        """Upload non-CSV file returns 400."""
        data = {"file": (io.BytesIO(b"not a csv"), "data.txt")}
        response = client.post("/upload", content_type="multipart/form-data", data=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    # Summary errors
    def test_summary_nonexistent_file(self, client):
        """Summary for nonexistent file returns 404."""
        response = client.get("/summary?filename=nonexistent_file.csv")

        assert response.status_code == 404
        json_data = response.get_json()
        assert "error" in json_data
        assert json_data["code"] == "NOT_FOUND"

    def test_summary_missing_filename(self, client):
        """Summary without filename parameter returns 400."""
        response = client.get("/summary")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    # Clean errors
    def test_clean_invalid_operation(self, client):
        """Clean with invalid operation returns 400."""
        _upload_test_csv(client)

        response = client.post("/clean", json={
            "filename": "integration_test.csv",
            "operation": "invalid_op_that_does_not_exist",
            "params": {},
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_clean_missing_filename(self, client):
        """Clean without filename returns 400."""
        response = client.post("/clean", json={
            "operation": "remove_duplicates",
            "params": {},
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_clean_missing_operation(self, client):
        """Clean without operation returns 400."""
        _upload_test_csv(client)

        response = client.post("/clean", json={
            "filename": "integration_test.csv",
            "params": {},
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_clean_nonexistent_file(self, client):
        """Clean with nonexistent dataset returns 404."""
        response = client.post("/clean", json={
            "filename": "ghost_file.csv",
            "operation": "remove_duplicates",
            "params": {},
        })

        assert response.status_code == 404
        json_data = response.get_json()
        assert json_data["code"] == "NOT_FOUND"

    # Train errors
    def test_train_insufficient_data(self, client):
        """Train with insufficient data returns 400."""
        # Upload a very small CSV (less than 10 rows)
        small_csv = "a,b,target\n1,2,yes\n3,4,no\n5,6,yes\n"
        _upload_test_csv(client, csv_content=small_csv, filename="small.csv")

        response = client.post("/train", json={
            "filename": "small.csv",
            "target": "target",
            "algorithm": "random_forest",
            "task_type": "classification",
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_train_missing_target(self, client):
        """Train without target field returns 400."""
        _upload_test_csv(client)

        response = client.post("/train", json={
            "filename": "integration_test.csv",
            "algorithm": "random_forest",
            "task_type": "classification",
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_train_invalid_task_type(self, client):
        """Train with invalid task_type returns 400."""
        _upload_test_csv(client)

        response = client.post("/train", json={
            "filename": "integration_test.csv",
            "target": "department",
            "algorithm": "random_forest",
            "task_type": "invalid_type",
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_train_regression_on_categorical_target(self, client):
        """Regression on categorical target returns 400."""
        _upload_test_csv(client)

        response = client.post("/train", json={
            "filename": "integration_test.csv",
            "target": "department",
            "algorithm": "linear_regression",
            "task_type": "regression",
        })

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data
        assert "categorical" in json_data["error"].lower() or "numerical" in json_data["error"].lower()

    # Visualization errors
    def test_visualization_missing_type(self, client):
        """Visualization without type parameter returns 400."""
        _upload_test_csv(client)

        response = client.get("/visualizations?filename=integration_test.csv")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_visualization_missing_x_param(self, client):
        """Histogram without x parameter returns 400."""
        _upload_test_csv(client)

        response = client.get("/visualizations?filename=integration_test.csv&type=histogram")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_visualization_scatter_missing_y(self, client):
        """Scatter plot without y parameter returns 400."""
        _upload_test_csv(client)

        response = client.get(
            "/visualizations?filename=integration_test.csv&type=scatter&x=age"
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    def test_visualization_invalid_type(self, client):
        """Invalid chart type returns 400."""
        _upload_test_csv(client)

        response = client.get(
            "/visualizations?filename=integration_test.csv&type=invalid_chart&x=age"
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    # Feature importance errors
    def test_feature_importance_no_model_trained(self, client):
        """Feature importance without prior training returns available=False."""
        # Use a unique filename to avoid module-level state from other tests
        _upload_test_csv(client, filename="fi_no_model_test.csv")

        response = client.get("/feature-importance?filename=fi_no_model_test.csv")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["available"] is False
        assert json_data["message"] is not None

    def test_feature_importance_missing_filename(self, client):
        """Feature importance without filename returns 400."""
        response = client.get("/feature-importance")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    # Download report errors
    def test_download_report_nonexistent_file(self, client):
        """Download report for nonexistent file returns 404."""
        response = client.get("/download-report?filename=nonexistent.csv")

        assert response.status_code == 404
        json_data = response.get_json()
        assert json_data["code"] == "NOT_FOUND"

    def test_download_report_missing_filename(self, client):
        """Download report without filename returns 400."""
        response = client.get("/download-report")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data

    # Download cleaned errors
    def test_download_cleaned_nonexistent_file(self, client):
        """Download cleaned for nonexistent file returns 404."""
        response = client.get("/download-cleaned?filename=nonexistent.csv")

        assert response.status_code == 404
        json_data = response.get_json()
        assert json_data["code"] == "NOT_FOUND"

    def test_download_cleaned_missing_filename(self, client):
        """Download cleaned without filename returns 400."""
        response = client.get("/download-cleaned")

        assert response.status_code == 400
        json_data = response.get_json()
        assert "error" in json_data


# ---------------------------------------------------------------------------
# CORS Configuration Tests
# ---------------------------------------------------------------------------

class TestCORSConfiguration:
    """Test CORS headers are configured correctly."""

    def test_cors_options_upload(self, client):
        """OPTIONS preflight request to /upload includes CORS headers."""
        response = client.options("/upload", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })

        # Flask-CORS responds to OPTIONS with 200
        assert response.status_code == 200
        # Verify CORS headers are present
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_allow_origin_header(self, client):
        """Regular requests from allowed origin include Access-Control-Allow-Origin."""
        csv_content = _build_test_csv()
        data = {"file": (io.BytesIO(csv_content.encode()), "cors_test.csv")}

        response = client.post(
            "/upload",
            content_type="multipart/form-data",
            data=data,
            headers={"Origin": "http://localhost:5173"},
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"

    def test_cors_options_clean(self, client):
        """OPTIONS preflight request to /clean includes CORS headers."""
        response = client.options("/clean", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_options_train(self, client):
        """OPTIONS preflight request to /train includes CORS headers."""
        response = client.options("/train", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_get_endpoint(self, client):
        """GET request from allowed origin includes CORS headers."""
        _upload_test_csv(client)

        response = client.get(
            "/summary?filename=integration_test.csv",
            headers={"Origin": "http://localhost:5173"},
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
