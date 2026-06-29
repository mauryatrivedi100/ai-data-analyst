"""API route definitions for the AI Data Analyst backend."""

import io
import os

import pandas as pd
from flask import Blueprint, current_app, jsonify, make_response, request, send_file
from werkzeug.utils import secure_filename

from cleaning import (
    convert_column_type,
    detect_outliers,
    drop_columns,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
    label_encode,
    min_max_scale,
    one_hot_encode,
    remove_duplicates,
    remove_missing_rows,
    remove_outliers,
    rename_column,
    standard_scale,
)
from eda import (
    compute_bar,
    compute_box_plot,
    compute_correlation_heatmap,
    compute_histogram,
    compute_line,
    compute_pie,
    compute_scatter,
)
from insights import generate_insights
from ml import train_classification, train_regression, validate_target_column
from report import generate_pdf
from utils import format_file_size, get_column_types, get_dataset_path, load_dataset, save_dataset, validate_csv

bp = Blueprint("routes", __name__)

# Module-level storage for feature importance results keyed by filename
_feature_importance_store = {}

# Module-level storage for cleaning operations history keyed by filename
_cleaning_operations_store = {}

# Module-level storage for AI-generated insights keyed by filename
_insights_store = {}

# Module-level storage for model metrics keyed by filename
_model_metrics_store = {}


@bp.route("/reset-session", methods=["POST"])
def reset_session():
    """Clear all server-side session data for a dataset.

    This removes stored cleaning operations, model metrics, feature importance,
    and insights for the specified file. Optionally deletes the uploaded file.

    Request Body (JSON):
        filename: Name of the uploaded CSV file to reset.

    Returns:
        JSON with success status.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "code": "VALIDATION_ERROR"}), 400

    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "filename is required.", "code": "VALIDATION_ERROR"}), 400

    # Clear all session stores for this filename
    _feature_importance_store.pop(filename, None)
    _cleaning_operations_store.pop(filename, None)
    _insights_store.pop(filename, None)
    _model_metrics_store.pop(filename, None)

    # Optionally remove the uploaded file
    try:
        filepath = get_dataset_path(filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
    except Exception:
        pass  # If file removal fails, that's okay

    return jsonify({"success": True, "message": "Session data cleared."}), 200


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Handle CSV file upload.

    Accepts multipart/form-data with a 'file' field.
    Validates the file extension and size, saves it to uploads/,
    verifies CSV parsability, and returns dataset metadata.

    Returns:
        JSON with filename, original_name, rows, columns, column_names,
        preview (first 20 rows), file_size, and file_size_bytes.
    """
    # Check that a file was included in the request
    if "file" not in request.files:
        return jsonify({"error": "No file provided.", "code": "VALIDATION_ERROR"}), 400

    file = request.files["file"]

    # Validate CSV (extension and size)
    is_valid, error_message = validate_csv(file)
    if not is_valid:
        return jsonify({"error": error_message, "code": "VALIDATION_ERROR"}), 400

    # Reset stream position after validation (validate_csv may have read the stream)
    file.stream.seek(0)

    # Generate a safe filename and save the file
    original_name = file.filename
    safe_name = secure_filename(original_name)

    # Ensure safe_name is not empty after sanitization
    if not safe_name:
        return jsonify({"error": "Invalid filename.", "code": "VALIDATION_ERROR"}), 400

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    save_path = os.path.join(upload_folder, safe_name)

    try:
        file.save(save_path)
    except Exception:
        return jsonify(
            {"error": "Failed to save uploaded file.", "code": "INTERNAL_ERROR"}
        ), 500

    # Verify the file was saved and has content
    file_size_bytes = os.path.getsize(save_path)
    if file_size_bytes < 1:
        # Clean up empty file
        os.remove(save_path)
        return jsonify(
            {"error": "File is empty. Please upload a non-empty CSV file.", "code": "VALIDATION_ERROR"}
        ), 400

    # Try to parse the CSV with pandas
    try:
        df = pd.read_csv(save_path)
    except Exception:
        # File is not a valid/parseable CSV — clean up and return error
        os.remove(save_path)
        return jsonify(
            {"error": "Could not parse file as valid CSV.", "code": "PARSE_ERROR"}
        ), 422

    # Check for empty datasets (header only, no data rows)
    if len(df) == 0:
        os.remove(save_path)
        return jsonify(
            {"error": "CSV file has no data rows. Please upload a file with at least one row of data.", "code": "VALIDATION_ERROR"}
        ), 400

    # Build the success response
    rows = len(df)
    columns = len(df.columns)
    column_names = df.columns.tolist()
    # Replace NaN/Inf with None for JSON serialization
    preview_df = df.head(20).where(df.head(20).notna(), None)
    preview = preview_df.to_dict(orient="records")
    file_size = format_file_size(file_size_bytes)

    return jsonify({
        "filename": safe_name,
        "original_name": original_name,
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
        "preview": preview,
        "file_size": file_size,
        "file_size_bytes": file_size_bytes,
    }), 200


@bp.route("/summary", methods=["GET"])
def get_summary():
    """Return dataset summary metadata.

    Query Parameters:
        filename: Name of the uploaded CSV file.

    Returns:
        JSON with row_count, column_count, columns, dtypes, missing_values,
        duplicate_rows, and memory_usage.
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    row_count = len(df)
    column_count = len(df.columns)
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_usage = format_file_size(memory_bytes)

    return jsonify({
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "memory_usage": memory_usage,
    }), 200


@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Return descriptive statistics for numerical and categorical columns.

    Query Parameters:
        filename: Name of the uploaded CSV file.

    Returns:
        JSON with numerical stats (mean, median, std, min, max, q1, q2, q3)
        and categorical stats (unique_count, top_5).
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    col_types = get_column_types(df)

    # Compute numerical statistics
    numerical_stats = {}
    for col in col_types["numerical"]:
        series = df[col]
        if series.isna().all():
            # All values are missing — return None for all stats
            numerical_stats[col] = {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "q1": None,
                "q2": None,
                "q3": None,
            }
        else:
            numerical_stats[col] = {
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "q1": round(float(series.quantile(0.25)), 2),
                "q2": round(float(series.quantile(0.50)), 2),
                "q3": round(float(series.quantile(0.75)), 2),
            }

    # Compute categorical statistics
    categorical_stats = {}
    for col in col_types["categorical"]:
        series = df[col]
        unique_count = int(series.nunique())
        value_counts = series.value_counts().head(5)
        top_5 = [
            {"value": str(val), "count": int(cnt)}
            for val, cnt in value_counts.items()
        ]
        categorical_stats[col] = {
            "unique_count": unique_count,
            "top_5": top_5,
        }

    return jsonify({
        "numerical": numerical_stats,
        "categorical": categorical_stats,
    }), 200


# Valid cleaning operations
VALID_OPERATIONS = {
    "remove_missing",
    "fill_mean",
    "fill_median",
    "fill_mode",
    "remove_duplicates",
    "detect_outliers",
    "remove_outliers",
    "drop_columns",
    "rename_column",
    "convert_type",
    "label_encode",
    "one_hot_encode",
    "standard_scale",
    "min_max_scale",
}


@bp.route("/clean", methods=["POST"])
def clean_dataset():
    """Apply a cleaning operation to a dataset.

    Request Body (JSON):
        filename: Name of the uploaded CSV file.
        operation: Cleaning operation to perform.
        params: Optional dictionary of operation-specific parameters.

    Returns:
        JSON with success status, operation summary, and updated dataset info.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "code": "VALIDATION_ERROR"}), 400

    filename = data.get("filename")
    operation = data.get("operation")
    params = data.get("params", {})

    # Validate required fields
    if not filename:
        return jsonify({"error": "filename is required.", "code": "VALIDATION_ERROR"}), 400
    if not operation:
        return jsonify({"error": "operation is required.", "code": "VALIDATION_ERROR"}), 400

    # Validate operation is supported
    if operation not in VALID_OPERATIONS:
        return jsonify({
            "error": f"Unknown operation '{operation}'. Valid operations: {sorted(VALID_OPERATIONS)}",
            "code": "VALIDATION_ERROR",
        }), 400

    # Load the dataset
    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Dispatch to the appropriate cleaning function
    try:
        result = _dispatch_operation(df, operation, params)
    except ValueError as e:
        error_msg = str(e)
        # Determine if this is a conflict (would produce invalid state) or validation error
        if "cannot" in error_msg.lower() or "would result" in error_msg.lower():
            return jsonify({"error": error_msg, "code": "OPERATION_ERROR"}), 409
        return jsonify({"error": error_msg, "code": "VALIDATION_ERROR"}), 400
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Handle detect_outliers specially (read-only, no save)
    if operation == "detect_outliers":
        outlier_count, lower_bound, upper_bound, outlier_indices = result
        return jsonify({
            "success": True,
            "summary": f"Detected {outlier_count} outliers in column '{params.get('column')}'.",
            "outlier_info": {
                "outlier_count": outlier_count,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_indices": outlier_indices,
            },
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
            },
        }), 200

    # For all other operations, save the cleaned dataset
    cleaned_df, summary_msg = result
    save_dataset(cleaned_df, filename)

    # Track cleaning operation in session state
    if filename not in _cleaning_operations_store:
        _cleaning_operations_store[filename] = []
    _cleaning_operations_store[filename].append(summary_msg)

    return jsonify({
        "success": True,
        "summary": summary_msg,
        "dataset_info": {
            "rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
        },
    }), 200


def _dispatch_operation(df, operation, params):
    """Dispatch a cleaning operation and return its result.

    For detect_outliers, returns the raw tuple from the function.
    For all other operations, returns (cleaned_df, summary_message).

    Raises:
        ValueError: If required parameters are missing or operation fails validation.
    """
    if operation == "remove_missing":
        cleaned_df, rows_removed = remove_missing_rows(df)
        return cleaned_df, f"Removed {rows_removed} rows with missing values."

    elif operation == "fill_mean":
        column = _get_required_param(params, "column")
        cleaned_df, mean_value = fill_missing_mean(df, column)
        return cleaned_df, f"Filled missing values in '{column}' with mean ({mean_value:.4f})."

    elif operation == "fill_median":
        column = _get_required_param(params, "column")
        cleaned_df, median_value = fill_missing_median(df, column)
        return cleaned_df, f"Filled missing values in '{column}' with median ({median_value:.4f})."

    elif operation == "fill_mode":
        column = _get_required_param(params, "column")
        cleaned_df, mode_value = fill_missing_mode(df, column)
        return cleaned_df, f"Filled missing values in '{column}' with mode ({mode_value})."

    elif operation == "remove_duplicates":
        cleaned_df, duplicates_removed = remove_duplicates(df)
        return cleaned_df, f"Removed {duplicates_removed} duplicate rows."

    elif operation == "detect_outliers":
        column = _get_required_param(params, "column")
        return detect_outliers(df, column)

    elif operation == "remove_outliers":
        column = _get_required_param(params, "column")
        cleaned_df, rows_removed = remove_outliers(df, column)
        return cleaned_df, f"Removed {rows_removed} outlier rows based on column '{column}'."

    elif operation == "drop_columns":
        columns = _get_required_param(params, "columns")
        if not isinstance(columns, list):
            raise ValueError("'columns' parameter must be a list of column names.")
        cleaned_df = drop_columns(df, columns)
        return cleaned_df, f"Dropped {len(columns)} column(s): {columns}."

    elif operation == "rename_column":
        column = _get_required_param(params, "column")
        new_name = _get_required_param(params, "new_name")
        cleaned_df = rename_column(df, column, new_name)
        return cleaned_df, f"Renamed column '{column}' to '{new_name}'."

    elif operation == "convert_type":
        column = _get_required_param(params, "column")
        target_type = _get_required_param(params, "target_type")
        cleaned_df = convert_column_type(df, column, target_type)
        return cleaned_df, f"Converted column '{column}' to type '{target_type}'."

    elif operation == "label_encode":
        column = _get_required_param(params, "column")
        cleaned_df = label_encode(df, column)
        return cleaned_df, f"Applied label encoding to column '{column}'."

    elif operation == "one_hot_encode":
        column = _get_required_param(params, "column")
        cleaned_df = one_hot_encode(df, column)
        return cleaned_df, f"Applied one-hot encoding to column '{column}'."

    elif operation == "standard_scale":
        column = _get_required_param(params, "column")
        cleaned_df = standard_scale(df, column)
        return cleaned_df, f"Applied standard scaling to column '{column}'."

    elif operation == "min_max_scale":
        column = _get_required_param(params, "column")
        cleaned_df = min_max_scale(df, column)
        return cleaned_df, f"Applied min-max scaling to column '{column}'."


def _get_required_param(params, key):
    """Extract a required parameter from the params dict.

    Args:
        params: Dictionary of parameters.
        key: Parameter key to extract.

    Returns:
        The parameter value.

    Raises:
        ValueError: If the parameter is missing or empty.
    """
    value = params.get(key)
    if value is None:
        raise ValueError(f"Parameter '{key}' is required for this operation.")
    if isinstance(value, str) and len(value.strip()) == 0:
        raise ValueError(f"Parameter '{key}' cannot be empty.")
    return value


@bp.route("/download-cleaned", methods=["GET"])
def download_cleaned():
    """Download the cleaned dataset as a UTF-8 CSV file.

    Query Parameters:
        filename: Name of the uploaded/cleaned CSV file.

    Returns:
        CSV file download with Content-Disposition header.
        Filename format: cleaned_<original_name_without_extension>.csv
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Generate CSV content from DataFrame (including headers, even if empty)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Compute download filename: cleaned_<original_name_without_extension>.csv
    original_name_without_ext = filename.rsplit(".", 1)[0] if "." in filename else filename
    download_filename = f"cleaned_{original_name_without_ext}.csv"

    # Build response with CSV content
    response = make_response(csv_content)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'

    return response


# Valid visualization chart types
VALID_CHART_TYPES = {"histogram", "scatter", "line", "bar", "pie", "box", "heatmap"}

# Chart types that require an x parameter
CHARTS_REQUIRING_X = {"histogram", "scatter", "line", "bar", "pie", "box"}

# Chart types that require both x and y parameters
CHARTS_REQUIRING_XY = {"scatter", "line", "bar"}


@bp.route("/visualizations", methods=["GET"])
def get_visualizations():
    """Generate chart data for exploratory data analysis.

    Query Parameters:
        filename: Name of the uploaded CSV file (required).
        type: Chart type to generate (required). One of: histogram, scatter,
              line, bar, pie, box, heatmap.
        x: Column name for the x-axis (required for all types except heatmap).
        y: Column name for the y-axis (required for scatter, line, bar).

    Returns:
        JSON with chart_type and computed data.
    """
    filename = request.args.get("filename")
    chart_type = request.args.get("type")
    x_col = request.args.get("x")
    y_col = request.args.get("y")

    # Validate required parameters
    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400
    if not chart_type:
        return jsonify({"error": "type query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    # Validate chart type
    if chart_type not in VALID_CHART_TYPES:
        return jsonify({
            "error": f"Unknown chart type '{chart_type}'. Valid types: {sorted(VALID_CHART_TYPES)}",
            "code": "VALIDATION_ERROR",
        }), 400

    # Validate required axis parameters based on chart type
    if chart_type in CHARTS_REQUIRING_X and not x_col:
        return jsonify({
            "error": f"'x' query parameter is required for chart type '{chart_type}'.",
            "code": "VALIDATION_ERROR",
        }), 400

    if chart_type in CHARTS_REQUIRING_XY and not y_col:
        return jsonify({
            "error": f"'y' query parameter is required for chart type '{chart_type}'.",
            "code": "VALIDATION_ERROR",
        }), 400

    # Load the dataset
    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Dispatch to the appropriate compute function
    try:
        if chart_type == "histogram":
            data = compute_histogram(df, x_col)
        elif chart_type == "scatter":
            data = compute_scatter(df, x_col, y_col)
        elif chart_type == "line":
            data = compute_line(df, x_col, y_col)
        elif chart_type == "bar":
            data = compute_bar(df, x_col, y_col)
        elif chart_type == "pie":
            data = compute_pie(df, x_col)
        elif chart_type == "box":
            data = compute_box_plot(df, x_col)
        elif chart_type == "heatmap":
            data = compute_correlation_heatmap(df)
    except ValueError as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    return jsonify({"chart_type": chart_type, "data": data}), 200


@bp.route("/train", methods=["POST"])
def train_model():
    """Train a machine learning model on a dataset.

    Request Body (JSON):
        filename: Name of the uploaded CSV file.
        target: Target column name for prediction.
        algorithm: Algorithm identifier (e.g., "random_forest", "logistic_regression").
        task_type: Either "classification" or "regression".

    Returns:
        JSON with model metrics, predictions, and feature importance.
        Classification: {accuracy, precision, recall, f1_score, confusion_matrix, predictions, feature_importance}
        Regression: {r2_score, mae, mse, rmse, predictions, feature_importance}
    """
    global _feature_importance_store

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "code": "VALIDATION_ERROR"}), 400

    filename = data.get("filename")
    target = data.get("target")
    algorithm = data.get("algorithm")
    task_type = data.get("task_type")

    # Validate required fields
    if not filename:
        return jsonify({"error": "filename is required.", "code": "VALIDATION_ERROR"}), 400
    if not target:
        return jsonify({"error": "target is required.", "code": "VALIDATION_ERROR"}), 400
    if not algorithm:
        return jsonify({"error": "algorithm is required.", "code": "VALIDATION_ERROR"}), 400
    if not task_type:
        return jsonify({"error": "task_type is required.", "code": "VALIDATION_ERROR"}), 400

    # Validate task_type value
    if task_type not in ("classification", "regression"):
        return jsonify({
            "error": "task_type must be 'classification' or 'regression'.",
            "code": "VALIDATION_ERROR",
        }), 400

    # Load the dataset
    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Validate target column
    is_valid, error_msg = validate_target_column(df, target, task_type)
    if not is_valid:
        return jsonify({"error": error_msg, "code": "VALIDATION_ERROR"}), 400

    # Dispatch to appropriate training function
    try:
        if task_type == "classification":
            result = train_classification(df, target, algorithm)
            # Store feature importance for later retrieval by filename
            _feature_importance_store[filename] = result.get("feature_importance")
            # Build response without the model object
            response = {
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "confusion_matrix": result["confusion_matrix"],
                "predictions": result["predictions"],
                "feature_importance": result["feature_importance"],
            }
            # Store model metrics for report generation
            _model_metrics_store[filename] = {
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "confusion_matrix": result["confusion_matrix"],
            }
        else:
            result = train_regression(df, target, algorithm)
            # Store feature importance for later retrieval by filename
            _feature_importance_store[filename] = result.get("feature_importance")
            # Build response without the model object
            response = {
                "r2_score": result["r2_score"],
                "mae": result["mae"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "predictions": result["predictions"],
                "feature_importance": result["feature_importance"],
            }
            # Store model metrics for report generation
            _model_metrics_store[filename] = {
                "r2_score": result["r2_score"],
                "mae": result["mae"],
                "mse": result["mse"],
                "rmse": result["rmse"],
            }
    except ValueError as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
    except Exception as e:
        return jsonify({
            "error": f"Model training failed: {str(e)}",
            "code": "TRAINING_ERROR",
        }), 422

    return jsonify(response), 200


@bp.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """Return feature importance scores from the most recently trained model for a dataset.

    Query Parameters:
        filename: Name of the dataset to retrieve feature importance for.

    Returns:
        JSON with features list, availability flag, and optional message.
        {features: [{name, importance}], available: bool, message: str|null}
    """
    filename = request.args.get("filename")

    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    importance = _feature_importance_store.get(filename)

    if importance is None:
        return jsonify({
            "features": [],
            "available": False,
            "message": "No model has been trained yet. Please train a model first.",
        }), 200

    return jsonify(importance), 200


@bp.route("/generate-insights", methods=["POST"])
def generate_insights_route():
    """Generate AI-powered insights for a dataset using the Gemini API.

    Request Body (JSON):
        filename: Name of the uploaded CSV file.

    Returns:
        JSON with structured insights: {overview, observations, business_insights,
        risks, recommendations}.

    Error Responses:
        400: Missing filename or no dataset specified.
        404: Dataset file not found.
        504: Gemini API did not respond in time (timeout).
        502: Gemini API returned an error or is unavailable.
        500: Unexpected internal error.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON.", "code": "VALIDATION_ERROR"}), 400

    filename = data.get("filename")

    # Validate filename is provided
    if not filename:
        return jsonify({"error": "filename is required.", "code": "VALIDATION_ERROR"}), 400

    # Load the dataset
    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Gather dataset context: summary
    row_count = len(df)
    column_count = len(df.columns)
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())

    dataset_summary = {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }

    # Gather dataset context: descriptive statistics
    col_types = get_column_types(df)
    statistics = {}

    numerical_stats = {}
    for col in col_types["numerical"]:
        series = df[col]
        if not series.isna().all():
            numerical_stats[col] = {
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }

    categorical_stats = {}
    for col in col_types["categorical"]:
        series = df[col]
        unique_count = int(series.nunique())
        value_counts = series.value_counts().head(5)
        top_5 = [
            {"value": str(val), "count": int(cnt)}
            for val, cnt in value_counts.items()
        ]
        categorical_stats[col] = {
            "unique_count": unique_count,
            "top_5": top_5,
        }

    statistics = {"numerical": numerical_stats, "categorical": categorical_stats}

    # Gather dataset context: correlations (requires at least 2 numerical columns)
    correlations = None
    try:
        if len(col_types["numerical"]) >= 2:
            correlations = compute_correlation_heatmap(df)
    except Exception:
        # If correlation computation fails, proceed without it
        correlations = None

    # Get model results if available from the feature importance store
    model_results = _feature_importance_store.get(filename)

    # Call the insights generation pipeline
    try:
        insights = generate_insights(dataset_summary, statistics, correlations, model_results)
    except TimeoutError:
        return jsonify({
            "error": "AI service did not respond in time.",
            "code": "TIMEOUT_ERROR",
        }), 504
    except (RuntimeError, ConnectionError):
        return jsonify({
            "error": "AI service is currently unavailable.",
            "code": "API_ERROR",
        }), 502
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "code": "CONFIGURATION_ERROR",
        }), 500
    except Exception:
        return jsonify({
            "error": "An unexpected error occurred while generating insights.",
            "code": "INTERNAL_ERROR",
        }), 500

    # Store insights in session state for report generation
    _insights_store[filename] = insights

    return jsonify(insights), 200


@bp.route("/download-report", methods=["GET"])
def download_report():
    """Generate and download a PDF report of the analysis session.

    Query Parameters:
        filename: Name of the uploaded CSV file.

    Returns:
        PDF file download with Content-Disposition attachment header.

    Error Responses:
        400: Missing filename query parameter.
        404: Dataset file not found.
        500: PDF generation failed.
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename query parameter is required.", "code": "VALIDATION_ERROR"}), 400

    # Verify the dataset exists
    try:
        df = load_dataset(filename)
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please upload again.", "code": "NOT_FOUND"}), 404
    except Exception:
        return jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_ERROR"}), 500

    # Gather session data: compute summary from current dataset state
    row_count = len(df)
    column_count = len(df.columns)
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_usage = format_file_size(memory_bytes)

    summary = {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "memory_usage": memory_usage,
    }

    # Gather cleaning operations from session state
    cleaning_operations = _cleaning_operations_store.get(filename, [])

    # Visualizations are client-side, pass empty list
    visualizations = []

    # Gather model metrics from session state
    model_metrics = _model_metrics_store.get(filename)

    # Gather feature importance from session state
    feature_importance = _feature_importance_store.get(filename)

    # Gather insights from session state
    insights = _insights_store.get(filename)

    # Derive original name from the filename (strip extension for display)
    original_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Build session data dict for PDF generation
    session_data = {
        "filename": filename,
        "original_name": original_name,
        "summary": summary,
        "statistics": None,  # Not stored separately; summary covers key info
        "cleaning_operations": cleaning_operations,
        "visualizations": visualizations,
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "insights": insights,
    }

    # Generate PDF
    try:
        pdf_path = generate_pdf(session_data)
    except Exception:
        return jsonify({
            "error": "Failed to generate PDF report.",
            "code": "INTERNAL_ERROR",
        }), 500

    # Return the generated PDF file as a download
    try:
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
        )
    except Exception:
        return jsonify({
            "error": "Failed to send PDF report.",
            "code": "INTERNAL_ERROR",
        }), 500
