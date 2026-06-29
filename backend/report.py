"""Report generation module for compiling PDF reports."""

import os
from datetime import datetime
from fpdf import FPDF


# Output directory for generated reports
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


class ReportPDF(FPDF):
    """Custom PDF class with header/footer styling."""

    def header(self):
        """Add page header with a thin line."""
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "AI Data Analyst Report", align="C")
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

    def footer(self):
        """Add page footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf(session_data):
    """
    Compile a PDF report from completed analysis steps.

    Args:
        session_data: dict with keys:
            - filename: str
            - original_name: str
            - summary: DatasetSummary or None
            - statistics: DescriptiveStatistics or None
            - cleaning_operations: list[str] or None
            - visualizations: list[dict] or None
            - model_metrics: dict or None
            - feature_importance: dict or None
            - insights: dict or None

    Returns:
        str: File path of the generated PDF.
    """
    if not session_data:
        raise ValueError("Session data is required to generate a report.")

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Extract metadata
    original_name = session_data.get("original_name", "dataset")
    timestamp = datetime.now()

    # Build cover page
    build_cover_page(pdf, original_name, timestamp)

    # Only include sections for completed analysis steps
    summary = session_data.get("summary")
    if summary:
        build_summary_section(pdf, summary)

    cleaning_operations = session_data.get("cleaning_operations")
    if cleaning_operations:
        build_cleaning_section(pdf, cleaning_operations)

    visualizations = session_data.get("visualizations")
    if visualizations:
        build_visualizations_section(pdf, visualizations)

    model_metrics = session_data.get("model_metrics")
    feature_importance = session_data.get("feature_importance")
    if model_metrics:
        build_model_section(pdf, model_metrics, feature_importance)

    insights = session_data.get("insights")
    if insights:
        build_insights_section(pdf, insights)

    # Generate filename and save
    safe_name = _sanitize_filename(original_name)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"report_{safe_name}_{timestamp_str}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)

    pdf.output(pdf_path)
    return pdf_path


def build_cover_page(pdf, dataset_name, timestamp):
    """
    Build the cover page with title, dataset name, and generation timestamp.

    Args:
        pdf: FPDF instance
        dataset_name: str - name of the dataset
        timestamp: datetime - report generation timestamp
    """
    pdf.add_page()

    # Add vertical spacing for centered layout
    pdf.ln(60)

    # Title
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 15, "AI Data Analyst Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Decorative line
    pdf.set_draw_color(52, 152, 219)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)

    # Dataset name
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(0, 10, f"Dataset: {dataset_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Timestamp
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(127, 140, 141)
    formatted_time = timestamp.strftime("%B %d, %Y at %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {formatted_time}", align="C", new_x="LMARGIN", new_y="NEXT")


def build_summary_section(pdf, summary_data):
    """
    Build the dataset summary section.

    Args:
        pdf: FPDF instance
        summary_data: dict with row_count, column_count, columns, missing_values, etc.
    """
    pdf.add_page()
    _add_section_header(pdf, "Dataset Summary")

    # Basic stats
    row_count = summary_data.get("row_count", "N/A")
    col_count = summary_data.get("column_count", "N/A")

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 8, f"Rows: {row_count}    Columns: {col_count}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Column names
    columns = summary_data.get("columns", [])
    if columns:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Columns:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        # Display columns in wrapped format
        col_text = ", ".join(columns)
        pdf.multi_cell(0, 6, col_text)
        pdf.ln(4)

    # Data types
    dtypes = summary_data.get("dtypes", {})
    if dtypes:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Data Types:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for col_name, dtype in dtypes.items():
            pdf.cell(0, 6, f"  {col_name}: {dtype}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Missing values
    missing = summary_data.get("missing_values", {})
    if missing:
        has_missing = any(v > 0 for v in missing.values())
        if has_missing:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 8, "Missing Values:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for col_name, count in missing.items():
                if count > 0:
                    pdf.cell(0, 6, f"  {col_name}: {count} missing", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # Duplicate rows
    duplicates = summary_data.get("duplicate_rows")
    if duplicates is not None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"Duplicate Rows: {duplicates}", new_x="LMARGIN", new_y="NEXT")

    # Memory usage
    memory = summary_data.get("memory_usage")
    if memory:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"Memory Usage: {memory}", new_x="LMARGIN", new_y="NEXT")


def build_cleaning_section(pdf, cleaning_operations):
    """
    Build the cleaning operations section.

    Args:
        pdf: FPDF instance
        cleaning_operations: list[str] - list of cleaning operations performed
    """
    pdf.add_page()
    _add_section_header(pdf, "Data Cleaning Operations")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(33, 37, 41)

    if not cleaning_operations:
        pdf.cell(0, 8, "No cleaning operations were performed.", new_x="LMARGIN", new_y="NEXT")
        return

    for i, operation in enumerate(cleaning_operations, 1):
        pdf.set_font("Helvetica", "", 10)
        # Encode to handle special characters
        safe_op = _safe_text(operation)
        pdf.cell(0, 7, f"  {i}. {safe_op}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 8, f"Total operations performed: {len(cleaning_operations)}", new_x="LMARGIN", new_y="NEXT")


def build_visualizations_section(pdf, charts):
    """
    Build the visualizations section with text descriptions of charts.
    Chart images can be embedded later if matplotlib plots are saved to disk.

    Args:
        pdf: FPDF instance
        charts: list[dict] - chart configurations with type, columns, etc.
    """
    pdf.add_page()
    _add_section_header(pdf, "Visualizations")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(33, 37, 41)

    if not charts:
        pdf.cell(0, 8, "No visualizations were generated.", new_x="LMARGIN", new_y="NEXT")
        return

    for i, chart in enumerate(charts, 1):
        chart_type = chart.get("type", "Unknown")
        x_col = chart.get("x", "")
        y_col = chart.get("y", "")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"Chart {i}: {chart_type.title()}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)

        # Build description based on chart type
        if chart_type in ("histogram", "box_plot", "pie"):
            description = f"  Column: {x_col or chart.get('column', 'N/A')}"
        elif chart_type == "correlation_heatmap":
            description = "  Pairwise Pearson correlation for all numerical columns"
        else:
            description = f"  X-axis: {x_col}, Y-axis: {y_col}"

        pdf.cell(0, 6, description, new_x="LMARGIN", new_y="NEXT")

        # If chart has an image path, try to embed it
        image_path = chart.get("image_path")
        if image_path and os.path.exists(image_path):
            try:
                pdf.image(image_path, x=15, w=180)
                pdf.ln(5)
            except Exception:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(127, 140, 141)
                pdf.cell(0, 6, "  [Chart image could not be embedded]", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(33, 37, 41)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(127, 140, 141)
            pdf.cell(0, 6, "  [Chart rendered in browser - see application for interactive view]", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(33, 37, 41)

        pdf.ln(4)


def build_model_section(pdf, metrics, feature_importance=None):
    """
    Build the model evaluation metrics section.

    Args:
        pdf: FPDF instance
        metrics: dict - classification or regression metrics
        feature_importance: dict or None - feature importance data
    """
    pdf.add_page()
    _add_section_header(pdf, "Model Evaluation")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(33, 37, 41)

    # Determine if classification or regression based on metrics keys
    if "accuracy" in metrics:
        _build_classification_metrics(pdf, metrics)
    elif "r2_score" in metrics:
        _build_regression_metrics(pdf, metrics)
    else:
        pdf.cell(0, 8, "Model metrics format not recognized.", new_x="LMARGIN", new_y="NEXT")

    # Feature importance
    if feature_importance and feature_importance.get("available", False):
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(52, 73, 94)
        pdf.cell(0, 10, "Feature Importance", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        features = feature_importance.get("features", [])
        if features:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(33, 37, 41)
            # Table header
            pdf.cell(100, 7, "Feature", border=1)
            pdf.cell(50, 7, "Importance", border=1, new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 9)
            for feat in features:
                name = _safe_text(str(feat.get("name", "")))
                importance = feat.get("importance", 0)
                pdf.cell(100, 7, name, border=1)
                pdf.cell(50, 7, f"{importance:.4f}", border=1, new_x="LMARGIN", new_y="NEXT")


def build_insights_section(pdf, insights):
    """
    Build the AI-generated insights section.

    Args:
        pdf: FPDF instance
        insights: dict with keys: overview, observations, business_insights, risks, recommendations
    """
    pdf.add_page()
    _add_section_header(pdf, "AI-Generated Insights")

    sections = [
        ("overview", "Dataset Overview"),
        ("observations", "Key Observations"),
        ("business_insights", "Business Insights"),
        ("risks", "Potential Risks"),
        ("recommendations", "Actionable Recommendations"),
    ]

    for key, title in sections:
        content = insights.get(key)
        if content:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(52, 73, 94)
            pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(33, 37, 41)
            safe_content = _safe_text(str(content))
            pdf.multi_cell(0, 5, safe_content)
            pdf.ln(6)


# --- Private Helpers ---


def _build_classification_metrics(pdf, metrics):
    """Build classification metrics table."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(0, 9, "Classification Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Metrics table
    metric_items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision (weighted)", metrics.get("precision")),
        ("Recall (weighted)", metrics.get("recall")),
        ("F1 Score (weighted)", metrics.get("f1_score")),
    ]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(90, 7, "Metric", border=1)
    pdf.cell(50, 7, "Value", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for name, value in metric_items:
        if value is not None:
            pdf.cell(90, 7, name, border=1)
            pdf.cell(50, 7, f"{value:.4f}", border=1, new_x="LMARGIN", new_y="NEXT")

    # Confusion matrix note
    confusion_matrix = metrics.get("confusion_matrix")
    if confusion_matrix:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, "Confusion Matrix:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for row in confusion_matrix:
            row_str = "  " + "  ".join(str(val) for val in row)
            pdf.cell(0, 6, row_str, new_x="LMARGIN", new_y="NEXT")


def _build_regression_metrics(pdf, metrics):
    """Build regression metrics table."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(52, 73, 94)
    pdf.cell(0, 9, "Regression Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    metric_items = [
        ("R² Score", metrics.get("r2_score")),
        ("MAE", metrics.get("mae")),
        ("MSE", metrics.get("mse")),
        ("RMSE", metrics.get("rmse")),
    ]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(90, 7, "Metric", border=1)
    pdf.cell(50, 7, "Value", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for name, value in metric_items:
        if value is not None:
            pdf.cell(90, 7, name, border=1)
            pdf.cell(50, 7, f"{value:.4f}", border=1, new_x="LMARGIN", new_y="NEXT")


def _add_section_header(pdf, title):
    """Add a styled section header."""
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(52, 152, 219)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)


def _sanitize_filename(name):
    """Sanitize a filename by removing extension and special characters."""
    # Remove file extension
    base = os.path.splitext(name)[0] if name else "dataset"
    # Replace problematic characters
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)
    return safe or "dataset"


def _safe_text(text):
    """Encode text safely for PDF output, replacing problematic characters."""
    if not text:
        return ""
    # Replace characters that might cause issues with latin-1 encoding
    return text.encode("latin-1", errors="replace").decode("latin-1")
