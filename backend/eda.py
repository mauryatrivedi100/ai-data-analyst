"""Exploratory data analysis module for chart data computation."""

import numpy as np
import pandas as pd


def compute_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> list[dict]:
    """Bin data into histogram format.

    Args:
        df: Input DataFrame.
        column: Column name to compute histogram for (must be numerical).
        bins: Number of bins (default 30).

    Returns:
        List of dicts with bin_start, bin_end, and count.

    Raises:
        ValueError: If column doesn't exist or is not numerical.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical. Histogram requires a numerical column."
        )

    series = df[column].dropna()

    if series.empty:
        return []

    counts, bin_edges = np.histogram(series, bins=bins)

    result = []
    for i in range(len(counts)):
        result.append(
            {
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "count": int(counts[i]),
            }
        )

    return result


def compute_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> list[dict]:
    """Extract x/y data points for a scatter plot.

    Args:
        df: Input DataFrame.
        x_col: Column name for x-axis (must be numerical).
        y_col: Column name for y-axis (must be numerical).

    Returns:
        List of dicts with x and y values.

    Raises:
        ValueError: If columns don't exist or are not numerical.
    """
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in the dataset.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise ValueError(
            f"Column '{x_col}' is not numerical. Scatter plot requires numerical columns."
        )
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(
            f"Column '{y_col}' is not numerical. Scatter plot requires numerical columns."
        )

    subset = df[[x_col, y_col]].dropna()

    result = [
        {"x": float(row[x_col]), "y": float(row[y_col])}
        for _, row in subset.iterrows()
    ]

    return result


def compute_line(df: pd.DataFrame, x_col: str, y_col: str) -> list[dict]:
    """Extract sorted x/y data points for a line chart.

    Args:
        df: Input DataFrame.
        x_col: Column name for x-axis (must be numerical).
        y_col: Column name for y-axis (must be numerical).

    Returns:
        List of dicts with x and y values, sorted by x.

    Raises:
        ValueError: If columns don't exist or are not numerical.
    """
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in the dataset.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise ValueError(
            f"Column '{x_col}' is not numerical. Line chart requires numerical columns."
        )
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(
            f"Column '{y_col}' is not numerical. Line chart requires numerical columns."
        )

    subset = df[[x_col, y_col]].dropna()
    subset = subset.sort_values(by=x_col)

    result = [
        {"x": float(row[x_col]), "y": float(row[y_col])}
        for _, row in subset.iterrows()
    ]

    return result


def compute_bar(df: pd.DataFrame, x_col: str, y_col: str) -> list[dict]:
    """Aggregate data by category for a bar chart.

    Args:
        df: Input DataFrame.
        x_col: Column name for categories (categorical).
        y_col: Column name for values (numerical).

    Returns:
        List of dicts with category and value (mean per category).

    Raises:
        ValueError: If columns don't exist or have wrong types.
    """
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' does not exist in the dataset.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise ValueError(
            f"Column '{y_col}' is not numerical. Bar chart requires a numerical Y-axis column."
        )

    subset = df[[x_col, y_col]].dropna()

    if subset.empty:
        return []

    grouped = subset.groupby(x_col)[y_col].mean()

    result = [
        {"category": str(category), "value": float(value)}
        for category, value in grouped.items()
    ]

    return result


def compute_pie(df: pd.DataFrame, column: str) -> list[dict]:
    """Compute value counts as name/value pairs for a pie chart.

    Args:
        df: Input DataFrame.
        column: Column name to compute value counts for.

    Returns:
        List of dicts with name and value (count).

    Raises:
        ValueError: If column doesn't exist.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    series = df[column].dropna()

    if series.empty:
        return []

    value_counts = series.value_counts()

    result = [
        {"name": str(name), "value": int(count)}
        for name, count in value_counts.items()
    ]

    return result


def compute_box_plot(df: pd.DataFrame, column: str) -> dict:
    """Compute box plot statistics: min, Q1, median, Q3, max, and outliers.

    Args:
        df: Input DataFrame.
        column: Column name to compute box plot for (must be numerical).

    Returns:
        Dict with min, q1, median, q3, max, and outliers list.

    Raises:
        ValueError: If column doesn't exist or is not numerical.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical. Box plot requires a numerical column."
        )

    series = df[column].dropna()

    if series.empty:
        return {
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "outliers": [],
        }

    q1 = float(np.percentile(series, 25))
    median = float(np.percentile(series, 50))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    min_val = float(series[series >= lower_bound].min()) if any(series >= lower_bound) else float(series.min())
    max_val = float(series[series <= upper_bound].max()) if any(series <= upper_bound) else float(series.max())

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_list = [float(v) for v in outliers.values]

    return {
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_val,
        "outliers": outlier_list,
    }


def compute_correlation_heatmap(df: pd.DataFrame) -> dict:
    """Compute pairwise Pearson correlations for all numerical columns.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with columns (list of column names) and matrix (2D list of correlation values).

    Raises:
        ValueError: If DataFrame has fewer than 2 numerical columns.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) < 2:
        raise ValueError(
            "Correlation heatmap requires at least 2 numerical columns in the dataset."
        )

    numerical_df = df[numerical_cols].dropna()

    if numerical_df.empty:
        # Return NaN matrix if no complete rows exist
        size = len(numerical_cols)
        matrix = [[float("nan")] * size for _ in range(size)]
        return {"columns": numerical_cols, "matrix": matrix}

    corr_matrix = numerical_df.corr(method="pearson")

    matrix = corr_matrix.values.tolist()

    return {
        "columns": numerical_cols,
        "matrix": matrix,
    }
