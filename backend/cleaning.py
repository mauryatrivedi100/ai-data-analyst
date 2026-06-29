"""Data cleaning module providing transformation and cleaning operations."""

import pandas as pd
import numpy as np


# --- Missing Value Handling ---


def remove_missing_rows(df):
    """Remove all rows containing at least one missing (NaN) value.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned_df, rows_removed) where cleaned_df has no NaN values
        and rows_removed is the count of rows that were removed.

    Raises:
        ValueError: If all rows contain at least one NaN value, meaning removal
            would result in an empty dataset.
    """
    df_copy = df.copy()
    cleaned_df = df_copy.dropna()
    rows_removed = len(df_copy) - len(cleaned_df)

    if len(cleaned_df) == 0:
        raise ValueError(
            "Cannot remove rows: all rows contain missing values. "
            "Removal would result in an empty dataset."
        )

    return cleaned_df.reset_index(drop=True), rows_removed


def fill_missing_mean(df, column):
    """Fill missing values in a column with the arithmetic mean of non-missing values.

    Args:
        df: Input DataFrame.
        column: Name of the column to fill.

    Returns:
        Tuple of (cleaned_df, mean_value) where cleaned_df has NaN values in the
        specified column replaced with the mean, and mean_value is the computed mean.

    Raises:
        ValueError: If column does not exist in the DataFrame.
        ValueError: If column is not numerical (categorical columns cannot use mean).
        ValueError: If all values in the column are NaN.
    """
    _validate_column_exists(df, column)
    _validate_numerical_column(df, column)
    _validate_not_all_nan(df, column)

    df_copy = df.copy()
    mean_value = df_copy[column].mean()
    df_copy[column] = df_copy[column].fillna(mean_value)

    return df_copy, float(mean_value)


def fill_missing_median(df, column):
    """Fill missing values in a column with the median of non-missing values.

    Args:
        df: Input DataFrame.
        column: Name of the column to fill.

    Returns:
        Tuple of (cleaned_df, median_value) where cleaned_df has NaN values in the
        specified column replaced with the median, and median_value is the computed median.

    Raises:
        ValueError: If column does not exist in the DataFrame.
        ValueError: If column is not numerical (categorical columns cannot use median).
        ValueError: If all values in the column are NaN.
    """
    _validate_column_exists(df, column)
    _validate_numerical_column(df, column)
    _validate_not_all_nan(df, column)

    df_copy = df.copy()
    median_value = df_copy[column].median()
    df_copy[column] = df_copy[column].fillna(median_value)

    return df_copy, float(median_value)


def fill_missing_mode(df, column):
    """Fill missing values in a column with the mode (most frequent value).

    If multiple values share the highest frequency, the first encountered value
    is used.

    Args:
        df: Input DataFrame.
        column: Name of the column to fill.

    Returns:
        Tuple of (cleaned_df, mode_value) where cleaned_df has NaN values in the
        specified column replaced with the mode, and mode_value is the computed mode.

    Raises:
        ValueError: If column does not exist in the DataFrame.
        ValueError: If all values in the column are NaN.
    """
    _validate_column_exists(df, column)
    _validate_not_all_nan(df, column)

    df_copy = df.copy()
    mode_series = df_copy[column].mode()
    mode_value = mode_series.iloc[0]
    df_copy[column] = df_copy[column].fillna(mode_value)

    return df_copy, mode_value


# --- Duplicate Handling ---


def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame, keeping the first occurrence.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned_df, duplicates_removed) where cleaned_df is a copy
        with duplicates removed and duplicates_removed is the count of rows removed.
    """
    df_copy = df.copy()
    duplicates_mask = df_copy.duplicated(keep='first')
    duplicates_removed = int(duplicates_mask.sum())
    cleaned_df = df_copy[~duplicates_mask].reset_index(drop=True)
    return cleaned_df, duplicates_removed


# --- Outlier Detection and Removal ---


def detect_outliers(df, column):
    """Detect outliers in a numerical column using the IQR method.

    Args:
        df: Input DataFrame.
        column: Name of the column to check for outliers.

    Returns:
        Tuple of (outlier_count, lower_bound, upper_bound, outlier_indices) where
        outlier_count is the number of outlier values, lower_bound is Q1 - 1.5*IQR,
        upper_bound is Q3 + 1.5*IQR, and outlier_indices is a list of row indices
        containing outlier values.

    Raises:
        ValueError: If the column does not exist in the DataFrame.
        ValueError: If the column is not numerical.
        ValueError: If the column contains no non-NaN values.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical. Outlier detection requires a numerical column."
        )

    series = df[column].dropna()

    if len(series) == 0:
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot compute outlier bounds from an empty set of non-missing values."
        )

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Only flag non-NaN values outside bounds as outliers (NaN rows are not outliers)
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_count = len(outlier_indices)

    return outlier_count, lower_bound, upper_bound, outlier_indices


def remove_outliers(df, column):
    """Remove rows containing outlier values in the specified column.

    Uses the IQR method to detect outliers and removes rows where the column
    value falls outside [lower_bound, upper_bound] (inclusive bounds kept).
    Rows with NaN in the column are preserved (not treated as outliers).

    Args:
        df: Input DataFrame.
        column: Name of the numerical column to check for outliers.

    Returns:
        Tuple of (cleaned_df, rows_removed) where cleaned_df is a copy with
        outlier rows removed and rows_removed is the count of rows removed.

    Raises:
        ValueError: If the column does not exist in the DataFrame.
        ValueError: If the column is not numerical.
        ValueError: If the column contains no non-NaN values.
    """
    outlier_count, lower_bound, upper_bound, outlier_indices = detect_outliers(df, column)

    df_copy = df.copy()
    # Remove only the rows identified as outliers (NaN rows are kept)
    cleaned_df = df_copy.drop(index=outlier_indices).reset_index(drop=True)
    rows_removed = len(outlier_indices)

    return cleaned_df, rows_removed


# --- Column Operations ---


def drop_columns(df, columns: list):
    """Drop specified columns from a DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to drop.

    Returns:
        A new DataFrame with the specified columns removed.

    Raises:
        ValueError: If any column doesn't exist or if all columns would be removed.
    """
    df_copy = df.copy()

    # Validate all specified columns exist
    missing = [col for col in columns if col not in df_copy.columns]
    if missing:
        raise ValueError(
            f"Columns not found in dataset: {missing}"
        )

    # Reject if dropping all columns
    if len(columns) >= len(df_copy.columns):
        raise ValueError(
            "Cannot drop all columns. At least one column must remain in the dataset."
        )

    df_copy = df_copy.drop(columns=columns)
    return df_copy


def rename_column(df, old_name: str, new_name: str):
    """Rename a column in a DataFrame.

    Args:
        df: Input DataFrame.
        old_name: Current column name.
        new_name: New column name (non-empty, max 128 characters, no duplicates).

    Returns:
        A new DataFrame with the column renamed.

    Raises:
        ValueError: If old_name doesn't exist, or new_name is invalid.
    """
    df_copy = df.copy()

    # Validate old_name exists
    if old_name not in df_copy.columns:
        raise ValueError(f"Column '{old_name}' not found in dataset.")

    # Validate new_name is non-empty
    if not new_name or len(new_name.strip()) == 0:
        raise ValueError("New column name cannot be empty.")

    # Validate new_name length
    if len(new_name) > 128:
        raise ValueError(
            "New column name exceeds 128 characters."
        )

    # Validate new_name doesn't duplicate an existing column (unless same as old_name)
    if new_name != old_name and new_name in df_copy.columns:
        raise ValueError(
            f"Column '{new_name}' already exists in the dataset."
        )

    df_copy = df_copy.rename(columns={old_name: new_name})
    return df_copy


def convert_column_type(df, column: str, target_type: str):
    """Convert a column to the specified data type.

    Args:
        df: Input DataFrame.
        column: Name of the column to convert.
        target_type: Target type - one of "int", "float", "string", "datetime".

    Returns:
        A new DataFrame with the column converted to the target type.

    Raises:
        ValueError: If column doesn't exist, target_type is invalid, or conversion fails.
    """
    df_copy = df.copy()

    # Validate column exists
    if column not in df_copy.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    # Validate target_type
    valid_types = ("int", "float", "string", "datetime")
    if target_type not in valid_types:
        raise ValueError(
            f"Invalid target type '{target_type}'. Must be one of: {', '.join(valid_types)}."
        )

    try:
        if target_type == "int":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="raise").astype(int)
        elif target_type == "float":
            df_copy[column] = pd.to_numeric(df_copy[column], errors="raise").astype(float)
        elif target_type == "string":
            df_copy[column] = df_copy[column].astype(str)
        elif target_type == "datetime":
            df_copy[column] = pd.to_datetime(df_copy[column], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to convert column '{column}' to {target_type}: {e}"
        )

    return df_copy


# --- Private helper functions ---


def _validate_column_exists(df, column):
    """Validate that a column exists in the DataFrame.

    Raises:
        ValueError: If the column is not found in the DataFrame.
    """
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_numerical_column(df, column):
    """Validate that a column has a numerical dtype.

    Raises:
        ValueError: If the column is not numerical (i.e., is categorical/object).
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical (dtype: {df[column].dtype}). "
            f"Mean and median operations are only applicable to numerical columns."
        )


def _validate_not_all_nan(df, column):
    """Validate that not all values in a column are NaN.

    Raises:
        ValueError: If all values in the column are missing/NaN.
    """
    if df[column].isna().all():
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot compute fill value from an empty set of non-missing values."
        )

# --- Encoding and Scaling ---


def label_encode(df, column):
    """Apply label encoding to a categorical column.

    Maps each unique category to an integer starting from 0,
    assigned in alphabetical order of category values.

    Args:
        df: Input DataFrame.
        column: Name of the column to encode.

    Returns:
        A new DataFrame with the column values replaced by integer codes.

    Raises:
        ValueError: If column does not exist, is numerical, or contains no valid categories.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is numerical (dtype: {df[column].dtype}). "
            f"Label encoding is only applicable to categorical columns."
        )

    if df[column].isna().all():
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot perform label encoding on a column with no valid categories."
        )

    cleaned_df = df.copy()

    # Get unique non-null values sorted alphabetically
    unique_values = sorted(cleaned_df[column].dropna().unique(), key=str)

    if len(unique_values) == 0:
        raise ValueError(
            f"Column '{column}' contains no valid categories for label encoding."
        )

    # Create mapping: {category: index}
    mapping = {category: idx for idx, category in enumerate(unique_values)}

    # Replace column values with their mapped integers
    cleaned_df[column] = cleaned_df[column].map(mapping)

    return cleaned_df


def one_hot_encode(df, column):
    """Apply one-hot encoding to a categorical column.

    Creates one binary column per unique category, removing the original column.
    New columns are named as OriginalColumnName_CategoryValue.

    Args:
        df: Input DataFrame.
        column: Name of the column to encode.

    Returns:
        A new DataFrame with the original column replaced by binary columns.

    Raises:
        ValueError: If column does not exist, is numerical, or has more than 50 unique categories.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is numerical (dtype: {df[column].dtype}). "
            f"One-hot encoding is only applicable to categorical columns."
        )

    if df[column].isna().all():
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot perform one-hot encoding on a column with no valid categories."
        )

    cleaned_df = df.copy()

    # Check unique count (excluding NaN)
    unique_count = cleaned_df[column].dropna().nunique()

    if unique_count > 50:
        raise ValueError(
            f"Column '{column}' has {unique_count} unique categories, "
            f"which exceeds the maximum of 50 for one-hot encoding."
        )

    if unique_count == 0:
        raise ValueError(
            f"Column '{column}' contains no valid categories for one-hot encoding."
        )

    # Get the position of the original column
    col_idx = cleaned_df.columns.get_loc(column)

    # Create dummy variables
    dummies = pd.get_dummies(cleaned_df[column], prefix=column, dtype=int)

    # Remove the original column
    cleaned_df = cleaned_df.drop(columns=[column])

    # Insert new binary columns at the position of the original column
    for i, dummy_col in enumerate(dummies.columns):
        cleaned_df.insert(col_idx + i, dummy_col, dummies[dummy_col])

    return cleaned_df


def standard_scale(df, column):
    """Apply standard scaling (z-score normalization) to a numerical column.

    Transforms the column using (x - mean) / std, resulting in zero mean
    and unit variance.

    Args:
        df: Input DataFrame.
        column: Name of the column to scale.

    Returns:
        A new DataFrame with the column values z-score normalized.

    Raises:
        ValueError: If column does not exist, is not numerical, all values are NaN,
            or has zero standard deviation.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical (dtype: {df[column].dtype}). "
            f"Standard scaling requires a numerical column."
        )

    if df[column].isna().all():
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot perform standard scaling on a column with no valid values."
        )

    cleaned_df = df.copy()

    # Compute mean and std of non-null values
    col_data = cleaned_df[column].dropna()

    mean = col_data.mean()
    std = col_data.std(ddof=0)

    if std == 0:
        raise ValueError(
            f"Column '{column}' has zero standard deviation (all values are identical). "
            f"Standard scaling cannot be applied to a constant column."
        )

    # Apply z-score normalization
    cleaned_df[column] = (cleaned_df[column] - mean) / std

    return cleaned_df


def min_max_scale(df, column):
    """Apply min-max scaling to a numerical column.

    Transforms the column using (x - min) / (max - min), scaling all values
    to the range [0, 1].

    Args:
        df: Input DataFrame.
        column: Name of the column to scale.

    Returns:
        A new DataFrame with the column values scaled to [0, 1].

    Raises:
        ValueError: If column does not exist, is not numerical, all values are NaN,
            or has min equal to max.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical (dtype: {df[column].dtype}). "
            f"Min-max scaling requires a numerical column."
        )

    if df[column].isna().all():
        raise ValueError(
            f"Column '{column}' contains only missing values. "
            f"Cannot perform min-max scaling on a column with no valid values."
        )

    cleaned_df = df.copy()

    # Compute min and max of non-null values
    col_data = cleaned_df[column].dropna()

    col_min = col_data.min()
    col_max = col_data.max()

    if col_min == col_max:
        raise ValueError(
            f"Column '{column}' has identical min and max values ({col_min}). "
            f"Min-max scaling cannot be applied to a constant column."
        )

    # Apply min-max scaling
    cleaned_df[column] = (cleaned_df[column] - col_min) / (col_max - col_min)

    return cleaned_df
