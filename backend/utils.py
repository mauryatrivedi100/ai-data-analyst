"""Shared utility functions for the AI Data Analyst backend."""

import os

import pandas as pd


# Path to the uploads directory (relative to this file's location)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Maximum file size: 50 MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes


def get_dataset_path(filename):
    """Return the full path to a file in the uploads/ directory.

    Args:
        filename: Name of the file within uploads/.

    Returns:
        Full absolute path to the file.

    Raises:
        ValueError: If filename is empty or None.
    """
    if not filename:
        raise ValueError("Filename cannot be empty.")
    return os.path.join(UPLOAD_FOLDER, filename)


def load_dataset(filename):
    """Load a CSV file from the uploads/ directory into a pandas DataFrame.

    Args:
        filename: Name of the CSV file within uploads/.

    Returns:
        A pandas DataFrame with the file contents.

    Raises:
        ValueError: If filename is empty or None.
        FileNotFoundError: If the file does not exist.
    """
    path = get_dataset_path(filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {filename}")
    return pd.read_csv(path)


def save_dataset(df, filename):
    """Save a DataFrame to CSV in the uploads/ directory.

    Args:
        df: The pandas DataFrame to save.
        filename: Name for the CSV file within uploads/.

    Returns:
        The full path where the file was saved.

    Raises:
        ValueError: If filename is empty or None, or if df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    path = get_dataset_path(filename)
    df.to_csv(path, index=False)
    return path


def get_column_types(df):
    """Classify DataFrame columns into numerical and categorical types.

    Args:
        df: A pandas DataFrame.

    Returns:
        A dict with keys 'numerical' and 'categorical', each containing
        a list of column names.

    Raises:
        ValueError: If df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(exclude=["number"]).columns.tolist()
    return {"numerical": numerical, "categorical": categorical}


def format_file_size(size_bytes):
    """Format a byte count into a human-readable string.

    Uses "bytes" for values < 1024, "KB" for values < 1024*1024,
    and "MB" for values >= 1024*1024.

    Args:
        size_bytes: Non-negative integer representing file size in bytes.

    Returns:
        A formatted string like "512 bytes", "3.45 KB", or "2.10 MB".

    Raises:
        ValueError: If size_bytes is negative.
    """
    if size_bytes < 0:
        raise ValueError("Size in bytes cannot be negative.")

    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        value = size_bytes / 1024
        return f"{value:.2f} KB"
    else:
        value = size_bytes / (1024 * 1024)
        return f"{value:.2f} MB"


def validate_csv(file):
    """Validate that a file is an acceptable CSV upload.

    Checks that the file has a .csv extension (case-insensitive) and that
    its size is between 1 byte and 50 MB inclusive.

    Args:
        file: A Flask FileStorage object (or similar) with .filename and
              .content_length attributes. If content_length is 0 or None,
              the file stream is read to determine size.

    Returns:
        A tuple (is_valid, error_message). If valid, error_message is None.
    """
    if file is None:
        return False, "No file provided."

    # Check filename exists
    filename = getattr(file, "filename", None)
    if not filename:
        return False, "No filename provided."

    # Check .csv extension (case-insensitive)
    if not filename.lower().endswith(".csv"):
        return False, "Only CSV files are accepted. Please upload a file with a .csv extension."

    # Determine file size
    content_length = getattr(file, "content_length", None)
    if content_length and content_length > 0:
        size = content_length
    else:
        # Fall back to reading the stream to determine size
        current_pos = file.stream.tell() if hasattr(file, "stream") else 0
        if hasattr(file, "stream"):
            file.stream.seek(0, 2)  # Seek to end
            size = file.stream.tell()
            file.stream.seek(current_pos)  # Reset position
        else:
            return False, "Unable to determine file size."

    # Check minimum size (1 byte)
    if size < 1:
        return False, "File is empty. Please upload a non-empty CSV file."

    # Check maximum size (50 MB)
    if size > MAX_FILE_SIZE:
        return False, f"File exceeds the 50 MB size limit. Your file is {format_file_size(size)}."

    return True, None
