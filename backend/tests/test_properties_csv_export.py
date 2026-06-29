"""Property-based tests for CSV export round-trip and cleaned filename formatting.

Feature: ai-data-analyst
Properties: 18, 19
"""

import io
import os

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cleaned_filename(original_name: str) -> str:
    """Generate the cleaned download filename.

    Strips the last extension from the original filename and returns
    'cleaned_<name_without_extension>.csv'.

    This implements the logic specified in Requirement 8.2:
    filename formatted as "cleaned_<original_dataset_name>.csv" where
    <original_dataset_name> is the name of the uploaded file without its extension.
    """
    # Strip the last extension (everything after the last dot)
    base, ext = os.path.splitext(original_name)
    # If base is empty after stripping (e.g., original was ".csv"), use original_name
    if not base:
        base = original_name
    return f"cleaned_{base}.csv"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating simple DataFrames suitable for CSV round-trip
# We use floats, ints, and strings as column values
simple_float = st.floats(
    min_value=-1e6, max_value=1e6,
    allow_nan=False, allow_infinity=False,
)

simple_int = st.integers(min_value=-10000, max_value=10000)

simple_string = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Pd"),
        whitelist_characters=" _",
    ),
    min_size=1,
    max_size=20,
)

# Strategy for column names: valid identifiers, no duplicates handled by unique_by
column_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=15,
)


@st.composite
def arbitrary_dataframe(draw):
    """Generate a random DataFrame with 1-5 columns and 1-10 rows.

    All columns are numeric (float) for simplicity in round-trip testing.
    """
    n_cols = draw(st.integers(min_value=1, max_value=5))
    n_rows = draw(st.integers(min_value=1, max_value=10))

    # Generate unique column names
    col_names = draw(
        st.lists(column_name, min_size=n_cols, max_size=n_cols, unique=True)
    )

    # Generate data for each column (all floats for reliable round-trip)
    data = {}
    for col in col_names:
        values = draw(st.lists(simple_float, min_size=n_rows, max_size=n_rows))
        data[col] = values

    return pd.DataFrame(data)


@st.composite
def arbitrary_int_dataframe(draw):
    """Generate a random DataFrame with integer columns for exact round-trip."""
    n_cols = draw(st.integers(min_value=1, max_value=5))
    n_rows = draw(st.integers(min_value=1, max_value=10))

    col_names = draw(
        st.lists(column_name, min_size=n_cols, max_size=n_cols, unique=True)
    )

    data = {}
    for col in col_names:
        values = draw(st.lists(simple_int, min_size=n_rows, max_size=n_rows))
        data[col] = values

    return pd.DataFrame(data)


# Strategy for filenames with various extensions
file_extension = st.sampled_from([".csv", ".CSV", ".data.csv", ".tsv", ".xlsx", ".json", ".txt", ".parquet"])
file_basename = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd"), whitelist_characters="_"),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 18: CSV Export Round-Trip
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_csv_export_round_trip_column_names(df):
    """Exporting a DataFrame to CSV and re-reading it preserves column names."""
    # Export to CSV in-memory
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Re-read
    result = pd.read_csv(buffer)

    # Column names must be identical
    assert list(result.columns) == list(df.columns), (
        f"Column mismatch: {list(result.columns)} != {list(df.columns)}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_csv_export_round_trip_shape(df):
    """Exporting to CSV and re-reading preserves the DataFrame shape."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    result = pd.read_csv(buffer)

    assert result.shape == df.shape, (
        f"Shape mismatch: {result.shape} != {df.shape}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_csv_export_round_trip_numeric_values(df):
    """Exporting to CSV and re-reading produces equivalent numeric values within tolerance."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    result = pd.read_csv(buffer)

    # Check each numeric column for approximate equality
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            np.testing.assert_allclose(
                result[col].values,
                df[col].values,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Values mismatch in column '{col}'",
            )


@settings(max_examples=100)
@given(df=arbitrary_int_dataframe())
def test_csv_export_round_trip_integer_exact(df):
    """Integer DataFrames produce exact round-trip via CSV export."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    result = pd.read_csv(buffer)

    # Integer columns should be exactly equal
    for col in df.columns:
        assert list(result[col]) == list(df[col]), (
            f"Exact mismatch in integer column '{col}'"
        )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 19: Cleaned Filename Formatting
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(basename=file_basename, ext=file_extension)
def test_cleaned_filename_has_prefix(basename, ext):
    """The cleaned filename always starts with 'cleaned_' prefix."""
    original = basename + ext
    result = cleaned_filename(original)
    assert result.startswith("cleaned_"), (
        f"Expected 'cleaned_' prefix, got: {result!r}"
    )


@settings(max_examples=100)
@given(basename=file_basename, ext=file_extension)
def test_cleaned_filename_ends_with_csv(basename, ext):
    """The cleaned filename always ends with '.csv'."""
    original = basename + ext
    result = cleaned_filename(original)
    assert result.endswith(".csv"), (
        f"Expected '.csv' extension, got: {result!r}"
    )


@settings(max_examples=100)
@given(basename=file_basename, ext=file_extension)
def test_cleaned_filename_strips_last_extension(basename, ext):
    """The cleaned filename strips the last extension from the original name."""
    original = basename + ext
    result = cleaned_filename(original)

    # The expected result: cleaned_ + basename (original without last extension) + .csv
    expected_base = os.path.splitext(original)[0]
    if not expected_base:
        expected_base = original
    expected = f"cleaned_{expected_base}.csv"

    assert result == expected, (
        f"For original={original!r}: expected {expected!r}, got {result!r}"
    )


@settings(max_examples=100)
@given(basename=file_basename)
def test_cleaned_filename_csv_extension_stripped(basename):
    """For .csv files specifically, the extension is stripped correctly."""
    original = basename + ".csv"
    result = cleaned_filename(original)
    expected = f"cleaned_{basename}.csv"
    assert result == expected, (
        f"For original={original!r}: expected {expected!r}, got {result!r}"
    )


@settings(max_examples=100)
@given(basename=file_basename, ext=file_extension)
def test_cleaned_filename_format_structure(basename, ext):
    """The cleaned filename follows the pattern: cleaned_<name_without_ext>.csv"""
    original = basename + ext
    result = cleaned_filename(original)

    # Must match: "cleaned_" + some_non_empty_content + ".csv"
    assert result.startswith("cleaned_")
    assert result.endswith(".csv")

    # The middle part (between 'cleaned_' and '.csv') should not be empty
    middle = result[len("cleaned_"):-len(".csv")]
    assert len(middle) > 0, (
        f"Middle part of cleaned filename is empty for original={original!r}"
    )
