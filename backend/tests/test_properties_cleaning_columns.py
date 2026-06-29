"""Property-based tests for column operations in the cleaning module."""

import string

import pandas as pd
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.cleaning import drop_columns, rename_column, convert_column_type


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid column names (simple ASCII identifiers)
column_name_st = st.text(
    alphabet=string.ascii_letters + string.digits + "_",
    min_size=1,
    max_size=20,
)

# Strategy for generating a DataFrame with at least 2 columns and some rows
@st.composite
def dataframe_with_columns(draw, min_cols=2, max_cols=8, min_rows=1, max_rows=20):
    """Generate a DataFrame with unique column names and integer data."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate unique column names
    col_names = draw(
        st.lists(
            column_name_st,
            min_size=n_cols,
            max_size=n_cols,
            unique=True,
        )
    )

    # Generate integer data for each column
    data = {}
    for col in col_names:
        data[col] = draw(
            st.lists(
                st.integers(min_value=-1000, max_value=1000),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    return pd.DataFrame(data)


@st.composite
def dataframe_and_drop_subset(draw):
    """Generate a DataFrame and a proper subset of its columns to drop."""
    df = draw(dataframe_with_columns(min_cols=2, max_cols=8, min_rows=1, max_rows=20))
    all_cols = list(df.columns)

    # Pick a proper subset (at least 1, at most len-1 columns to drop)
    max_drop = len(all_cols) - 1
    n_drop = draw(st.integers(min_value=1, max_value=max_drop))
    cols_to_drop = draw(
        st.lists(
            st.sampled_from(all_cols),
            min_size=n_drop,
            max_size=n_drop,
            unique=True,
        )
    )

    return df, cols_to_drop


@st.composite
def dataframe_and_rename(draw):
    """Generate a DataFrame, pick a column to rename, and generate a valid new name."""
    df = draw(dataframe_with_columns(min_cols=1, max_cols=8, min_rows=1, max_rows=20))
    all_cols = list(df.columns)

    # Pick a column to rename
    old_name = draw(st.sampled_from(all_cols))

    # Generate a new name: 1-128 chars, not already existing
    new_name = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=50,
        )
    )
    # Ensure new_name doesn't duplicate an existing column (unless it is old_name)
    assume(new_name not in all_cols or new_name == old_name)
    # Ensure new_name is different from old_name (otherwise rename is a no-op)
    assume(new_name != old_name)

    return df, old_name, new_name


@st.composite
def dataframe_with_int_column(draw):
    """Generate a DataFrame with at least one integer column for type conversion."""
    n_rows = draw(st.integers(min_value=1, max_value=20))

    # Create an integer column
    int_col_name = draw(column_name_st)
    int_data = draw(
        st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    # Optionally add extra columns
    extra_cols = draw(st.integers(min_value=0, max_value=3))
    data = {int_col_name: int_data}
    used_names = {int_col_name}

    for _ in range(extra_cols):
        col_name = draw(column_name_st)
        assume(col_name not in used_names)
        used_names.add(col_name)
        data[col_name] = draw(
            st.lists(
                st.integers(min_value=-100, max_value=100),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    return pd.DataFrame(data), int_col_name


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 11: Column Drop Preserves Remaining Data
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=dataframe_and_drop_subset())
def test_drop_columns_removes_dropped_columns(data):
    """After drop, none of the dropped columns appear in the result."""
    df, cols_to_drop = data
    result = drop_columns(df, cols_to_drop)

    for col in cols_to_drop:
        assert col not in result.columns, (
            f"Dropped column '{col}' still present in result"
        )


@settings(max_examples=100)
@given(data=dataframe_and_drop_subset())
def test_drop_columns_preserves_remaining_data(data):
    """After drop, all non-dropped columns retain their original data unchanged."""
    df, cols_to_drop = data
    result = drop_columns(df, cols_to_drop)

    remaining_cols = [c for c in df.columns if c not in cols_to_drop]
    for col in remaining_cols:
        assert col in result.columns, f"Expected column '{col}' not in result"
        pd.testing.assert_series_equal(
            result[col].reset_index(drop=True),
            df[col].reset_index(drop=True),
            check_names=False,
        )


@settings(max_examples=100)
@given(data=dataframe_and_drop_subset())
def test_drop_columns_preserves_row_count(data):
    """After drop, the row count remains the same as the original."""
    df, cols_to_drop = data
    result = drop_columns(df, cols_to_drop)

    assert len(result) == len(df), (
        f"Row count changed: {len(df)} -> {len(result)}"
    )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 12: Column Rename Preserves Data
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=dataframe_and_rename())
def test_rename_column_old_name_gone(data):
    """After rename, the old column name does not appear in the result."""
    df, old_name, new_name = data
    result = rename_column(df, old_name, new_name)

    assert old_name not in result.columns, (
        f"Old name '{old_name}' still present after rename"
    )


@settings(max_examples=100)
@given(data=dataframe_and_rename())
def test_rename_column_new_name_present(data):
    """After rename, the new column name appears in the result."""
    df, old_name, new_name = data
    result = rename_column(df, old_name, new_name)

    assert new_name in result.columns, (
        f"New name '{new_name}' not found in result columns"
    )


@settings(max_examples=100)
@given(data=dataframe_and_rename())
def test_rename_column_data_identical(data):
    """After rename, data in the renamed column is identical to the original."""
    df, old_name, new_name = data
    result = rename_column(df, old_name, new_name)

    pd.testing.assert_series_equal(
        result[new_name].reset_index(drop=True),
        df[old_name].reset_index(drop=True),
        check_names=False,
    )


@settings(max_examples=100)
@given(data=dataframe_and_rename())
def test_rename_column_other_columns_unchanged(data):
    """After rename, all other columns are unchanged."""
    df, old_name, new_name = data
    result = rename_column(df, old_name, new_name)

    other_cols = [c for c in df.columns if c != old_name]
    for col in other_cols:
        assert col in result.columns, f"Other column '{col}' missing from result"
        pd.testing.assert_series_equal(
            result[col].reset_index(drop=True),
            df[col].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 13: Column Type Conversion
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=dataframe_with_int_column())
def test_convert_int_to_float_dtype(data):
    """After converting an int column to float, the dtype is float."""
    df, col_name = data
    result = convert_column_type(df, col_name, "float")

    assert result[col_name].dtype == np.float64, (
        f"Expected float64 dtype, got {result[col_name].dtype}"
    )


@settings(max_examples=100)
@given(data=dataframe_with_int_column())
def test_convert_int_to_float_values_equivalent(data):
    """After converting int to float, values are numerically equivalent."""
    df, col_name = data
    result = convert_column_type(df, col_name, "float")

    for i in range(len(df)):
        original = df[col_name].iloc[i]
        converted = result[col_name].iloc[i]
        assert float(original) == converted, (
            f"Value mismatch at row {i}: {original} != {converted}"
        )


@settings(max_examples=100)
@given(data=dataframe_with_int_column())
def test_convert_int_to_string_dtype(data):
    """After converting an int column to string, the dtype is object."""
    df, col_name = data
    result = convert_column_type(df, col_name, "string")

    assert result[col_name].dtype == object, (
        f"Expected object dtype, got {result[col_name].dtype}"
    )


@settings(max_examples=100)
@given(data=dataframe_with_int_column())
def test_convert_int_to_string_values_equivalent(data):
    """After converting int to string, string values represent the original integers."""
    df, col_name = data
    result = convert_column_type(df, col_name, "string")

    for i in range(len(df)):
        original = df[col_name].iloc[i]
        converted = result[col_name].iloc[i]
        assert converted == str(original), (
            f"Value mismatch at row {i}: str({original}) != {converted!r}"
        )
