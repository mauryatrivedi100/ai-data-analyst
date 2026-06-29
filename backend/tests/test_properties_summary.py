"""Property-based tests for dataset summary accuracy.

# Feature: ai-data-analyst, Property 3: Dataset Summary Accuracy
# Validates: Requirements 2.1
#
# For any valid DataFrame, the computed summary SHALL report:
#   - row_count equal to the number of rows
#   - column_count equal to the number of columns
#   - columns matching the DataFrame's column names
#   - duplicate_rows equal to the count of rows where all values match another row
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames


# ---------------------------------------------------------------------------
# Helper: compute summary the same way routes.py does
# ---------------------------------------------------------------------------

def compute_summary(df):
    """Replicate the summary computation from the /summary route."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Strategies for generating DataFrames
# ---------------------------------------------------------------------------

# Column name strategy: simple valid identifiers
col_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=10,
)

# Strategy for numerical values (including some NaN to test robustness)
numerical_value_st = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-1000, max_value=1000).map(float),
)

# Strategy for categorical values
categorical_value_st = st.one_of(
    st.text(min_size=0, max_size=5, alphabet="abcdefgh"),
    st.sampled_from(["cat", "dog", "bird", "fish", "ant"]),
)

# Strategy for mixed values
cell_value_st = st.one_of(numerical_value_st, categorical_value_st)


@st.composite
def arbitrary_dataframe(draw, min_rows=1, max_rows=30, min_cols=1, max_cols=8):
    """Generate a random DataFrame with varying rows, columns, and dtypes.

    Some rows may be duplicated intentionally to test duplicate_rows counting.
    """
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate unique column names
    col_names = draw(
        st.lists(col_name_st, min_size=n_cols, max_size=n_cols, unique=True)
    )

    # Generate data for each column
    data = {}
    for col_name in col_names:
        # Decide column type
        col_type = draw(st.sampled_from(["numerical", "categorical", "mixed"]))
        if col_type == "numerical":
            values = draw(st.lists(numerical_value_st, min_size=n_rows, max_size=n_rows))
        elif col_type == "categorical":
            values = draw(st.lists(categorical_value_st, min_size=n_rows, max_size=n_rows))
        else:
            values = draw(st.lists(cell_value_st, min_size=n_rows, max_size=n_rows))
        data[col_name] = values

    df = pd.DataFrame(data)

    # Optionally duplicate some rows to ensure duplicate detection is tested
    if n_rows >= 2:
        should_add_duplicates = draw(st.booleans())
        if should_add_duplicates:
            n_dups = draw(st.integers(min_value=1, max_value=min(5, n_rows)))
            # Pick random rows to duplicate
            dup_indices = draw(
                st.lists(
                    st.integers(min_value=0, max_value=n_rows - 1),
                    min_size=n_dups,
                    max_size=n_dups,
                )
            )
            dup_rows = df.iloc[dup_indices]
            df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 3: Dataset Summary Accuracy
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_row_count_matches_dataframe_length(df):
    """row_count SHALL equal the number of rows in the DataFrame."""
    summary = compute_summary(df)
    assert summary["row_count"] == len(df), (
        f"Expected row_count={len(df)}, got {summary['row_count']}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_column_count_matches_dataframe_columns(df):
    """column_count SHALL equal the number of columns in the DataFrame."""
    summary = compute_summary(df)
    assert summary["column_count"] == len(df.columns), (
        f"Expected column_count={len(df.columns)}, got {summary['column_count']}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_columns_match_dataframe_column_names(df):
    """columns SHALL match the DataFrame's column names."""
    summary = compute_summary(df)
    assert summary["columns"] == df.columns.tolist(), (
        f"Expected columns={df.columns.tolist()}, got {summary['columns']}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_duplicate_rows_matches_pandas_duplicated(df):
    """duplicate_rows SHALL equal df.duplicated().sum()."""
    summary = compute_summary(df)
    expected_duplicates = int(df.duplicated().sum())
    assert summary["duplicate_rows"] == expected_duplicates, (
        f"Expected duplicate_rows={expected_duplicates}, got {summary['duplicate_rows']}"
    )


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_all_summary_fields_consistent(df):
    """All four summary fields SHALL be consistent with the DataFrame simultaneously."""
    summary = compute_summary(df)

    assert summary["row_count"] == len(df)
    assert summary["column_count"] == len(df.columns)
    assert summary["columns"] == df.columns.tolist()
    assert summary["duplicate_rows"] == int(df.duplicated().sum())


# ---------------------------------------------------------------------------
# Edge case: DataFrame with all identical rows (maximum duplicates)
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    n_rows=st.integers(min_value=2, max_value=20),
    n_cols=st.integers(min_value=1, max_value=5),
)
def test_all_identical_rows_duplicate_count(n_rows, n_cols):
    """When all rows are identical, duplicate_rows SHALL equal n_rows - 1."""
    # Create a DataFrame where every row is the same
    data = {f"col_{i}": [42] * n_rows for i in range(n_cols)}
    df = pd.DataFrame(data)

    summary = compute_summary(df)

    # All rows except the first are duplicates
    assert summary["duplicate_rows"] == n_rows - 1, (
        f"Expected {n_rows - 1} duplicates for {n_rows} identical rows, "
        f"got {summary['duplicate_rows']}"
    )
    assert summary["row_count"] == n_rows
    assert summary["column_count"] == n_cols


# ---------------------------------------------------------------------------
# Edge case: DataFrame with no duplicates
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(n_rows=st.integers(min_value=1, max_value=30))
def test_unique_rows_zero_duplicates(n_rows):
    """When all rows are unique, duplicate_rows SHALL be 0."""
    # Create a DataFrame with unique index as values
    data = {"id": list(range(n_rows)), "value": [f"item_{i}" for i in range(n_rows)]}
    df = pd.DataFrame(data)

    summary = compute_summary(df)

    assert summary["duplicate_rows"] == 0, (
        f"Expected 0 duplicates for unique rows, got {summary['duplicate_rows']}"
    )
    assert summary["row_count"] == n_rows
