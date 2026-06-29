"""Property-based tests for EDA computations (pie chart and correlation heatmap)."""

import numpy as np
import pandas as pd

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.eda import compute_pie, compute_correlation_heatmap


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating categorical column values
category_values = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_- "),
    min_size=1,
    max_size=10,
)

# Strategy for generating a categorical column (list of category strings)
categorical_column_strategy = st.lists(
    category_values,
    min_size=1,
    max_size=100,
)

# Strategy for generating numerical columns with finite values
finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def numerical_dataframe_strategy(draw):
    """Generate a DataFrame with at least 2 numerical columns and at least 3 rows."""
    n_cols = draw(st.integers(min_value=2, max_value=5))
    n_rows = draw(st.integers(min_value=3, max_value=50))

    data = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        values = draw(
            st.lists(finite_floats, min_size=n_rows, max_size=n_rows)
        )
        data[col_name] = values

    df = pd.DataFrame(data)

    # Ensure at least 2 distinct values per column to avoid constant columns
    # (which would produce NaN correlations)
    for col in df.columns:
        assume(df[col].nunique() >= 2)

    return df


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 20: Pie Chart Proportional Completeness
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(values=categorical_column_strategy)
def test_pie_chart_one_slice_per_unique_value(values):
    """Pie chart has exactly one slice per unique non-null value."""
    df = pd.DataFrame({"category": values})
    result = compute_pie(df, "category")

    unique_values = set(df["category"].dropna().astype(str))
    slice_names = {item["name"] for item in result}

    assert slice_names == unique_values, (
        f"Expected slices for {unique_values}, got {slice_names}"
    )


@settings(max_examples=100)
@given(values=categorical_column_strategy)
def test_pie_chart_counts_match_actual_occurrences(values):
    """Each slice's count matches the actual occurrence count of that category."""
    df = pd.DataFrame({"category": values})
    result = compute_pie(df, "category")

    # Build expected counts from the data
    expected_counts = df["category"].dropna().value_counts()

    for item in result:
        name = item["name"]
        count = item["value"]
        assert count == expected_counts[name], (
            f"Slice '{name}' has count {count}, expected {expected_counts[name]}"
        )


@settings(max_examples=100)
@given(values=categorical_column_strategy)
def test_pie_chart_sum_equals_total_non_null_rows(values):
    """Sum of all slice counts equals total number of non-null rows."""
    df = pd.DataFrame({"category": values})
    result = compute_pie(df, "category")

    total_slices = sum(item["value"] for item in result)
    total_non_null = df["category"].dropna().shape[0]

    assert total_slices == total_non_null, (
        f"Sum of slices {total_slices} != total non-null rows {total_non_null}"
    )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 21: Correlation Matrix Mathematical Invariants
# Validates: Requirements 9.6
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=numerical_dataframe_strategy())
def test_correlation_matrix_is_symmetric(df):
    """Correlation matrix satisfies matrix[i][j] == matrix[j][i]."""
    result = compute_correlation_heatmap(df)
    matrix = result["matrix"]
    n = len(matrix)

    for i in range(n):
        for j in range(n):
            assert np.isclose(matrix[i][j], matrix[j][i], atol=1e-10), (
                f"Matrix not symmetric at [{i}][{j}]: "
                f"{matrix[i][j]} != {matrix[j][i]}"
            )


@settings(max_examples=100)
@given(df=numerical_dataframe_strategy())
def test_correlation_matrix_diagonal_is_one(df):
    """Correlation matrix has 1.0 on the diagonal."""
    result = compute_correlation_heatmap(df)
    matrix = result["matrix"]
    n = len(matrix)

    for i in range(n):
        assert np.isclose(matrix[i][i], 1.0, atol=1e-10), (
            f"Diagonal [{i}][{i}] = {matrix[i][i]}, expected 1.0"
        )


@settings(max_examples=100)
@given(df=numerical_dataframe_strategy())
def test_correlation_matrix_off_diagonal_in_range(df):
    """All off-diagonal values are in the range [-1.0, 1.0]."""
    result = compute_correlation_heatmap(df)
    matrix = result["matrix"]
    n = len(matrix)

    for i in range(n):
        for j in range(n):
            if i != j:
                assert -1.0 <= matrix[i][j] <= 1.0, (
                    f"Off-diagonal [{i}][{j}] = {matrix[i][j]} "
                    f"is outside [-1.0, 1.0]"
                )
