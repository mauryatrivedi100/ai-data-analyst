"""Property-based tests for encoding and scaling operations."""

import numpy as np
import pandas as pd

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.cleaning import label_encode, one_hot_encode, standard_scale, min_max_scale


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating category values (short alphabetic strings)
category_value = st.text(
    alphabet=st.characters(whitelist_categories=("L",)),
    min_size=1,
    max_size=10,
)

# Strategy for generating a list of unique category values (at least 2)
unique_categories = st.lists(
    category_value,
    min_size=2,
    max_size=15,
    unique=True,
)

# Strategy for generating unique categories limited to <=50 for one-hot encoding
unique_categories_ohe = st.lists(
    category_value,
    min_size=2,
    max_size=10,
    unique=True,
)


def categorical_column_strategy(min_rows=5, max_rows=50):
    """Generate a DataFrame with a single categorical column of known categories."""
    @st.composite
    def _build(draw):
        categories = draw(unique_categories)
        num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
        # Sample from the categories for each row (no NaN)
        values = draw(
            st.lists(
                st.sampled_from(categories),
                min_size=num_rows,
                max_size=num_rows,
            )
        )
        df = pd.DataFrame({"category": values})
        return df, categories
    return _build()


def categorical_column_ohe_strategy(min_rows=5, max_rows=50):
    """Generate a DataFrame with a categorical column suitable for one-hot encoding."""
    @st.composite
    def _build(draw):
        categories = draw(unique_categories_ohe)
        num_rows = draw(st.integers(min_value=max(len(categories), min_rows), max_value=max_rows))
        # Ensure all categories appear at least once
        values = list(categories)  # ensure each appears at least once
        remaining = num_rows - len(categories)
        if remaining > 0:
            extra = draw(
                st.lists(
                    st.sampled_from(categories),
                    min_size=remaining,
                    max_size=remaining,
                )
            )
            values.extend(extra)
        # Shuffle via hypothesis
        shuffled = draw(st.permutations(values))
        df = pd.DataFrame({"color": shuffled})
        return df, categories
    return _build()


def numerical_column_strategy(min_rows=5, max_rows=50):
    """Generate a DataFrame with a numerical column with at least 2 distinct values."""
    @st.composite
    def _build(draw):
        num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
        values = draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=num_rows,
                max_size=num_rows,
            )
        )
        # Ensure at least 2 distinct values
        assume(len(set(values)) >= 2)
        df = pd.DataFrame({"numeric_col": values})
        return df
    return _build()


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 14: Label Encoding Alphabetical Mapping
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=categorical_column_strategy())
def test_label_encode_unique_integers(data):
    """Each unique category maps to a unique integer after label encoding."""
    df, categories = data
    result = label_encode(df, "category")

    # Count unique categories actually present in the DataFrame
    actual_unique_categories = df["category"].nunique()
    encoded_values = result["category"].dropna().unique()
    # Number of unique encoded values equals number of unique categories in the data
    assert len(encoded_values) == actual_unique_categories


@settings(max_examples=100)
@given(data=categorical_column_strategy())
def test_label_encode_alphabetical_order(data):
    """Integer assignments follow alphabetical ordering: first alphabetically → 0."""
    df, categories = data
    result = label_encode(df, "category")

    sorted_categories = sorted(set(df["category"].dropna()), key=str)
    # Build the expected mapping
    expected_mapping = {cat: idx for idx, cat in enumerate(sorted_categories)}

    # Verify each row's encoded value matches the expected mapping
    for i, row in df.iterrows():
        original_val = row["category"]
        encoded_val = result.at[i, "category"]
        assert encoded_val == expected_mapping[original_val], (
            f"Row {i}: category '{original_val}' expected code {expected_mapping[original_val]}, "
            f"got {encoded_val}"
        )


@settings(max_examples=100)
@given(data=categorical_column_strategy())
def test_label_encode_same_category_same_int(data):
    """Rows with the same original category have the same integer value."""
    df, categories = data
    result = label_encode(df, "category")

    # Group by original values and check encoded values are consistent
    for cat in df["category"].unique():
        mask = df["category"] == cat
        encoded_vals = result.loc[mask, "category"].unique()
        assert len(encoded_vals) == 1, (
            f"Category '{cat}' maps to multiple encoded values: {encoded_vals}"
        )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 15: One-Hot Encoding Binary Decomposition
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=categorical_column_ohe_strategy())
def test_one_hot_encode_original_column_removed(data):
    """After one-hot encoding, the original column is removed."""
    df, categories = data
    result = one_hot_encode(df, "color")

    assert "color" not in result.columns, "Original column 'color' should be removed"


@settings(max_examples=100)
@given(data=categorical_column_ohe_strategy())
def test_one_hot_encode_binary_columns_created(data):
    """Exactly N new binary columns are created for N unique categories."""
    df, categories = data
    result = one_hot_encode(df, "color")

    unique_vals = df["color"].nunique()
    new_cols = [c for c in result.columns if c.startswith("color_")]
    assert len(new_cols) == unique_vals, (
        f"Expected {unique_vals} new columns, got {len(new_cols)}: {new_cols}"
    )

    # All values in new columns are 0 or 1
    for col in new_cols:
        vals = result[col].unique()
        assert set(vals).issubset({0, 1}), (
            f"Column '{col}' has non-binary values: {vals}"
        )


@settings(max_examples=100)
@given(data=categorical_column_ohe_strategy())
def test_one_hot_encode_exactly_one_per_row(data):
    """Each row has exactly one 1 and (N-1) zeros across the new columns."""
    df, categories = data
    result = one_hot_encode(df, "color")

    new_cols = [c for c in result.columns if c.startswith("color_")]
    for idx in range(len(result)):
        row_sum = sum(result.at[idx, col] for col in new_cols)
        assert row_sum == 1, (
            f"Row {idx} has sum {row_sum} across one-hot columns (expected 1)"
        )


@settings(max_examples=100)
@given(data=categorical_column_ohe_strategy())
def test_one_hot_encode_correct_column_has_one(data):
    """The column with value 1 corresponds to the row's original category value."""
    df, categories = data
    result = one_hot_encode(df, "color")

    new_cols = [c for c in result.columns if c.startswith("color_")]
    for idx in range(len(df)):
        original_value = df.at[idx, "color"]
        expected_col = f"color_{original_value}"
        assert expected_col in new_cols, (
            f"Expected column '{expected_col}' not found in result columns"
        )
        assert result.at[idx, expected_col] == 1, (
            f"Row {idx}: column '{expected_col}' should be 1 for original value '{original_value}'"
        )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 16: Standard Scaling Zero-Mean Unit-Variance
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=numerical_column_strategy())
def test_standard_scale_zero_mean(data):
    """After standard scaling, the column mean is approximately 0."""
    df = data
    result = standard_scale(df, "numeric_col")

    mean = result["numeric_col"].mean()
    assert abs(mean) < 1e-10, (
        f"Mean after standard scaling is {mean}, expected ≈ 0"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy())
def test_standard_scale_unit_variance(data):
    """After standard scaling, the column std (ddof=0) is approximately 1."""
    df = data
    result = standard_scale(df, "numeric_col")

    std = result["numeric_col"].std(ddof=0)
    assert abs(std - 1.0) < 1e-10, (
        f"Std (ddof=0) after standard scaling is {std}, expected ≈ 1"
    )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 17: Min-Max Scaling Range
# Validates: Requirements 7.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=numerical_column_strategy())
def test_min_max_scale_values_in_range(data):
    """After min-max scaling, all values are in [0, 1]."""
    df = data
    result = min_max_scale(df, "numeric_col")

    col_values = result["numeric_col"]
    assert col_values.min() >= 0.0 - 1e-15, (
        f"Min value {col_values.min()} is below 0"
    )
    assert col_values.max() <= 1.0 + 1e-15, (
        f"Max value {col_values.max()} is above 1"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy())
def test_min_max_scale_min_is_zero(data):
    """After min-max scaling, the minimum value is 0.0."""
    df = data
    result = min_max_scale(df, "numeric_col")

    min_val = result["numeric_col"].min()
    assert abs(min_val - 0.0) < 1e-15, (
        f"Min value after min-max scaling is {min_val}, expected 0.0"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy())
def test_min_max_scale_max_is_one(data):
    """After min-max scaling, the maximum value is 1.0."""
    df = data
    result = min_max_scale(df, "numeric_col")

    max_val = result["numeric_col"].max()
    assert abs(max_val - 1.0) < 1e-15, (
        f"Max value after min-max scaling is {max_val}, expected 1.0"
    )
