"""Property-based tests for missing value operations in cleaning.py.

Tests Properties 6, 7, and 8 from the design document.
Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.7
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.cleaning import (
    remove_missing_rows,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def dataframe_with_some_nan_rows(min_rows=3, max_rows=30, min_cols=1, max_cols=5):
    """Generate a DataFrame where at least one row is fully non-NaN and at least one has NaN."""
    @st.composite
    def strategy(draw):
        n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
        n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
        col_names = [f"col_{i}" for i in range(n_cols)]

        # Generate base data as floats
        data = {}
        for col in col_names:
            values = draw(st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=n_rows, max_size=n_rows,
            ))
            data[col] = values

        df = pd.DataFrame(data)

        # Ensure at least one row is clean (no NaN) — keep row 0 clean
        # Inject NaN into some other rows (at least 1 row must have NaN)
        nan_row_indices = draw(st.lists(
            st.integers(min_value=1, max_value=n_rows - 1),
            min_size=1, max_size=max(1, n_rows - 2),
            unique=True,
        ))

        for idx in nan_row_indices:
            # Pick at least one column to set NaN
            nan_col_idx = draw(st.integers(min_value=0, max_value=n_cols - 1))
            df.iloc[idx, nan_col_idx] = np.nan

        return df

    return strategy()


def numerical_column_with_nan(min_size=3, max_size=30):
    """Generate a DataFrame with a numerical column that has some NaN and some non-NaN values."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(min_value=min_size, max_value=max_size))

        # Generate non-NaN values
        non_nan_values = draw(st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=size,
        ))

        # Determine how many NaN values to inject (at least 1)
        max_nan = size - len(non_nan_values)
        if max_nan <= 0:
            # Truncate non_nan_values and add at least 1 NaN
            non_nan_values = non_nan_values[:size - 1]
            nan_count = 1
        else:
            nan_count = draw(st.integers(min_value=1, max_value=max(1, max_nan)))

        all_values = non_nan_values + [np.nan] * nan_count

        # Shuffle
        indices = draw(st.permutations(range(len(all_values))))
        shuffled = [all_values[i] for i in indices]

        df = pd.DataFrame({"target": shuffled})
        return df

    return strategy()


def categorical_column_with_nan(min_size=3, max_size=20):
    """Generate a DataFrame with a categorical (object) column that has some NaN."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(min_value=min_size, max_value=max_size))

        # Generate categorical values
        categories = draw(st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",)),
                min_size=1,
                max_size=5,
            ),
            min_size=1, max_size=min(5, size),
        ))

        # Build values from categories
        values = draw(st.lists(
            st.sampled_from(categories),
            min_size=max(2, size - 2),
            max_size=size - 1,
        ))

        # Add at least one None/NaN
        nan_count = draw(st.integers(min_value=1, max_value=max(1, size - len(values))))
        all_values = values + [None] * nan_count

        # Shuffle
        indices = draw(st.permutations(range(len(all_values))))
        shuffled = [all_values[i] for i in indices]

        df = pd.DataFrame({"target": shuffled})
        return df

    return strategy()


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 6: Missing Value Row Removal
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=dataframe_with_some_nan_rows())
def test_remove_missing_rows_no_nan_remaining(df):
    """After removing rows with NaN, the result contains zero NaN values."""
    cleaned, _ = remove_missing_rows(df)
    assert cleaned.isna().sum().sum() == 0, "NaN values remain after row removal"


@settings(max_examples=100)
@given(df=dataframe_with_some_nan_rows())
def test_remove_missing_rows_subset_integrity(df):
    """All rows in the cleaned result exist in the original DataFrame."""
    cleaned, _ = remove_missing_rows(df)

    # Each remaining row must correspond to a row in the original
    for _, row in cleaned.iterrows():
        # Find matching rows in original
        match = df[(df == row).all(axis=1)]
        assert len(match) > 0, f"Row {row.to_dict()} not found in original DataFrame"


@settings(max_examples=100)
@given(df=dataframe_with_some_nan_rows())
def test_remove_missing_rows_count_accuracy(df):
    """rows_removed equals original row count minus result row count."""
    cleaned, rows_removed = remove_missing_rows(df)

    # Count rows that have at least one NaN
    rows_with_nan = df.isna().any(axis=1).sum()
    assert rows_removed == rows_with_nan, (
        f"Expected rows_removed={rows_with_nan}, got {rows_removed}"
    )
    assert rows_removed == len(df) - len(cleaned), (
        f"rows_removed ({rows_removed}) != original ({len(df)}) - result ({len(cleaned)})"
    )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 7: Missing Value Imputation Preserves Non-Missing and Eliminates NaN
# Validates: Requirements 3.2, 3.3, 3.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_mean_no_nan_remaining(df):
    """After fill_missing_mean, no NaN values remain in the target column."""
    cleaned, _ = fill_missing_mean(df, "target")
    assert cleaned["target"].isna().sum() == 0, "NaN remains after mean imputation"


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_mean_non_missing_unchanged(df):
    """Originally non-NaN values are unchanged after mean imputation."""
    cleaned, _ = fill_missing_mean(df, "target")

    original_non_nan_mask = ~df["target"].isna()
    original_values = df.loc[original_non_nan_mask, "target"].values
    cleaned_values = cleaned.loc[original_non_nan_mask, "target"].values

    np.testing.assert_array_almost_equal(
        cleaned_values, original_values,
        err_msg="Non-NaN values changed after mean imputation"
    )


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_mean_value_correctness(df):
    """The fill value equals the arithmetic mean of original non-NaN values."""
    cleaned, mean_value = fill_missing_mean(df, "target")

    expected_mean = df["target"].dropna().mean()
    assert abs(mean_value - expected_mean) < 1e-10, (
        f"Fill mean {mean_value} != expected {expected_mean}"
    )


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_median_no_nan_remaining(df):
    """After fill_missing_median, no NaN values remain in the target column."""
    cleaned, _ = fill_missing_median(df, "target")
    assert cleaned["target"].isna().sum() == 0, "NaN remains after median imputation"


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_median_non_missing_unchanged(df):
    """Originally non-NaN values are unchanged after median imputation."""
    cleaned, _ = fill_missing_median(df, "target")

    original_non_nan_mask = ~df["target"].isna()
    original_values = df.loc[original_non_nan_mask, "target"].values
    cleaned_values = cleaned.loc[original_non_nan_mask, "target"].values

    np.testing.assert_array_almost_equal(
        cleaned_values, original_values,
        err_msg="Non-NaN values changed after median imputation"
    )


@settings(max_examples=100)
@given(df=numerical_column_with_nan())
def test_fill_median_value_correctness(df):
    """The fill value equals the median of original non-NaN values."""
    cleaned, median_value = fill_missing_median(df, "target")

    expected_median = df["target"].dropna().median()
    assert abs(median_value - expected_median) < 1e-10, (
        f"Fill median {median_value} != expected {expected_median}"
    )


@settings(max_examples=100)
@given(df=categorical_column_with_nan())
def test_fill_mode_no_nan_remaining(df):
    """After fill_missing_mode, no NaN values remain in the target column."""
    cleaned, _ = fill_missing_mode(df, "target")
    assert cleaned["target"].isna().sum() == 0, "NaN remains after mode imputation"


@settings(max_examples=100)
@given(df=categorical_column_with_nan())
def test_fill_mode_non_missing_unchanged(df):
    """Originally non-NaN values are unchanged after mode imputation."""
    cleaned, _ = fill_missing_mode(df, "target")

    original_non_nan_mask = ~df["target"].isna()
    original_values = df.loc[original_non_nan_mask, "target"].values
    cleaned_values = cleaned.loc[original_non_nan_mask, "target"].values

    assert list(cleaned_values) == list(original_values), (
        "Non-NaN values changed after mode imputation"
    )


@settings(max_examples=100)
@given(df=categorical_column_with_nan())
def test_fill_mode_value_correctness(df):
    """The fill value equals the mode of original non-NaN values."""
    cleaned, mode_value = fill_missing_mode(df, "target")

    expected_mode = df["target"].mode().iloc[0]
    assert mode_value == expected_mode, (
        f"Fill mode {mode_value!r} != expected {expected_mode!r}"
    )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 8: Mean/Median Rejection for Categorical Columns
# Validates: Requirements 3.7
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=categorical_column_with_nan())
def test_fill_mean_rejects_categorical_column(df):
    """Calling fill_missing_mean on a non-numeric column raises ValueError."""
    # Ensure the column is truly categorical (object dtype)
    assume(not pd.api.types.is_numeric_dtype(df["target"]))

    original_df = df.copy()

    try:
        fill_missing_mean(df, "target")
        # If it doesn't raise, the test fails
        assert False, "fill_missing_mean did not raise ValueError for categorical column"
    except ValueError:
        pass

    # DataFrame should remain unchanged
    pd.testing.assert_frame_equal(df, original_df)


@settings(max_examples=100)
@given(df=categorical_column_with_nan())
def test_fill_median_rejects_categorical_column(df):
    """Calling fill_missing_median on a non-numeric column raises ValueError."""
    # Ensure the column is truly categorical (object dtype)
    assume(not pd.api.types.is_numeric_dtype(df["target"]))

    original_df = df.copy()

    try:
        fill_missing_median(df, "target")
        # If it doesn't raise, the test fails
        assert False, "fill_missing_median did not raise ValueError for categorical column"
    except ValueError:
        pass

    # DataFrame should remain unchanged
    pd.testing.assert_frame_equal(df, original_df)
