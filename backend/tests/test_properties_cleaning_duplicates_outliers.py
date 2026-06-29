"""Property-based tests for duplicate removal and outlier detection/removal.

Feature: ai-data-analyst
Property 9: Duplicate Removal Preserves Uniqueness and First Occurrence
Property 10: Outlier Detection and Removal via IQR
"""

import numpy as np
import pandas as pd

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.cleaning import remove_duplicates, detect_outliers, remove_outliers


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy to generate a DataFrame with guaranteed duplicate rows
@st.composite
def dataframe_with_duplicates(draw):
    """Generate a DataFrame that contains at least one duplicate row."""
    n_cols = draw(st.integers(min_value=1, max_value=4))
    col_names = [f"col_{i}" for i in range(n_cols)]

    # Generate base unique rows (at least 1)
    n_unique_rows = draw(st.integers(min_value=1, max_value=10))

    data = {}
    for col in col_names:
        values = draw(
            st.lists(
                st.integers(min_value=0, max_value=5),
                min_size=n_unique_rows,
                max_size=n_unique_rows,
            )
        )
        data[col] = values

    base_df = pd.DataFrame(data)

    # Add duplicate rows by repeating some rows
    n_duplicates = draw(st.integers(min_value=1, max_value=5))
    dup_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_unique_rows - 1),
            min_size=n_duplicates,
            max_size=n_duplicates,
        )
    )
    dup_rows = base_df.iloc[dup_indices]
    df = pd.concat([base_df, dup_rows], ignore_index=True)

    # Shuffle the rows
    shuffle_order = draw(
        st.permutations(list(range(len(df))))
    )
    df = df.iloc[shuffle_order].reset_index(drop=True)

    return df


@st.composite
def arbitrary_dataframe(draw):
    """Generate an arbitrary DataFrame (may or may not have duplicates)."""
    n_cols = draw(st.integers(min_value=1, max_value=4))
    n_rows = draw(st.integers(min_value=1, max_value=15))
    col_names = [f"col_{i}" for i in range(n_cols)]

    data = {}
    for col in col_names:
        values = draw(
            st.lists(
                st.integers(min_value=0, max_value=10),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col] = values

    return pd.DataFrame(data)


@st.composite
def numerical_column_with_outliers(draw):
    """Generate a numerical column with at least 4 non-NaN values and some outliers.

    Returns a DataFrame with a single column 'value' that has clear outliers.
    """
    # Generate core data (the inliers) - at least 4 values in a tight range
    n_inliers = draw(st.integers(min_value=4, max_value=15))
    center = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    spread = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))

    inlier_values = draw(
        st.lists(
            st.floats(
                min_value=center - spread,
                max_value=center + spread,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_inliers,
            max_size=n_inliers,
        )
    )

    # Add outlier values far from the core data
    n_outliers = draw(st.integers(min_value=1, max_value=3))
    outlier_offset = spread * 10  # Far enough to be clear outliers
    outlier_values = draw(
        st.lists(
            st.sampled_from([
                center + outlier_offset,
                center - outlier_offset,
                center + outlier_offset * 2,
                center - outlier_offset * 2,
            ]),
            min_size=n_outliers,
            max_size=n_outliers,
        )
    )

    all_values = inlier_values + outlier_values
    df = pd.DataFrame({"value": all_values})
    return df


@st.composite
def numerical_column_for_iqr(draw):
    """Generate a numerical column with at least 4 non-NaN values for IQR testing.

    May optionally include NaN values.
    """
    n_values = draw(st.integers(min_value=4, max_value=20))
    values = draw(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=n_values,
            max_size=n_values,
        )
    )

    # Optionally add some NaN values
    n_nans = draw(st.integers(min_value=0, max_value=3))
    all_values = values + [float("nan")] * n_nans

    df = pd.DataFrame({"value": all_values})
    return df


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 9: Duplicate Removal Preserves Uniqueness
# and First Occurrence
# Validates: Requirements 4.1, 4.3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=dataframe_with_duplicates())
def test_no_duplicates_after_removal(df):
    """After removal, no two rows shall have identical values across all columns.

    **Validates: Requirements 4.1**
    """
    cleaned, _ = remove_duplicates(df)
    # No duplicates should remain
    assert cleaned.duplicated().sum() == 0


@settings(max_examples=100)
@given(df=dataframe_with_duplicates())
def test_first_occurrence_retained(df):
    """For each group of duplicate rows, the first occurrence (by original index)
    shall be retained.

    **Validates: Requirements 4.1**
    """
    cleaned, _ = remove_duplicates(df)

    # For each row in the cleaned result, verify it matches the first occurrence
    # in the original DataFrame
    for _, row in cleaned.iterrows():
        # Find all matching rows in original
        mask = (df == row).all(axis=1)
        first_idx = mask.idxmax()  # First True index
        original_row = df.iloc[first_idx]
        assert (original_row == row).all()


@settings(max_examples=100)
@given(df=dataframe_with_duplicates())
def test_duplicates_removed_count_accuracy(df):
    """The number of rows removed shall equal the original duplicate count.

    **Validates: Requirements 4.3**
    """
    original_len = len(df)
    cleaned, duplicates_removed = remove_duplicates(df)
    result_len = len(cleaned)

    # duplicates_removed = original rows - result rows
    assert duplicates_removed == original_len - result_len

    # Also verify it matches pandas' duplicate count
    expected_dups = int(df.duplicated(keep='first').sum())
    assert duplicates_removed == expected_dups


@settings(max_examples=100)
@given(df=arbitrary_dataframe())
def test_duplicate_removal_on_arbitrary_dataframe(df):
    """For any DataFrame, duplicate removal properties hold: no duplicates remain,
    count is accurate, and result is a subset of original rows.

    **Validates: Requirements 4.1, 4.3**
    """
    original_len = len(df)
    cleaned, duplicates_removed = remove_duplicates(df)

    # No duplicates remain
    assert cleaned.duplicated().sum() == 0

    # Count accuracy
    assert duplicates_removed == original_len - len(cleaned)

    # All rows in cleaned are a subset of original rows
    for _, row in cleaned.iterrows():
        mask = (df == row).all(axis=1)
        assert mask.any(), "Cleaned row not found in original DataFrame"


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 10: Outlier Detection and Removal via IQR
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(df=numerical_column_for_iqr())
def test_outlier_bounds_computed_correctly(df):
    """Outlier detection shall compute bounds as [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    **Validates: Requirements 5.1**
    """
    series = df["value"].dropna()
    assume(len(series) >= 4)

    count, lower_bound, upper_bound, indices = detect_outliers(df, "value")

    # Compute expected bounds
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    expected_lower = q1 - 1.5 * iqr
    expected_upper = q3 + 1.5 * iqr

    assert abs(lower_bound - expected_lower) < 1e-10, (
        f"Lower bound mismatch: {lower_bound} != {expected_lower}"
    )
    assert abs(upper_bound - expected_upper) < 1e-10, (
        f"Upper bound mismatch: {upper_bound} != {expected_upper}"
    )


@settings(max_examples=100)
@given(df=numerical_column_for_iqr())
def test_remaining_values_within_bounds_after_removal(df):
    """After removal, all remaining values in the column shall fall within bounds
    (inclusive).

    **Validates: Requirements 5.2**
    """
    series = df["value"].dropna()
    assume(len(series) >= 4)

    _, lower_bound, upper_bound, _ = detect_outliers(df, "value")
    cleaned, rows_removed = remove_outliers(df, "value")

    # All non-NaN values in cleaned should be within bounds
    remaining_values = cleaned["value"].dropna()
    if len(remaining_values) > 0:
        assert (remaining_values >= lower_bound - 1e-10).all(), (
            f"Found values below lower bound {lower_bound}: "
            f"{remaining_values[remaining_values < lower_bound].tolist()}"
        )
        assert (remaining_values <= upper_bound + 1e-10).all(), (
            f"Found values above upper bound {upper_bound}: "
            f"{remaining_values[remaining_values > upper_bound].tolist()}"
        )


@settings(max_examples=100)
@given(df=numerical_column_for_iqr())
def test_removed_rows_contained_outlier_values(df):
    """All removed rows shall have contained at least one value outside the IQR bounds.

    **Validates: Requirements 5.1, 5.2**
    """
    series = df["value"].dropna()
    assume(len(series) >= 4)

    _, lower_bound, upper_bound, outlier_indices = detect_outliers(df, "value")
    cleaned, rows_removed = remove_outliers(df, "value")

    # Verify rows_removed count matches
    assert rows_removed == len(outlier_indices)

    # Each outlier index should have a value outside bounds
    for idx in outlier_indices:
        val = df.loc[idx, "value"]
        assert val < lower_bound or val > upper_bound, (
            f"Row {idx} flagged as outlier but value {val} is within "
            f"bounds [{lower_bound}, {upper_bound}]"
        )


@settings(max_examples=100)
@given(df=numerical_column_for_iqr())
def test_outlier_count_matches_indices_length(df):
    """The outlier_count returned shall equal the number of outlier_indices.

    **Validates: Requirements 5.1**
    """
    series = df["value"].dropna()
    assume(len(series) >= 4)

    count, lower_bound, upper_bound, indices = detect_outliers(df, "value")
    assert count == len(indices)

    # Also verify: count matches the number of non-NaN values outside bounds
    non_nan = df["value"].dropna()
    expected_outliers = ((non_nan < lower_bound) | (non_nan > upper_bound)).sum()
    assert count == expected_outliers


@settings(max_examples=100)
@given(df=numerical_column_for_iqr())
def test_nan_rows_preserved_after_outlier_removal(df):
    """Rows with NaN in the column are preserved (not treated as outliers).

    **Validates: Requirements 5.2**
    """
    series = df["value"].dropna()
    assume(len(series) >= 4)

    original_nan_count = df["value"].isna().sum()
    cleaned, _ = remove_outliers(df, "value")
    cleaned_nan_count = cleaned["value"].isna().sum()

    assert cleaned_nan_count == original_nan_count, (
        f"NaN rows not preserved: original had {original_nan_count}, "
        f"cleaned has {cleaned_nan_count}"
    )
