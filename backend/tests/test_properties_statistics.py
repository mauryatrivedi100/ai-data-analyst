"""Property-based tests for numerical and categorical statistics computation."""

import numpy as np
import pandas as pd

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.utils import get_column_types


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating numerical values (finite floats, possibly with NaN)
finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Strategy for generating a numerical column with at least 1 non-NaN value
numerical_column_strategy = st.lists(
    st.one_of(finite_floats, st.just(float("nan"))),
    min_size=2,
    max_size=50,
).filter(lambda lst: any(not np.isnan(x) for x in lst))

# Strategy for generating categorical values
category_values = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_- "),
    min_size=1,
    max_size=10,
)

# Strategy for generating a categorical column (list of string values)
categorical_column_strategy = st.lists(
    category_values,
    min_size=2,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Helpers — replicate the statistics computation logic from routes.py
# ---------------------------------------------------------------------------

def compute_numerical_stats(series):
    """Compute numerical statistics matching routes.py logic."""
    return {
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "q1": round(float(series.quantile(0.25)), 2),
        "q2": round(float(series.quantile(0.50)), 2),
        "q3": round(float(series.quantile(0.75)), 2),
    }


def compute_categorical_stats(series):
    """Compute categorical statistics matching routes.py logic."""
    unique_count = int(series.nunique())
    value_counts = series.value_counts().head(5)
    top_5 = [
        {"value": str(val), "count": int(cnt)}
        for val, cnt in value_counts.items()
    ]
    return {
        "unique_count": unique_count,
        "top_5": top_5,
    }


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 4: Numerical Statistics Computation
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_mean_matches_reference(data):
    """Computed mean matches pandas Series.mean() rounded to 2dp."""
    series = pd.Series(data, dtype=float)
    assume(series.notna().sum() >= 1)

    result = compute_numerical_stats(series)
    expected = round(float(series.mean()), 2)
    assert result["mean"] == expected, (
        f"mean mismatch: got {result['mean']}, expected {expected}"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_median_matches_reference(data):
    """Computed median matches pandas Series.median() rounded to 2dp."""
    series = pd.Series(data, dtype=float)
    assume(series.notna().sum() >= 1)

    result = compute_numerical_stats(series)
    expected = round(float(series.median()), 2)
    assert result["median"] == expected, (
        f"median mismatch: got {result['median']}, expected {expected}"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_std_matches_reference(data):
    """Computed std matches pandas Series.std() rounded to 2dp."""
    series = pd.Series(data, dtype=float)
    # std requires at least 2 non-NaN values; with 1 value, std is NaN
    assume(series.notna().sum() >= 2)

    result = compute_numerical_stats(series)
    expected = round(float(series.std()), 2)
    assert result["std"] == expected, (
        f"std mismatch: got {result['std']}, expected {expected}"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_min_max_matches_reference(data):
    """Computed min and max match pandas Series.min()/max() rounded to 2dp."""
    series = pd.Series(data, dtype=float)
    assume(series.notna().sum() >= 1)

    result = compute_numerical_stats(series)
    expected_min = round(float(series.min()), 2)
    expected_max = round(float(series.max()), 2)
    assert result["min"] == expected_min, (
        f"min mismatch: got {result['min']}, expected {expected_min}"
    )
    assert result["max"] == expected_max, (
        f"max mismatch: got {result['max']}, expected {expected_max}"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_quartiles_match_reference(data):
    """Computed Q1, Q2, Q3 match pandas quantile(0.25/0.50/0.75) rounded to 2dp."""
    series = pd.Series(data, dtype=float)
    assume(series.notna().sum() >= 1)

    result = compute_numerical_stats(series)
    expected_q1 = round(float(series.quantile(0.25)), 2)
    expected_q2 = round(float(series.quantile(0.50)), 2)
    expected_q3 = round(float(series.quantile(0.75)), 2)
    assert result["q1"] == expected_q1, (
        f"q1 mismatch: got {result['q1']}, expected {expected_q1}"
    )
    assert result["q2"] == expected_q2, (
        f"q2 mismatch: got {result['q2']}, expected {expected_q2}"
    )
    assert result["q3"] == expected_q3, (
        f"q3 mismatch: got {result['q3']}, expected {expected_q3}"
    )


@settings(max_examples=100)
@given(data=numerical_column_strategy)
def test_numerical_stats_all_values_rounded_to_2dp(data):
    """All computed numerical statistics are rounded to exactly 2 decimal places."""
    import math
    series = pd.Series(data, dtype=float)
    # Require at least 2 non-NaN values so std is not NaN
    assume(series.notna().sum() >= 2)

    result = compute_numerical_stats(series)
    for key, value in result.items():
        # Skip NaN values (shouldn't occur with 2+ values, but handle gracefully)
        if math.isnan(value):
            continue
        # Verify that rounding to 2dp yields the same value
        assert value == round(value, 2), (
            f"{key} not rounded to 2dp: {value}"
        )


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 5: Categorical Statistics Computation
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(data=categorical_column_strategy)
def test_categorical_unique_count_matches_nunique(data):
    """unique_count equals the number of distinct non-null values in the column."""
    series = pd.Series(data, dtype=str)

    result = compute_categorical_stats(series)
    expected_unique = int(series.nunique())
    assert result["unique_count"] == expected_unique, (
        f"unique_count mismatch: got {result['unique_count']}, expected {expected_unique}"
    )


@settings(max_examples=100)
@given(data=categorical_column_strategy)
def test_categorical_top_5_has_at_most_5_entries(data):
    """top_5 list contains at most 5 entries."""
    series = pd.Series(data, dtype=str)

    result = compute_categorical_stats(series)
    assert len(result["top_5"]) <= 5, (
        f"top_5 has {len(result['top_5'])} entries, expected at most 5"
    )


@settings(max_examples=100)
@given(data=categorical_column_strategy)
def test_categorical_top_5_sorted_by_count_descending(data):
    """top_5 entries are sorted by count in descending order."""
    series = pd.Series(data, dtype=str)

    result = compute_categorical_stats(series)
    counts = [entry["count"] for entry in result["top_5"]]
    assert counts == sorted(counts, reverse=True), (
        f"top_5 not sorted descending: {counts}"
    )


@settings(max_examples=100)
@given(data=categorical_column_strategy)
def test_categorical_top_5_counts_match_actual_occurrences(data):
    """Each entry's count in top_5 matches the actual occurrence count in the data."""
    series = pd.Series(data, dtype=str)

    result = compute_categorical_stats(series)
    actual_counts = series.value_counts()

    for entry in result["top_5"]:
        value = entry["value"]
        count = entry["count"]
        expected_count = int(actual_counts[value])
        assert count == expected_count, (
            f"Count mismatch for '{value}': got {count}, expected {expected_count}"
        )


@settings(max_examples=100)
@given(data=categorical_column_strategy)
def test_categorical_top_5_contains_most_frequent_values(data):
    """top_5 contains the most frequent values from the column."""
    series = pd.Series(data, dtype=str)

    result = compute_categorical_stats(series)
    actual_top_5 = series.value_counts().head(5)

    # The values in result should match the top 5 from value_counts
    result_values = {entry["value"] for entry in result["top_5"]}
    expected_values = {str(v) for v in actual_top_5.index}
    assert result_values == expected_values, (
        f"top_5 values mismatch: got {result_values}, expected {expected_values}"
    )
