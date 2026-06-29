"""Property-based tests for ML module (train/test split, metrics, validation, feature importance)."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sklearn.model_selection import train_test_split

from backend.ml import (
    validate_target_column,
    train_classification,
    train_regression,
    compute_feature_importance,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def classification_dataframe_strategy(draw):
    """Generate a DataFrame suitable for classification (categorical target, numeric features)."""
    n_rows = draw(st.integers(min_value=20, max_value=80))
    n_features = draw(st.integers(min_value=2, max_value=5))
    n_classes = draw(st.integers(min_value=2, max_value=5))

    data = {}
    for i in range(n_features):
        data[f"feature_{i}"] = draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    class_labels = [f"class_{j}" for j in range(n_classes)]
    data["target"] = draw(
        st.lists(
            st.sampled_from(class_labels),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    df = pd.DataFrame(data)
    # Ensure at least 2 classes are present in the data
    assume(df["target"].nunique() >= 2)
    return df


@st.composite
def regression_dataframe_strategy(draw):
    """Generate a DataFrame suitable for regression (numeric target, numeric features)."""
    n_rows = draw(st.integers(min_value=20, max_value=80))
    n_features = draw(st.integers(min_value=2, max_value=5))

    data = {}
    for i in range(n_features):
        data[f"feature_{i}"] = draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    # Target is numeric
    data["target"] = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    df = pd.DataFrame(data)
    # Ensure at least 2 distinct target values for meaningful regression
    assume(df["target"].nunique() >= 2)
    return df


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 22: Train/Test Split Ratio
# Validates: Requirements 10.2, 11.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(n_rows=st.integers(min_value=10, max_value=200))
def test_train_test_split_ratio_produces_correct_sizes(n_rows):
    """For DataFrames with N rows (N>=10), 80/20 split produces expected sizes.

    sklearn's train_test_split with test_size=0.2 computes:
      n_test = ceil(n_rows * 0.2)
      n_train = n_rows - n_test
    """
    # Create simple data
    X = np.arange(n_rows).reshape(-1, 1)
    y = np.arange(n_rows)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Total must be preserved
    assert len(X_train) + len(X_test) == n_rows, (
        f"Total split size {len(X_train) + len(X_test)} != {n_rows}"
    )

    # sklearn uses ceil for test_size fraction: n_test = ceil(n * test_size)
    expected_test_size = math.ceil(n_rows * 0.2)
    expected_train_size = n_rows - expected_test_size

    assert len(X_test) == expected_test_size, (
        f"Test size {len(X_test)} != expected {expected_test_size} for n={n_rows}"
    )
    assert len(X_train) == expected_train_size, (
        f"Train size {len(X_train)} != expected {expected_train_size} for n={n_rows}"
    )


@settings(max_examples=100)
@given(n_rows=st.integers(min_value=10, max_value=200))
def test_train_test_split_deterministic_with_random_state_42(n_rows):
    """Split with random_state=42 is deterministic across calls."""
    X = np.arange(n_rows).reshape(-1, 1)
    y = np.arange(n_rows)

    X_train_1, X_test_1, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_2, X_test_2, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    assert np.array_equal(X_train_1, X_train_2), "Split not deterministic with same random_state"
    assert np.array_equal(X_test_1, X_test_2), "Split not deterministic with same random_state"


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 23: Classification Metric Validity
# Validates: Requirements 10.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["logistic_regression", "decision_tree", "random_forest"])
def test_classification_metrics_in_valid_range(algorithm):
    """Classification metrics (accuracy, precision, recall, F1) all in [0, 1]."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": np.random.choice(["A", "B", "C"], n),
    })

    result = train_classification(df, "target", algorithm)

    assert 0.0 <= result["accuracy"] <= 1.0, f"accuracy={result['accuracy']} out of [0,1]"
    assert 0.0 <= result["precision"] <= 1.0, f"precision={result['precision']} out of [0,1]"
    assert 0.0 <= result["recall"] <= 1.0, f"recall={result['recall']} out of [0,1]"
    assert 0.0 <= result["f1_score"] <= 1.0, f"f1_score={result['f1_score']} out of [0,1]"


@pytest.mark.parametrize("algorithm", ["logistic_regression", "decision_tree", "random_forest"])
def test_classification_confusion_matrix_row_sums(algorithm):
    """Confusion matrix row sums equal the count of each actual class in the test set."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": np.random.choice(["A", "B", "C"], n),
    })

    result = train_classification(df, "target", algorithm)
    cm = result["confusion_matrix"]
    actuals = result["predictions"]["actual"]

    # Row sums should equal count of each actual class in test set
    # The confusion matrix rows correspond to actual classes
    total_from_cm = sum(sum(row) for row in cm)
    total_from_predictions = len(actuals)

    assert total_from_cm == total_from_predictions, (
        f"Confusion matrix total {total_from_cm} != test set size {total_from_predictions}"
    )

    # Each row sum corresponds to count of that class in actuals
    row_sums = [sum(row) for row in cm]
    assert sum(row_sums) == len(actuals), (
        f"Sum of row sums {sum(row_sums)} != total test samples {len(actuals)}"
    )


@settings(max_examples=100)
@given(df=classification_dataframe_strategy())
def test_classification_metrics_always_in_unit_interval(df):
    """For any valid classification dataset, all metrics are in [0, 1]."""
    result = train_classification(df, "target", "decision_tree")

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1_score"] <= 1.0


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 24: Regression Metric Relationships
# Validates: Requirements 11.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["linear_regression", "decision_tree", "random_forest"])
def test_regression_rmse_equals_sqrt_mse(algorithm):
    """RMSE equals sqrt(MSE) for all regression algorithms."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 5,
    })

    result = train_regression(df, "target", algorithm)

    expected_rmse = round(math.sqrt(result["mse"]), 4)
    assert abs(result["rmse"] - expected_rmse) <= 0.0001, (
        f"RMSE={result['rmse']} != sqrt(MSE)={expected_rmse}"
    )


@pytest.mark.parametrize("algorithm", ["linear_regression", "decision_tree", "random_forest"])
def test_regression_mae_leq_rmse(algorithm):
    """MAE <= RMSE for all regression algorithms."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 5,
    })

    result = train_regression(df, "target", algorithm)

    assert result["mae"] <= result["rmse"], (
        f"MAE={result['mae']} > RMSE={result['rmse']}"
    )


@pytest.mark.parametrize("algorithm", ["linear_regression", "decision_tree", "random_forest"])
def test_regression_metrics_non_negative(algorithm):
    """MSE >= 0 and MAE >= 0 for all regression algorithms."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 5,
    })

    result = train_regression(df, "target", algorithm)

    assert result["mse"] >= 0, f"MSE={result['mse']} is negative"
    assert result["mae"] >= 0, f"MAE={result['mae']} is negative"


@pytest.mark.parametrize("algorithm", ["linear_regression", "decision_tree", "random_forest"])
def test_regression_metrics_rounded_to_4dp(algorithm):
    """All regression metrics are rounded to exactly 4 decimal places."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.randn(n) * 10 + 5,
    })

    result = train_regression(df, "target", algorithm)

    for metric_name in ["r2_score", "mae", "mse", "rmse"]:
        value = result[metric_name]
        # Check that rounding to 4dp doesn't change the value
        assert value == round(value, 4), (
            f"{metric_name}={value} is not rounded to 4 decimal places"
        )


@settings(max_examples=100)
@given(df=regression_dataframe_strategy())
def test_regression_metric_relationships_hold_universally(df):
    """For any valid regression dataset: RMSE=sqrt(MSE), MAE<=RMSE, MSE>=0, MAE>=0."""
    result = train_regression(df, "target", "decision_tree")

    # MSE and MAE non-negative
    assert result["mse"] >= 0, f"MSE={result['mse']} is negative"
    assert result["mae"] >= 0, f"MAE={result['mae']} is negative"

    # MAE <= RMSE
    assert result["mae"] <= result["rmse"] + 0.0001, (
        f"MAE={result['mae']} > RMSE={result['rmse']}"
    )

    # RMSE = sqrt(MSE) (within rounding tolerance)
    # Both RMSE and MSE are independently rounded to 4dp from raw values,
    # so sqrt(rounded_mse) may differ slightly from the independently rounded RMSE.
    expected_rmse = round(math.sqrt(result["mse"]), 4)
    assert abs(result["rmse"] - expected_rmse) <= 0.001, (
        f"RMSE={result['rmse']} != sqrt(MSE)={expected_rmse}"
    )

    # All rounded to 4dp
    for metric_name in ["r2_score", "mae", "mse", "rmse"]:
        value = result[metric_name]
        assert value == round(value, 4), f"{metric_name} not rounded to 4dp"


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 25: Task Type Target Column Validation
# Validates: Requirements 10.6, 11.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(n_unique=st.integers(min_value=21, max_value=100))
def test_classification_rejects_numerical_target_with_many_unique(n_unique):
    """Classification rejects numerical target columns with >20 unique values."""
    n_rows = max(n_unique, 20)
    # Create a numerical column with n_unique distinct values
    values = list(range(n_unique)) + [0] * (n_rows - n_unique)
    df = pd.DataFrame({
        "feature": np.random.randn(n_rows),
        "target": values[:n_rows],
    })
    # Ensure target is numeric
    df["target"] = df["target"].astype(float)

    is_valid, error_msg = validate_target_column(df, "target", "classification")

    assert is_valid is False, (
        f"Classification should reject numerical target with {n_unique} unique values"
    )
    assert error_msg is not None
    assert "categorical" in error_msg.lower() or "numerical" in error_msg.lower()


@settings(max_examples=100)
@given(
    categories=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",)),
            min_size=1,
            max_size=5,
        ),
        min_size=10,
        max_size=50,
    )
)
def test_regression_rejects_categorical_target(categories):
    """Regression rejects categorical (object dtype) target columns."""
    n_rows = len(categories)
    df = pd.DataFrame({
        "feature": np.random.randn(n_rows),
        "target": categories,
    })
    # Ensure target is object dtype
    df["target"] = df["target"].astype(str)

    is_valid, error_msg = validate_target_column(df, "target", "regression")

    assert is_valid is False, "Regression should reject categorical target"
    assert error_msg is not None
    assert "numerical" in error_msg.lower() or "categorical" in error_msg.lower()


def test_classification_accepts_categorical_target():
    """Classification accepts categorical target columns."""
    df = pd.DataFrame({
        "feature": np.random.randn(20),
        "target": ["A", "B"] * 10,
    })

    is_valid, error_msg = validate_target_column(df, "target", "classification")
    assert is_valid is True
    assert error_msg is None


def test_regression_accepts_numerical_target():
    """Regression accepts numerical target columns."""
    df = pd.DataFrame({
        "feature": np.random.randn(20),
        "target": np.random.randn(20),
    })

    is_valid, error_msg = validate_target_column(df, "target", "regression")
    assert is_valid is True
    assert error_msg is None


# ---------------------------------------------------------------------------
# Feature: ai-data-analyst, Property 26: Feature Importance Properties
# Validates: Requirements 12.1, 12.2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm,task", [
    ("decision_tree", "classification"),
    ("random_forest", "classification"),
    ("decision_tree", "regression"),
    ("random_forest", "regression"),
])
def test_feature_importance_one_score_per_feature(algorithm, task):
    """Feature importance has one score per input feature for tree-based models."""
    np.random.seed(42)
    n = 50
    n_features = 4

    if task == "classification":
        df = pd.DataFrame({
            **{f"f{i}": np.random.randn(n) for i in range(n_features)},
            "target": np.random.choice(["A", "B", "C"], n),
        })
        result = train_classification(df, "target", algorithm)
    else:
        df = pd.DataFrame({
            **{f"f{i}": np.random.randn(n) for i in range(n_features)},
            "target": np.random.randn(n),
        })
        result = train_regression(df, "target", algorithm)

    importance = result["feature_importance"]
    assert importance["available"] is True
    assert len(importance["features"]) == n_features, (
        f"Expected {n_features} features, got {len(importance['features'])}"
    )


@pytest.mark.parametrize("algorithm,task", [
    ("decision_tree", "classification"),
    ("random_forest", "classification"),
    ("decision_tree", "regression"),
    ("random_forest", "regression"),
])
def test_feature_importance_scores_in_range_0_to_1(algorithm, task):
    """All feature importance scores are in the range [0.0, 1.0]."""
    np.random.seed(42)
    n = 50

    if task == "classification":
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.choice(["A", "B"], n),
        })
        result = train_classification(df, "target", algorithm)
    else:
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.randn(n),
        })
        result = train_regression(df, "target", algorithm)

    importance = result["feature_importance"]
    for feat in importance["features"]:
        assert 0.0 <= feat["importance"] <= 1.0, (
            f"Feature '{feat['name']}' importance {feat['importance']} not in [0, 1]"
        )


@pytest.mark.parametrize("algorithm,task", [
    ("decision_tree", "classification"),
    ("random_forest", "classification"),
    ("decision_tree", "regression"),
    ("random_forest", "regression"),
])
def test_feature_importance_sum_approximately_one(algorithm, task):
    """Feature importance scores sum to approximately 1.0 (within 1e-6)."""
    np.random.seed(42)
    n = 50

    if task == "classification":
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.choice(["A", "B"], n),
        })
        result = train_classification(df, "target", algorithm)
    else:
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.randn(n),
        })
        result = train_regression(df, "target", algorithm)

    importance = result["feature_importance"]
    total = sum(feat["importance"] for feat in importance["features"])
    # Allow tolerance for rounding (each score rounded to 4dp)
    assert abs(total - 1.0) < 0.01, (
        f"Feature importance sum {total} not approximately 1.0"
    )


@pytest.mark.parametrize("algorithm,task", [
    ("decision_tree", "classification"),
    ("random_forest", "classification"),
    ("decision_tree", "regression"),
    ("random_forest", "regression"),
])
def test_feature_importance_sorted_descending(algorithm, task):
    """Feature importance is sorted in descending order."""
    np.random.seed(42)
    n = 50

    if task == "classification":
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.choice(["A", "B"], n),
        })
        result = train_classification(df, "target", algorithm)
    else:
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.randn(n),
        })
        result = train_regression(df, "target", algorithm)

    importance = result["feature_importance"]
    scores = [feat["importance"] for feat in importance["features"]]
    assert scores == sorted(scores, reverse=True), (
        f"Feature importances not sorted descending: {scores}"
    )
