"""Machine learning module for model training and evaluation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder


CLASSIFICATION_ALGORITHMS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}

REGRESSION_ALGORITHMS = {
    "linear_regression": LinearRegression,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
}


def validate_target_column(df, target_col, task_type):
    """Validate that the target column is appropriate for the task type.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column.
        task_type: Either "classification" or "regression".

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if target_col not in df.columns:
        return False, f"Column '{target_col}' does not exist in the dataset."

    if len(df) < 10:
        return False, "Dataset must have at least 10 rows for model training."

    col_dtype = df[target_col].dtype

    if task_type == "classification":
        # Target must be categorical (object dtype) or have few unique values
        if pd.api.types.is_numeric_dtype(col_dtype):
            # Allow numeric columns with few unique values (likely class labels)
            n_unique = df[target_col].nunique()
            if n_unique > 20:
                return False, (
                    "Classification requires a categorical target column. "
                    f"Column '{target_col}' is numerical with {n_unique} unique values."
                )
    elif task_type == "regression":
        # Target must be numeric
        if not pd.api.types.is_numeric_dtype(col_dtype):
            return False, (
                "Regression requires a numerical target column. "
                f"Column '{target_col}' is categorical."
            )
    else:
        return False, f"Invalid task type '{task_type}'. Must be 'classification' or 'regression'."

    return True, None


def _prepare_features(df, target_col):
    """Prepare feature matrix and target vector from a DataFrame.

    Drops rows with NaN in features or target, encodes categorical features,
    and returns the prepared X, y, and feature names.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y, feature_names) where X is the feature matrix,
        y is the target vector, and feature_names is a list of feature column names.

    Raises:
        ValueError: If insufficient rows after NaN removal or no valid features.
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    working_df = df[feature_cols + [target_col]].copy()

    # Drop rows with NaN in features or target
    working_df = working_df.dropna()

    if len(working_df) < 10:
        raise ValueError(
            f"Insufficient rows for training after removing missing values. "
            f"Need at least 10 rows, got {len(working_df)}."
        )

    y = working_df[target_col]
    X = working_df[feature_cols]

    # Handle non-numeric features: encode categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    encoded_dfs = [X[numeric_cols]] if numeric_cols else []
    encoded_feature_names = list(numeric_cols)

    for col in categorical_cols:
        # Label encode each categorical feature
        le = LabelEncoder()
        encoded_col = le.fit_transform(X[col].astype(str))
        encoded_dfs.append(pd.DataFrame({col: encoded_col}, index=X.index))
        encoded_feature_names.append(col)

    if not encoded_feature_names:
        raise ValueError("No valid features available for training.")

    X_encoded = pd.concat(encoded_dfs, axis=1)

    return X_encoded, y, encoded_feature_names


def train_classification(df, target_col, algorithm):
    """Train a classification model on the dataset.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column (must be categorical).
        algorithm: One of "logistic_regression", "decision_tree", "random_forest".

    Returns:
        Dictionary containing metrics, confusion matrix, predictions, and feature names.

    Raises:
        ValueError: If algorithm is invalid, data is incompatible, or insufficient rows.
    """
    if algorithm not in CLASSIFICATION_ALGORITHMS:
        raise ValueError(
            f"Invalid classification algorithm '{algorithm}'. "
            f"Must be one of: {list(CLASSIFICATION_ALGORITHMS.keys())}"
        )

    # Validate target column
    is_valid, error_msg = validate_target_column(df, target_col, "classification")
    if not is_valid:
        raise ValueError(error_msg)

    # Prepare features
    X, y, feature_names = _prepare_features(df, target_col)

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train model
    model_class = CLASSIFICATION_ALGORITHMS[algorithm]
    if algorithm == "logistic_regression":
        model = model_class(max_iter=1000, random_state=42)
    elif algorithm == "random_forest":
        model = model_class(n_estimators=100, random_state=42)
    else:
        model = model_class(random_state=42)

    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    prec = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    rec = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Decode predictions back to original labels
    actual_labels = le.inverse_transform(y_test).tolist()
    predicted_labels = le.inverse_transform(y_pred).tolist()

    # Compute feature importance
    importance = compute_feature_importance(model, feature_names)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predictions": {
            "actual": actual_labels,
            "predicted": predicted_labels,
        },
        "feature_names": feature_names,
        "feature_importance": importance,
        "model": model,
    }


def train_regression(df, target_col, algorithm):
    """Train a regression model on the dataset.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column (must be numerical).
        algorithm: One of "linear_regression", "decision_tree", "random_forest".

    Returns:
        Dictionary containing metrics, predictions, and feature names.

    Raises:
        ValueError: If algorithm is invalid, data is incompatible, or insufficient rows.
    """
    if algorithm not in REGRESSION_ALGORITHMS:
        raise ValueError(
            f"Invalid regression algorithm '{algorithm}'. "
            f"Must be one of: {list(REGRESSION_ALGORITHMS.keys())}"
        )

    # Validate target column
    is_valid, error_msg = validate_target_column(df, target_col, "regression")
    if not is_valid:
        raise ValueError(error_msg)

    # Prepare features
    X, y, feature_names = _prepare_features(df, target_col)

    # Ensure target is numeric
    y = y.astype(float)

    # Split data 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model_class = REGRESSION_ALGORITHMS[algorithm]
    if algorithm == "random_forest":
        model = model_class(n_estimators=100, random_state=42)
    elif algorithm == "linear_regression":
        model = model_class()
    else:
        model = model_class(random_state=42)

    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics (all rounded to 4 decimal places)
    r2 = round(float(r2_score(y_test, y_pred)), 4)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
    mse = round(float(mean_squared_error(y_test, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

    # Compute feature importance
    importance = compute_feature_importance(model, feature_names)

    return {
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "predictions": {
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist(),
        },
        "feature_names": feature_names,
        "feature_importance": importance,
        "model": model,
    }


def compute_feature_importance(model, feature_names):
    """Compute feature importance for a trained model.

    Args:
        model: A trained scikit-learn model.
        feature_names: List of feature names used during training.

    Returns:
        Dictionary with 'features' (list of {name, importance}),
        'available' (bool), and 'message' (str or None).
    """
    if not hasattr(model, "feature_importances_"):
        return {
            "features": [],
            "available": False,
            "message": "Feature importance is only available for tree-based models.",
        }

    importances = model.feature_importances_

    # Build feature importance list
    feature_list = [
        {"name": name, "importance": round(float(imp), 4)}
        for name, imp in zip(feature_names, importances)
    ]

    # Sort by importance descending
    feature_list.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "features": feature_list,
        "available": True,
        "message": None,
    }
