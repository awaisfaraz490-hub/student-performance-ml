"""
preprocessing.py

Shared data cleaning and preprocessing utilities used across the app and
the training script. A single sklearn ColumnTransformer / Pipeline is
reused by every model to avoid duplicated preprocessing logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns expected in a valid dataset
REQUIRED_COLUMNS = [
    "Student_ID",
    "Age",
    "Gender",
    "Study_Hours",
    "Attendance",
    "Assignments_Score",
    "Midterm_Score",
    "Previous_Score",
    "Sleep_Hours",
    "Internet_Access",
    "Extracurricular_Activities",
    "Final_Score",
]

NUMERICAL_FEATURES = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Assignments_Score",
    "Midterm_Score",
    "Previous_Score",
    "Sleep_Hours",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Internet_Access",
    "Extracurricular_Activities",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION = "Final_Score"
TARGET_CLASSIFICATION = "At_Risk"
AT_RISK_THRESHOLD = 50.0


def validate_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that a dataframe contains all required columns.

    Returns:
        (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary summary of the dataset for the overview page."""
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw dataframe: drop duplicates, drop rows with a missing
    target, and standardize categorical text values.

    Missing values in feature columns are intentionally left in place --
    they are handled later by the SimpleImputer inside the preprocessing
    pipeline, which is the more robust, sklearn-native approach.
    """
    cleaned = df.copy()

    # Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Drop rows where the target itself is missing (can't train/evaluate on these)
    if TARGET_REGRESSION in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_REGRESSION])

    # Standardize categorical text (strip whitespace, consistent casing)
    for col in CATEGORICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def add_at_risk_target(df: pd.DataFrame, threshold: float = AT_RISK_THRESHOLD) -> pd.DataFrame:
    """Add a binary At_Risk classification target derived from Final_Score."""
    result = df.copy()
    result[TARGET_CLASSIFICATION] = np.where(
        result[TARGET_REGRESSION] < threshold, "At Risk", "Not At Risk"
    )
    return result


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a reusable ColumnTransformer that:
      - imputes and scales numerical features
      - imputes and one-hot encodes categorical features

    This single transformer is shared by every regression and
    classification model to avoid duplicated preprocessing code.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recover human-readable feature names after a ColumnTransformer
    has transformed numerical + one-hot encoded categorical columns.
    """
    output_features = []

    # Numerical features keep their original names
    output_features.extend(NUMERICAL_FEATURES)

    # Categorical features are expanded by OneHotEncoder
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    output_features.extend(cat_feature_names.tolist())

    return output_features
