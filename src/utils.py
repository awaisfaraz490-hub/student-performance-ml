"""
utils.py

Small shared helper functions used across the app: performance
categorization, feature importance extraction, and model persistence
helpers.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "best_regression_model.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "best_classification_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.pkl")


def categorize_performance(score: float) -> str:
    """Map a predicted Final_Score into a human-readable performance category."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Average"
    elif score >= 50:
        return "Needs Improvement"
    else:
        return "At Risk"


def explain_prediction(student_input: dict, predicted_score: float) -> str:
    """Generate a short, rule-based, human-readable explanation for a
    prediction. This is a descriptive summary of contributing factors,
    not a diagnosis of any kind.
    """
    factors = []

    if student_input.get("Study_Hours", 0) < 10:
        factors.append("low weekly study hours")
    if student_input.get("Attendance", 0) < 70:
        factors.append("low attendance")
    if student_input.get("Sleep_Hours", 0) < 5 or student_input.get("Sleep_Hours", 0) > 9:
        factors.append("irregular sleep patterns")
    if student_input.get("Previous_Score", 0) < 50:
        factors.append("a weaker previous academic score")
    if student_input.get("Assignments_Score", 0) < 50:
        factors.append("low assignment scores")

    positive_factors = []
    if student_input.get("Study_Hours", 0) >= 20:
        positive_factors.append("strong study habits")
    if student_input.get("Attendance", 0) >= 90:
        positive_factors.append("excellent attendance")
    if student_input.get("Previous_Score", 0) >= 80:
        positive_factors.append("a strong academic history")

    if predicted_score < 50 and factors:
        return "Contributing factors: " + ", ".join(factors) + "."
    elif predicted_score >= 70 and positive_factors:
        return "Key strengths: " + ", ".join(positive_factors) + "."
    elif factors:
        return "Areas to watch: " + ", ".join(factors) + "."
    else:
        return "Performance appears balanced across the measured factors."


def get_feature_importance_df(pipeline, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract feature importance from a fitted pipeline's final model
    step, if the model type supports it (tree-based models).
    """
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        return None

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def save_models(regression_pipeline, classification_pipeline, metadata: dict) -> None:
    """Persist the best regression and classification pipelines plus metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(regression_pipeline, REGRESSION_MODEL_PATH)
    joblib.dump(classification_pipeline, CLASSIFICATION_MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)


def load_models():
    """Load persisted models and metadata. Returns (reg, clf, metadata) or
    (None, None, None) if any file is missing.
    """
    if not (os.path.exists(REGRESSION_MODEL_PATH) and
            os.path.exists(CLASSIFICATION_MODEL_PATH) and
            os.path.exists(METADATA_PATH)):
        return None, None, None

    regression_pipeline = joblib.load(REGRESSION_MODEL_PATH)
    classification_pipeline = joblib.load(CLASSIFICATION_MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return regression_pipeline, classification_pipeline, metadata


def models_exist() -> bool:
    return (os.path.exists(REGRESSION_MODEL_PATH) and
            os.path.exists(CLASSIFICATION_MODEL_PATH) and
            os.path.exists(METADATA_PATH))
