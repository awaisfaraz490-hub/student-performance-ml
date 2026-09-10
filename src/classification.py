"""
classification.py

Trains, evaluates and compares classification models that detect whether
a student is academically "At Risk" based on Final_Score < 50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessing_pipeline


def get_classification_models() -> dict:
    """Return a dictionary of candidate classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame, dict]:
    """Train every candidate classification model and evaluate on the test set.

    Returns:
        trained_pipelines: dict of {model_name: fitted sklearn Pipeline}
        results_df: comparison DataFrame with Accuracy, Precision, Recall, F1
        confusion_matrices: dict of {model_name: confusion_matrix ndarray}
    """
    models = get_classification_models()
    trained_pipelines = {}
    results = []
    confusion_matrices = {}

    positive_label = "At Risk"

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label=positive_label, zero_division=0)
        recall = recall_score(y_test, predictions, pos_label=positive_label, zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label=positive_label, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
        })

        trained_pipelines[name] = pipeline
        confusion_matrices[name] = confusion_matrix(
            y_test, predictions, labels=["At Risk", "Not At Risk"]
        )

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    return trained_pipelines, results_df, confusion_matrices


def get_best_classification_model(trained_pipelines: dict, results_df: pd.DataFrame):
    """Return (best_model_name, best_pipeline) based on highest F1 score."""
    best_model_name = results_df.iloc[0]["Model"]
    return best_model_name, trained_pipelines[best_model_name]
