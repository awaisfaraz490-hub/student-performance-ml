"""
train_models.py

Standalone training script for the Student Performance Intelligence
project. Loads the sample dataset, cleans it, trains all regression and
classification candidate models, selects the best of each, and saves
them to the models/ directory using joblib.

Usage:
    python train_models.py
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running this script directly from the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.classification import (
    get_best_classification_model,
    train_and_evaluate_classifiers,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    add_at_risk_target,
    clean_dataset,
)
from src.regression import get_best_regression_model, train_and_evaluate_regressors
from src.utils import save_models

DATA_PATH = os.path.join("data", "student_performance.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    print("=" * 60)
    print("Student Performance Intelligence - Model Training")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at '{DATA_PATH}'.")
        print("Run 'python generate_dataset.py' first, or place a valid CSV there.")
        sys.exit(1)

    print(f"\nLoading dataset from {DATA_PATH} ...")
    raw_df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {raw_df.shape}")

    print("\nCleaning dataset (removing duplicates, standardizing categories) ...")
    df = clean_dataset(raw_df)
    df = add_at_risk_target(df)
    print(f"Cleaned shape: {df.shape}")

    X = df[FEATURE_COLUMNS]
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]

    # --- Regression -----------------------------------------------------
    print("\n" + "-" * 60)
    print("Training regression models (predicting Final_Score) ...")
    print("-" * 60)
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    reg_pipelines, reg_results = train_and_evaluate_regressors(
        X_train, X_test, y_reg_train, y_reg_test
    )
    print("\nRegression Model Comparison:")
    print(reg_results.to_string(index=False))

    best_reg_name, best_reg_pipeline = get_best_regression_model(reg_pipelines, reg_results)
    print(f"\nBest Regression Model: {best_reg_name}")

    # --- Classification ---------------------------------------------------
    print("\n" + "-" * 60)
    print("Training classification models (predicting At_Risk) ...")
    print("-" * 60)
    X_train_c, X_test_c, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
    )
    clf_pipelines, clf_results, _ = train_and_evaluate_classifiers(
        X_train_c, X_test_c, y_clf_train, y_clf_test
    )
    print("\nClassification Model Comparison:")
    print(clf_results.to_string(index=False))

    best_clf_name, best_clf_pipeline = get_best_classification_model(clf_pipelines, clf_results)
    print(f"\nBest Classification Model: {best_clf_name}")

    # --- Save models ------------------------------------------------------
    print("\n" + "-" * 60)
    print("Saving best models to models/ ...")
    print("-" * 60)

    metadata = {
        "best_regression_model_name": best_reg_name,
        "best_classification_model_name": best_clf_name,
        "regression_results": reg_results.to_dict(orient="records"),
        "classification_results": clf_results.to_dict(orient="records"),
        "feature_columns": FEATURE_COLUMNS,
    }

    save_models(best_reg_pipeline, best_clf_pipeline, metadata)

    print("\nModels saved successfully:")
    print("  - models/best_regression_model.pkl")
    print("  - models/best_classification_model.pkl")
    print("  - models/metadata.pkl")
    print("\nTraining complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
