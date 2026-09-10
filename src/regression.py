"""
regression.py

Trains, evaluates and compares regression models that predict a
student's Final_Score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
    build_preprocessing_pipeline,
)


def get_regression_models() -> dict:
    """Return a dictionary of candidate regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def train_and_evaluate_regressors(X_train, X_test, y_train, y_test) -> tuple[dict, pd.DataFrame]:
    """Train every candidate regression model and evaluate on the test set.

    Returns:
        trained_pipelines: dict of {model_name: fitted sklearn Pipeline}
        results_df: comparison DataFrame with MAE, MSE, RMSE, R2
    """
    models = get_regression_models()
    trained_pipelines = {}
    results = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2 Score": round(r2, 4),
        })

        trained_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return trained_pipelines, results_df


def get_best_regression_model(trained_pipelines: dict, results_df: pd.DataFrame):
    """Return (best_model_name, best_pipeline) based on highest R2 score."""
    best_model_name = results_df.iloc[0]["Model"]
    return best_model_name, trained_pipelines[best_model_name]


def get_actual_vs_predicted(pipeline, X_test, y_test) -> pd.DataFrame:
    """Return a dataframe comparing actual vs predicted Final_Score values."""
    predictions = pipeline.predict(X_test)
    return pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": np.round(predictions, 2),
    }).reset_index(drop=True)
