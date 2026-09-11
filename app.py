"""
app.py

Student Performance Intelligence & Prediction System
A Streamlit dashboard for analyzing student academic data and predicting
performance / at-risk status using Machine Learning.

Run with:
    streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.auth import login_signup_gate, render_logout_sidebar
from src.classification import (
    get_best_classification_model,
    train_and_evaluate_classifiers,
)
from src.preprocessing import (
    AT_RISK_THRESHOLD,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    add_at_risk_target,
    clean_dataset,
    get_data_summary,
    get_transformed_feature_names,
    validate_columns,
)
from src.regression import (
    get_actual_vs_predicted,
    get_best_regression_model,
    train_and_evaluate_regressors,
)
from src.utils import (
    categorize_performance,
    explain_prediction,
    get_feature_importance_df,
    load_models,
    models_exist,
    save_models,
)
from src.visualization import (
    actual_vs_predicted_chart,
    average_score_by_group,
    category_distribution,
    confusion_matrix_chart,
    correlation_heatmap,
    feature_importance_chart,
    final_score_distribution,
    scatter_with_trend,
)

DATA_PATH = os.path.join("data", "student_performance.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #1a2b4c;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #5c6b7a;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a2b4c;
        border-bottom: 2px solid #e6e9ee;
        padding-bottom: 0.4rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #8a97a5;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e6e9ee;
    }
    div[data-testid="stMetric"] {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ee;
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Authentication gate — must pass before any other page renders
# --------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not login_signup_gate():
    st.stop()


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "cleaned_dataset" not in st.session_state:
    st.session_state.cleaned_dataset = None
if "trained" not in st.session_state:
    st.session_state.trained = False


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame | None:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def get_active_dataset() -> pd.DataFrame | None:
    """Return the dataset currently in use: uploaded dataset if present,
    otherwise the bundled sample dataset."""
    if st.session_state.dataset is not None:
        return st.session_state.dataset
    return load_default_dataset()


def prepare_cleaned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_dataset(df)
    cleaned = add_at_risk_target(cleaned)
    return cleaned


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("### Student Performance Intelligence")
render_logout_sidebar()

PAGES = [
    "Dashboard / Home",
    "Dataset Upload",
    "Data Overview",
    "Data Cleaning & Preprocessing",
    "Exploratory Data Analysis",
    "Regression Prediction",
    "Classification / At-Risk Detection",
    "Model Comparison",
    "Feature Importance",
    "Individual Student Prediction",
    "Download Results",
    "About Project",
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
if models_exist():
    st.sidebar.success("Trained models found on disk.")
else:
    st.sidebar.warning("No trained models found yet. Visit 'Model Comparison' to train.")


# ==========================================================================
# PAGE: Dashboard / Home
# ==========================================================================
if page == "Dashboard / Home":
    st.markdown('<div class="main-title">Student Performance Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Machine Learning based academic performance analysis and prediction</div>',
        unsafe_allow_html=True,
    )

    df = get_active_dataset()

    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        try:
            cleaned = prepare_cleaned_dataset(df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Students", f"{cleaned.shape[0]:,}")
            col2.metric("Average Final Score", f"{cleaned['Final_Score'].mean():.1f}")
            at_risk_count = (cleaned[TARGET_CLASSIFICATION] == "At Risk").sum()
            col3.metric("At-Risk Students", f"{at_risk_count:,}")
            at_risk_pct = at_risk_count / cleaned.shape[0] * 100
            col4.metric("At-Risk Rate", f"{at_risk_pct:.1f}%")

            st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
            st.write(
                "This dashboard analyzes student academic records and uses trained Machine "
                "Learning models to predict final performance and flag students who may need "
                "additional academic support. Use the sidebar to explore the dataset, review "
                "model performance, or generate a prediction for an individual student."
            )

            left, right = st.columns(2)
            with left:
                st.plotly_chart(final_score_distribution(cleaned), use_container_width=True)
            with right:
                st.plotly_chart(
                    average_score_by_group(cleaned, "Extracurricular_Activities",
                                            "Average Final Score by Extracurricular Activity"),
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Could not summarize dataset: {exc}")

    st.markdown(
        '<div class="footer">Built with Python, Scikit-learn and Streamlit</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE: Dataset Upload
# ==========================================================================
elif page == "Dataset Upload":
    st.markdown('<div class="section-header">Dataset Upload</div>', unsafe_allow_html=True)
    st.write(
        "Upload your own CSV file, or continue using the bundled sample dataset "
        "(`data/student_performance.csv`)."
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            is_valid, missing_cols = validate_columns(new_df)

            if not is_valid:
                st.error(
                    "The uploaded file is missing required columns: "
                    + ", ".join(missing_cols)
                )
                st.info(f"Required columns: {', '.join(list(new_df.columns))}")
            else:
                st.session_state.dataset = new_df
                st.session_state.trained = False
                st.success(f"Dataset uploaded successfully. Shape: {new_df.shape[0]} rows x {new_df.shape[1]} columns.")
                st.dataframe(new_df.head(10), use_container_width=True)
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or not a valid CSV.")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset to sample dataset"):
            st.session_state.dataset = None
            st.session_state.trained = False
            st.success("Reverted to the bundled sample dataset.")
    with col_b:
        active_df = get_active_dataset()
        if active_df is not None:
            st.caption(f"Active dataset shape: {active_df.shape[0]} rows x {active_df.shape[1]} columns")


# ==========================================================================
# PAGE: Data Overview
# ==========================================================================
elif page == "Data Overview":
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    df = get_active_dataset()

    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            summary = get_data_summary(df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", summary["n_rows"])
            col2.metric("Columns", summary["n_columns"])
            col3.metric("Missing Values", summary["missing_values"])
            col4.metric("Duplicate Rows", summary["duplicate_rows"])

            st.markdown("#### Preview")
            st.dataframe(df.head(15), use_container_width=True)

            st.markdown("#### Data Types")
            dtype_df = pd.DataFrame(
                list(summary["dtypes"].items()), columns=["Column", "Data Type"]
            )
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Numerical Columns")
                st.write(summary["numerical_columns"])
            with col_right:
                st.markdown("#### Categorical Columns")
                st.write(summary["categorical_columns"])

            if summary["missing_by_column"]:
                st.markdown("#### Missing Values by Column")
                missing_df = pd.DataFrame(
                    list(summary["missing_by_column"].items()),
                    columns=["Column", "Missing Count"],
                )
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

            st.markdown("#### Statistical Summary")
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# ==========================================================================
# PAGE: Data Cleaning & Preprocessing
# ==========================================================================
elif page == "Data Cleaning & Preprocessing":
    st.markdown('<div class="section-header">Data Cleaning &amp; Preprocessing</div>', unsafe_allow_html=True)
    df = get_active_dataset()

    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            before_rows = df.shape[0]
            before_duplicates = df.duplicated().sum()
            before_missing = df.isna().sum().sum()

            cleaned = prepare_cleaned_dataset(df)
            st.session_state.cleaned_dataset = cleaned

            after_rows = cleaned.shape[0]
            after_missing_features = cleaned[FEATURE_COLUMNS].isna().sum().sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows Before", before_rows)
            col2.metric("Duplicate Rows Removed", int(before_duplicates))
            col3.metric("Rows After Cleaning", after_rows)

            st.markdown("#### Cleaning Steps Applied")
            st.markdown(
                """
                1. **Duplicate rows removed** based on exact row matches.
                2. **Rows with a missing target** (`Final_Score`) removed, since they cannot be used for training or evaluation.
                3. **Categorical text standardized** (trimmed whitespace, consistent capitalization).
                4. **`At_Risk` target created**: students with `Final_Score` below 50 are labeled *At Risk*.
                5. **Missing feature values** (numerical and categorical) are handled downstream by an sklearn imputation pipeline (median for numerical, most frequent for categorical) — this keeps preprocessing consistent and reproducible across every model.
                """
            )

            st.info(
                f"Remaining missing values in feature columns (handled automatically by the "
                f"ML pipeline's imputers at training time): {int(after_missing_features)}"
            )

            st.markdown("#### Cleaned Dataset Preview")
            st.dataframe(cleaned.head(15), use_container_width=True)

            st.markdown("#### At-Risk Target Distribution")
            risk_counts = cleaned[TARGET_CLASSIFICATION].value_counts()
            st.bar_chart(risk_counts)


# ==========================================================================
# PAGE: Exploratory Data Analysis
# ==========================================================================
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = get_active_dataset()

    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            cleaned = prepare_cleaned_dataset(df)

            st.plotly_chart(final_score_distribution(cleaned), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    scatter_with_trend(cleaned, "Study_Hours", "Final_Score", "Study Hours vs Final Score"),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    scatter_with_trend(cleaned, "Attendance", "Final_Score", "Attendance vs Final Score"),
                    use_container_width=True,
                )

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(
                    scatter_with_trend(cleaned, "Previous_Score", "Final_Score", "Previous Score vs Final Score"),
                    use_container_width=True,
                )
            with col4:
                st.plotly_chart(
                    category_distribution(cleaned, "Gender", "Gender Distribution"),
                    use_container_width=True,
                )

            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(
                    category_distribution(cleaned, "Internet_Access", "Internet Access Distribution"),
                    use_container_width=True,
                )
            with col6:
                st.plotly_chart(
                    average_score_by_group(cleaned, "Gender", "Average Final Score by Gender"),
                    use_container_width=True,
                )

            st.plotly_chart(
                average_score_by_group(
                    cleaned, "Extracurricular_Activities", "Average Final Score by Extracurricular Activity"
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                correlation_heatmap(cleaned, NUMERICAL_FEATURES + [TARGET_REGRESSION]),
                use_container_width=True,
            )


# ==========================================================================
# PAGE: Regression Prediction
# ==========================================================================
elif page == "Regression Prediction":
    st.markdown('<div class="section-header">Regression Prediction</div>', unsafe_allow_html=True)
    st.write("Train and evaluate multiple regression models to predict a student's `Final_Score`.")

    df = get_active_dataset()
    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            if st.button("Train Regression Models", type="primary"):
                with st.spinner("Training regression models..."):
                    cleaned = prepare_cleaned_dataset(df)
                    X = cleaned[FEATURE_COLUMNS]
                    y = cleaned[TARGET_REGRESSION]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
                    )
                    pipelines, results = train_and_evaluate_regressors(X_train, X_test, y_train, y_test)
                    best_name, best_pipeline = get_best_regression_model(pipelines, results)

                    st.session_state.reg_pipelines = pipelines
                    st.session_state.reg_results = results
                    st.session_state.reg_X_test = X_test
                    st.session_state.reg_y_test = y_test
                    st.session_state.best_reg_name = best_name
                    st.session_state.best_reg_pipeline = best_pipeline

            if "reg_results" in st.session_state:
                st.markdown("#### Model Comparison")
                st.dataframe(st.session_state.reg_results, use_container_width=True, hide_index=True)

                st.success(f"Best Regression Model: {st.session_state.best_reg_name}")

                comparison_df = get_actual_vs_predicted(
                    st.session_state.best_reg_pipeline,
                    st.session_state.reg_X_test,
                    st.session_state.reg_y_test,
                )
                st.plotly_chart(actual_vs_predicted_chart(comparison_df), use_container_width=True)
                st.dataframe(comparison_df.head(20), use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Train Regression Models' to begin.")


# ==========================================================================
# PAGE: Classification / At-Risk Detection
# ==========================================================================
elif page == "Classification / At-Risk Detection":
    st.markdown('<div class="section-header">Classification / At-Risk Detection</div>', unsafe_allow_html=True)
    st.write(
        f"Train and evaluate multiple classification models to detect students who are "
        f"academically at risk (`Final_Score` below {AT_RISK_THRESHOLD:.0f})."
    )

    df = get_active_dataset()
    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            if st.button("Train Classification Models", type="primary"):
                with st.spinner("Training classification models..."):
                    cleaned = prepare_cleaned_dataset(df)
                    X = cleaned[FEATURE_COLUMNS]
                    y = cleaned[TARGET_CLASSIFICATION]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
                    )
                    pipelines, results, matrices = train_and_evaluate_classifiers(
                        X_train, X_test, y_train, y_test
                    )
                    best_name, best_pipeline = get_best_classification_model(pipelines, results)

                    st.session_state.clf_pipelines = pipelines
                    st.session_state.clf_results = results
                    st.session_state.clf_matrices = matrices
                    st.session_state.best_clf_name = best_name
                    st.session_state.best_clf_pipeline = best_pipeline

            if "clf_results" in st.session_state:
                st.markdown("#### Model Comparison")
                st.dataframe(st.session_state.clf_results, use_container_width=True, hide_index=True)

                st.success(f"Best Classification Model: {st.session_state.best_clf_name}")

                cm = st.session_state.clf_matrices[st.session_state.best_clf_name]
                st.plotly_chart(
                    confusion_matrix_chart(cm, ["At Risk", "Not At Risk"], "Confusion Matrix (Best Model)"),
                    use_container_width=True,
                )
            else:
                st.info("Click 'Train Classification Models' to begin.")


# ==========================================================================
# PAGE: Model Comparison
# ==========================================================================
elif page == "Model Comparison":
    st.markdown('<div class="section-header">Model Comparison &amp; Training</div>', unsafe_allow_html=True)
    st.write(
        "Train every regression and classification model in one step, compare their "
        "performance, and save the best of each to disk for reuse across the app."
    )

    df = get_active_dataset()
    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        else:
            if st.button("Train All Models & Save Best", type="primary"):
                with st.spinner("Training all regression and classification models..."):
                    cleaned = prepare_cleaned_dataset(df)
                    X = cleaned[FEATURE_COLUMNS]
                    y_reg = cleaned[TARGET_REGRESSION]
                    y_clf = cleaned[TARGET_CLASSIFICATION]

                    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
                        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
                    )
                    reg_pipelines, reg_results = train_and_evaluate_regressors(
                        X_train, X_test, y_reg_train, y_reg_test
                    )
                    best_reg_name, best_reg_pipeline = get_best_regression_model(reg_pipelines, reg_results)

                    X_train_c, X_test_c, y_clf_train, y_clf_test = train_test_split(
                        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
                    )
                    clf_pipelines, clf_results, clf_matrices = train_and_evaluate_classifiers(
                        X_train_c, X_test_c, y_clf_train, y_clf_test
                    )
                    best_clf_name, best_clf_pipeline = get_best_classification_model(clf_pipelines, clf_results)

                    metadata = {
                        "best_regression_model_name": best_reg_name,
                        "best_classification_model_name": best_clf_name,
                        "regression_results": reg_results.to_dict(orient="records"),
                        "classification_results": clf_results.to_dict(orient="records"),
                        "feature_columns": FEATURE_COLUMNS,
                    }
                    save_models(best_reg_pipeline, best_clf_pipeline, metadata)

                    st.session_state.reg_results = reg_results
                    st.session_state.reg_pipelines = reg_pipelines
                    st.session_state.reg_X_test = X_test
                    st.session_state.reg_y_test = y_reg_test
                    st.session_state.best_reg_name = best_reg_name
                    st.session_state.best_reg_pipeline = best_reg_pipeline

                    st.session_state.clf_results = clf_results
                    st.session_state.clf_pipelines = clf_pipelines
                    st.session_state.clf_matrices = clf_matrices
                    st.session_state.best_clf_name = best_clf_name
                    st.session_state.best_clf_pipeline = best_clf_pipeline
                    st.session_state.trained = True

                st.success("All models trained and best models saved to the models/ directory.")

            reg_pipeline, clf_pipeline, metadata = load_models()
            if metadata is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Best Regression Model")
                    st.info(metadata["best_regression_model_name"])
                    st.dataframe(pd.DataFrame(metadata["regression_results"]), use_container_width=True, hide_index=True)
                with col2:
                    st.markdown("#### Best Classification Model")
                    st.info(metadata["best_classification_model_name"])
                    st.dataframe(pd.DataFrame(metadata["classification_results"]), use_container_width=True, hide_index=True)
            else:
                st.info("No saved models yet. Click the button above to train and save models.")


# ==========================================================================
# PAGE: Feature Importance
# ==========================================================================
elif page == "Feature Importance":
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    st.write("Feature importance is available for tree-based models (Random Forest, Gradient Boosting, Decision Tree).")

    reg_pipeline, clf_pipeline, metadata = load_models()

    if reg_pipeline is None:
        st.warning("No trained models found. Please train models in the 'Model Comparison' section first.")
    else:
        try:
            preprocessor = reg_pipeline.named_steps["preprocessor"]
            feature_names = get_transformed_feature_names(preprocessor)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### Regression Model: {metadata['best_regression_model_name']}")
                reg_importance = get_feature_importance_df(reg_pipeline, feature_names)
                if reg_importance is not None:
                    st.plotly_chart(feature_importance_chart(reg_importance), use_container_width=True)
                else:
                    st.info("The best regression model does not natively support feature importance (e.g. linear models). Consider the coefficients instead.")
                    model = reg_pipeline.named_steps["model"]
                    if hasattr(model, "coef_"):
                        coef_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Coefficient": model.coef_,
                        }).sort_values("Coefficient", key=abs, ascending=False)
                        st.dataframe(coef_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown(f"#### Classification Model: {metadata['best_classification_model_name']}")
                clf_importance = get_feature_importance_df(clf_pipeline, feature_names)
                if clf_importance is not None:
                    st.plotly_chart(feature_importance_chart(clf_importance), use_container_width=True)
                else:
                    st.info("The best classification model does not natively support feature importance.")
        except Exception as exc:
            st.error(f"Could not compute feature importance: {exc}")


# ==========================================================================
# PAGE: Individual Student Prediction
# ==========================================================================
elif page == "Individual Student Prediction":
    st.markdown('<div class="section-header">Individual Student Prediction</div>', unsafe_allow_html=True)
    st.write("Enter a student's details below to predict their final academic performance.")

    reg_pipeline, clf_pipeline, metadata = load_models()

    if reg_pipeline is None:
        st.warning("No trained models found. Please train and save models in the 'Model Comparison' section first.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=15, max_value=35, value=20)
                gender = st.selectbox("Gender", ["Male", "Female"])
                study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)

            with col2:
                attendance = st.slider("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
                assignments_score = st.slider("Assignments Score", min_value=0.0, max_value=100.0, value=70.0)
                midterm_score = st.slider("Midterm Score", min_value=0.0, max_value=100.0, value=65.0)

            with col3:
                previous_score = st.slider("Previous Score", min_value=0.0, max_value=100.0, value=65.0)
                sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
                extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

            submitted = st.form_submit_button("Predict Student Performance", type="primary")

        if submitted:
            try:
                student_input = {
                    "Age": age,
                    "Gender": gender,
                    "Study_Hours": study_hours,
                    "Attendance": attendance,
                    "Assignments_Score": assignments_score,
                    "Midterm_Score": midterm_score,
                    "Previous_Score": previous_score,
                    "Sleep_Hours": sleep_hours,
                    "Internet_Access": internet_access,
                    "Extracurricular_Activities": extracurricular,
                }
                input_df = pd.DataFrame([student_input])[FEATURE_COLUMNS]

                predicted_score = float(reg_pipeline.predict(input_df)[0])
                predicted_score = max(0.0, min(100.0, predicted_score))
                risk_status = clf_pipeline.predict(input_df)[0]
                category = categorize_performance(predicted_score)
                explanation = explain_prediction(student_input, predicted_score)

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Final Score", f"{predicted_score:.1f}")
                col2.metric("At-Risk Status", risk_status)
                col3.metric("Performance Category", category)

                if risk_status == "At Risk":
                    st.error(f"This student is predicted to be **At Risk**. {explanation}")
                elif category in ("Excellent", "Good"):
                    st.success(f"This student is predicted to perform well. {explanation}")
                else:
                    st.warning(f"This student's predicted performance is moderate. {explanation}")

                st.caption(
                    "This explanation is a rule-based summary of the input values and does not "
                    "constitute a medical, psychological, or professional diagnosis."
                )
            except Exception as exc:
                st.error(f"Could not generate a prediction: {exc}")


# ==========================================================================
# PAGE: Download Results
# ==========================================================================
elif page == "Download Results":
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)

    df = get_active_dataset()
    reg_pipeline, clf_pipeline, metadata = load_models()

    if df is None:
        st.warning("No dataset available. Please upload a CSV in the 'Dataset Upload' section.")
    else:
        is_valid, missing_cols = validate_columns(df)
        if not is_valid:
            st.error("Dataset is missing required columns: " + ", ".join(missing_cols))
        elif reg_pipeline is None:
            st.warning("No trained models found. Please train and save models in the 'Model Comparison' section first.")
        else:
            try:
                cleaned = prepare_cleaned_dataset(df)
                X = cleaned[FEATURE_COLUMNS]

                predictions_df = cleaned.copy()
                predictions_df["Predicted_Final_Score"] = np.round(reg_pipeline.predict(X), 1)
                predictions_df["Predicted_At_Risk"] = clf_pipeline.predict(X)
                predictions_df["Performance_Category"] = predictions_df["Predicted_Final_Score"].apply(categorize_performance)

                st.markdown("#### Predictions Preview")
                st.dataframe(predictions_df.head(20), use_container_width=True)

                csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="student_performance_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Could not generate predictions for download: {exc}")


# ==========================================================================
# PAGE: About Project
# ==========================================================================
elif page == "About Project":
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
**Student Performance Intelligence & Prediction System** is a machine learning
web application that analyzes student academic data to predict final
performance and identify students who may be academically at risk.

**Machine Learning Approach**

- **Regression models** (Linear, Ridge, Lasso, Random Forest, Gradient Boosting) predict a
  continuous `Final_Score`.
- **Classification models** (Logistic Regression, Decision Tree, Random Forest, Gradient
  Boosting) predict a binary `At_Risk` label (`Final_Score` below 50).
- A shared `ColumnTransformer` pipeline handles missing values, scaling, and
  categorical encoding consistently across every model.
- The best model of each type is automatically selected (by R² for
  regression, F1 score for classification) and persisted with `joblib`.

**Tech Stack**

Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Seaborn, Joblib.

**Disclaimer**

Predictions and explanations generated by this application are statistical
estimates based on historical patterns in the data. They are intended for
educational and portfolio purposes and do not constitute academic,
psychological, or professional advice.
        """
    )
    st.markdown(
        '<div class="footer">Built with Python, Scikit-learn and Streamlit</div>',
        unsafe_allow_html=True,
    )
