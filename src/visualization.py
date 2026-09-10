"""
visualization.py

Reusable Plotly chart builders for the EDA and Feature Importance pages.
Keeping chart construction here avoids duplicating styling code inside
app.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Safe


def final_score_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Final_Score", nbins=30, template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Distribution of Final Score",
    )
    fig.update_layout(bargap=0.05, height=400)
    return fig


def scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col, trendline="ols", template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE, title=title, opacity=0.6,
    )
    fig.update_layout(height=400)
    return fig


def category_distribution(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.pie(
        counts, names=col, values="Count", template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE, title=title, hole=0.4,
    )
    fig.update_layout(height=400)
    return fig


def average_score_by_group(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    avg = df.groupby(group_col, observed=True)["Final_Score"].mean().reset_index()
    fig = px.bar(
        avg, x=group_col, y="Final_Score", template=TEMPLATE,
        color=group_col, color_discrete_sequence=COLOR_SEQUENCE, title=title,
        text_auto=".1f",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def correlation_heatmap(df: pd.DataFrame, numerical_cols: list[str]) -> go.Figure:
    corr = df[numerical_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", template=TEMPLATE, color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (Numerical Features)", aspect="auto", zmin=-1, zmax=1,
    )
    fig.update_layout(height=500)
    return fig


def actual_vs_predicted_chart(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison_df["Actual"], y=comparison_df["Predicted"],
        mode="markers", name="Predictions",
        marker=dict(color=COLOR_SEQUENCE[0], opacity=0.6),
    ))
    min_val = min(comparison_df["Actual"].min(), comparison_df["Predicted"].min())
    max_val = max(comparison_df["Actual"].max(), comparison_df["Predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(
        title="Actual vs Predicted Final Score", xaxis_title="Actual Score",
        yaxis_title="Predicted Score", template=TEMPLATE, height=450,
    )
    return fig


def feature_importance_chart(importances_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importances_df.sort_values("Importance", ascending=True),
        x="Importance", y="Feature", orientation="h", template=TEMPLATE,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Feature Importance",
    )
    fig.update_layout(height=max(400, len(importances_df) * 30))
    return fig


def confusion_matrix_chart(cm, labels: list[str], title: str) -> go.Figure:
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels, template=TEMPLATE,
        color_continuous_scale="Blues", title=title,
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(height=400)
    return fig
