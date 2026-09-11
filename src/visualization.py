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

# ---------------------------------------------------------------------------
# Global style constants
# ---------------------------------------------------------------------------
TEMPLATE = "plotly_white"

# A refined, professional color palette (deep blue / teal / slate / amber)
COLOR_SEQUENCE = [
    "#2563EB",  # blue
    "#0EA5A4",  # teal
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # violet
    "#10B981",  # green
    "#64748B",  # slate
    "#EC4899",  # pink
]

FONT_FAMILY = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
TITLE_COLOR = "#1E293B"
AXIS_COLOR = "#334155"
GRID_COLOR = "#E2E8F0"
PAPER_BG = "#FFFFFF"
PLOT_BG = "#FFFFFF"


def _apply_base_style(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """Apply a consistent, polished look-and-feel to any figure."""
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        font=dict(family=FONT_FAMILY, size=13, color=AXIS_COLOR),
        title=dict(
            text=title,
            font=dict(family=FONT_FAMILY, size=19, color=TITLE_COLOR),
            x=0.02,
            xanchor="left",
        ) if title else None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family=FONT_FAMILY,
            bordercolor=GRID_COLOR,
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
        zeroline=False, linecolor=GRID_COLOR, ticks="outside",
        tickcolor=GRID_COLOR, title_font=dict(size=13, color=AXIS_COLOR),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
        zeroline=False, linecolor=GRID_COLOR, ticks="outside",
        tickcolor=GRID_COLOR, title_font=dict(size=13, color=AXIS_COLOR),
    )
    return fig


def final_score_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Final_Score", nbins=30,
        color_discrete_sequence=[COLOR_SEQUENCE[0]],
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1,
        opacity=0.9,
        hovertemplate="Final Score: %{x}<br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(bargap=0.08)
    fig.add_vline(
        x=df["Final_Score"].mean(),
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {df['Final_Score'].mean():.1f}",
        annotation_font_color="#EF4444",
        annotation_position="top",
    )
    _apply_base_style(fig, title="Distribution of Final Score")
    fig.update_xaxes(title_text="Final Score")
    fig.update_yaxes(title_text="Number of Students")
    return fig


def scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col, trendline="ols",
        color_discrete_sequence=[COLOR_SEQUENCE[0]],
        opacity=0.55,
    )
    fig.update_traces(
        selector=dict(mode="markers"),
        marker=dict(size=8, line=dict(width=0.5, color="white")),
        hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(mode="lines"),
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        name="Trend line",
    )
    _apply_base_style(fig, title=title)
    fig.update_xaxes(title_text=x_col.replace("_", " "))
    fig.update_yaxes(title_text=y_col.replace("_", " "))
    return fig


def category_distribution(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.pie(
        counts, names=col, values="Count",
        color_discrete_sequence=COLOR_SEQUENCE, hole=0.55,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>",
    )
    _apply_base_style(fig, title=title)
    fig.update_layout(showlegend=True, legend=dict(orientation="v"))
    return fig


def average_score_by_group(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    avg = df.groupby(group_col, observed=True)["Final_Score"].mean().reset_index()
    avg = avg.sort_values("Final_Score", ascending=False)
    fig = px.bar(
        avg, x=group_col, y="Final_Score",
        color=group_col, color_discrete_sequence=COLOR_SEQUENCE,
        text_auto=".1f",
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1,
        textfont=dict(size=12, color=TITLE_COLOR),
        textposition="outside",
        hovertemplate=f"{group_col}: %{{x}}<br>Avg Final Score: %{{y:.1f}}<extra></extra>",
    )
    _apply_base_style(fig, title=title)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text=group_col.replace("_", " "))
    fig.update_yaxes(title_text="Average Final Score")
    return fig


def correlation_heatmap(df: pd.DataFrame, numerical_cols: list[str]) -> go.Figure:
    corr = df[numerical_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
    )
    fig.update_traces(
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        textfont=dict(size=11),
    )
    _apply_base_style(fig, title="Correlation Heatmap (Numerical Features)", height=520)
    fig.update_xaxes(side="bottom", tickangle=-40)
    fig.update_layout(coloraxis_colorbar=dict(title="Corr", thickness=15))
    return fig


def actual_vs_predicted_chart(comparison_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison_df["Actual"], y=comparison_df["Predicted"],
        mode="markers", name="Predictions",
        marker=dict(
            color=COLOR_SEQUENCE[0], opacity=0.6, size=9,
            line=dict(width=0.5, color="white"),
        ),
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
    ))
    min_val = min(comparison_df["Actual"].min(), comparison_df["Predicted"].min())
    max_val = max(comparison_df["Actual"].max(), comparison_df["Predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        hoverinfo="skip",
    ))
    _apply_base_style(fig, title="Actual vs Predicted Final Score", height=460)
    fig.update_xaxes(title_text="Actual Score")
    fig.update_yaxes(title_text="Predicted Score")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def feature_importance_chart(importances_df: pd.DataFrame) -> go.Figure:
    sorted_df = importances_df.sort_values("Importance", ascending=True)
    fig = px.bar(
        sorted_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#93C5FD", "#2563EB"],
    )
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )
    _apply_base_style(
        fig, title="Feature Importance",
        height=max(400, len(importances_df) * 32),
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="")
    return fig


def confusion_matrix_chart(cm, labels: list[str], title: str) -> go.Figure:
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_traces(
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        textfont=dict(size=14, color="white"),
    )
    _apply_base_style(fig, title=title)
    fig.update_layout(coloraxis_colorbar=dict(title="Count", thickness=15))
    return fig
