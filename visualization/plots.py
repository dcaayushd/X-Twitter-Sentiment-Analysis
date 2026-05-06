"""Plotly chart builders for sentiment analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SENTIMENT_COLORS = {
    "positive": "#2E8B57",
    "negative": "#C0392B",
    "neutral": "#7F8C8D",
}


def build_sentiment_trend_chart(frame: pd.DataFrame) -> go.Figure:
    """Build a combined time-series chart for sentiment and tweet volume."""
    if frame.empty:
        return _empty_figure("No tweets available for the selected time range.")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=frame["period"],
            y=frame["total_tweets"],
            name="Tweet Volume",
            marker_color="#B0BEC5",
            opacity=0.4,
        ),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["average_sentiment"],
            mode="lines+markers",
            name="Average Sentiment",
            line={"color": "#1F77B4", "width": 3},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["period"],
            y=frame["rolling_average"],
            mode="lines",
            name="Rolling Average",
            line={"color": "#FF7F0E", "width": 2, "dash": "dash"},
        ),
        secondary_y=False,
    )
    figure.update_layout(
        title="Sentiment Trend Over Time",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    figure.update_yaxes(title_text="Sentiment Score", range=[-1, 1], secondary_y=False)
    figure.update_yaxes(title_text="Tweet Volume", secondary_y=True)
    figure.update_xaxes(title_text="Time")
    return figure


def build_sentiment_distribution_chart(
    frame: pd.DataFrame,
    chart_type: str = "pie",
) -> go.Figure:
    """Build a sentiment distribution chart."""
    if frame.empty:
        return _empty_figure("No sentiment distribution is available yet.")

    if chart_type == "bar":
        figure = px.bar(
            frame,
            x="label",
            y="count",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment Distribution",
        )
    else:
        figure = px.pie(
            frame,
            names="label",
            values="count",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment Distribution",
            hole=0.45,
        )
    figure.update_layout(template="plotly_white", showlegend=True)
    return figure


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
    )
    return figure
