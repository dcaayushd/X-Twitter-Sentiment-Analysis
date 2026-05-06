"""Time-based aggregation helpers for sentiment results."""

from __future__ import annotations

import pandas as pd


def build_timeseries_frame(rows: list[dict], granularity: str) -> pd.DataFrame:
    """Convert repository time-series rows into a chart-ready DataFrame."""
    columns = [
        "period",
        "total_tweets",
        "average_sentiment",
        "positive_count",
        "negative_count",
        "neutral_count",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    frame["period"] = pd.to_datetime(frame["period"], utc=True)
    frame = frame.sort_values("period")

    frequency = "h" if granularity == "hourly" else "D"
    full_range = pd.date_range(
        start=frame["period"].min(),
        end=frame["period"].max(),
        freq=frequency,
        tz="UTC",
    )

    frame = frame.set_index("period").reindex(full_range).rename_axis("period").reset_index()
    for count_column in ("total_tweets", "positive_count", "negative_count", "neutral_count"):
        frame[count_column] = frame[count_column].fillna(0).astype(int)
    frame["average_sentiment"] = frame["average_sentiment"].fillna(0.0)
    return frame


def build_distribution_frame(rows: list[dict]) -> pd.DataFrame:
    """Convert repository distribution rows into a normalized DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["label", "count", "percentage"])

    frame = pd.DataFrame(rows)
    total = frame["count"].sum()
    frame["percentage"] = (frame["count"] / total * 100.0) if total else 0.0
    return frame
