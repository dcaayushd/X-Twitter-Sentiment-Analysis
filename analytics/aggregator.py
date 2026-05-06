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


def build_source_breakdown_frame(rows: list[dict]) -> pd.DataFrame:
    """Convert repository source rows into a chart-ready DataFrame."""
    columns = [
        "display_name",
        "username",
        "tweet_count",
        "average_sentiment",
        "average_confidence",
        "average_engagement",
        "first_seen",
        "last_seen",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    for column in ("first_seen", "last_seen"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("average_sentiment", "average_confidence", "average_engagement"):
        frame[column] = frame[column].fillna(0.0).astype(float)
    frame["tweet_count"] = frame["tweet_count"].fillna(0).astype(int)
    return frame.sort_values(["tweet_count", "average_sentiment"], ascending=[False, False])


def build_run_history_frame(rows: list[dict]) -> pd.DataFrame:
    """Convert recent run rows into a display-friendly DataFrame."""
    columns = [
        "id",
        "status",
        "model_name",
        "requested_backend",
        "resolved_backend",
        "served_backend",
        "fallback_used",
        "total_collected",
        "total_stored",
        "started_at",
        "completed_at",
        "source_query",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    for column in ("started_at", "completed_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["fallback_used"] = frame["fallback_used"].fillna(0).astype(bool)
    return frame
