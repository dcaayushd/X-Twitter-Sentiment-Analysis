"""Derived metrics for dashboard and reporting use cases."""

from __future__ import annotations

import math

import pandas as pd


def add_rolling_average(
    frame: pd.DataFrame,
    window: int = 3,
    source_column: str = "average_sentiment",
) -> pd.DataFrame:
    """Append a rolling average column to a time-series frame."""
    if frame.empty:
        frame["rolling_average"] = []
        return frame
    enriched = frame.copy()
    enriched["rolling_average"] = (
        enriched[source_column].rolling(window=window, min_periods=1).mean()
    )
    return enriched


def build_summary_stats(summary_row: dict, distribution_frame: pd.DataFrame) -> dict:
    """Merge repository summary stats with derived distribution metrics."""
    if not summary_row:
        return {
            "total_tweets": 0,
            "average_sentiment": 0.0,
            "average_confidence": 0.0,
            "dominant_sentiment": "neutral",
            "positive_percentage": 0.0,
            "negative_percentage": 0.0,
            "neutral_percentage": 0.0,
            "earliest_tweet": None,
            "latest_tweet": None,
        }

    stats = dict(summary_row)
    stats["average_sentiment"] = _safe_float(stats.get("average_sentiment"))
    stats["average_confidence"] = _safe_float(stats.get("average_confidence"))

    percentages = {row["label"]: row["percentage"] for _, row in distribution_frame.iterrows()}
    stats["positive_percentage"] = float(percentages.get("positive", 0.0))
    stats["negative_percentage"] = float(percentages.get("negative", 0.0))
    stats["neutral_percentage"] = float(percentages.get("neutral", 0.0))
    stats["dominant_sentiment"] = (
        str(distribution_frame.sort_values("count", ascending=False).iloc[0]["label"])
        if not distribution_frame.empty
        else "neutral"
    )
    return stats


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    result = float(value)
    return 0.0 if math.isnan(result) else result
