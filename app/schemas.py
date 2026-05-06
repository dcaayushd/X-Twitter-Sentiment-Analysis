"""Shared schemas used across the sentiment tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class SearchParameters:
    """Validated search parameters for a Twitter/X scraping run."""

    keyword: str | None = None
    hashtags: list[str] = field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    max_results: int = 100

    def normalized_hashtags(self) -> list[str]:
        """Return hashtags normalized with a leading `#`."""
        normalized: list[str] = []
        for hashtag in self.hashtags:
            value = hashtag.strip()
            if not value:
                continue
            normalized.append(value if value.startswith("#") else f"#{value}")
        return normalized

    def validate(self) -> None:
        """Validate the search parameters."""
        if not self.keyword and not self.normalized_hashtags():
            raise ValueError("At least one keyword or hashtag must be provided.")
        if self.max_results <= 0:
            raise ValueError("max_results must be a positive integer.")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")

    @property
    def source_query(self) -> str:
        """Build a canonical string used to identify an ingestion scope."""
        keyword = self.keyword.strip() if self.keyword else ""
        hashtags = ",".join(self.normalized_hashtags())
        start = self.start_date.isoformat() if self.start_date else ""
        end = self.end_date.isoformat() if self.end_date else ""
        return (
            f"keyword={keyword};hashtags={hashtags};start={start};"
            f"end={end};limit={self.max_results}"
        )


@dataclass(frozen=True)
class RawTweet:
    """Raw tweet metadata returned by the ingestion layer."""

    tweet_id: str
    created_at: datetime
    username: str
    display_name: str
    content: str
    url: str
    like_count: int = 0
    retweet_count: int = 0
    lang: str = "unknown"
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentimentPrediction:
    """Prediction returned by a sentiment model."""

    label: str
    score: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyzedTweet:
    """A tweet enriched with processed text and sentiment output."""

    raw_tweet: RawTweet
    cleaned_text: str
    tokens: list[str]
    sentiment: SentimentPrediction


@dataclass(frozen=True)
class PipelineRunResult:
    """Outcome of an ingestion and analysis run."""

    run_id: int
    source_query: str
    model_name: str
    collected_count: int
    stored_count: int
    raw_export_path: str | None = None
    processed_export_path: str | None = None
