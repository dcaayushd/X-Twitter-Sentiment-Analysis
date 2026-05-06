"""Repository layer for SQLite CRUD and analytical queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from app.schemas import AnalyzedTweet
from storage.database import DatabaseManager


class TweetRepository:
    """Persist and query tweets and ingestion runs."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def create_ingestion_run(self, source_query: str, model_name: str) -> int:
        """Create a run record before pipeline execution starts."""
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO ingestion_runs (source_query, model_name, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (source_query, model_name, started_at),
            )
            return int(cursor.lastrowid)

    def complete_ingestion_run(self, run_id: int, total_collected: int, total_stored: int) -> None:
        """Mark an ingestion run as complete."""
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET status = 'completed',
                    total_collected = ?,
                    total_stored = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (total_collected, total_stored, completed_at, run_id),
            )

    def fail_ingestion_run(self, run_id: int, error_message: str) -> None:
        """Mark an ingestion run as failed."""
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET status = 'failed',
                    error_message = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (error_message, completed_at, run_id),
            )

    def upsert_tweets(self, tweets: Iterable[AnalyzedTweet]) -> int:
        """Insert or update analyzed tweets."""
        payload = []
        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        for tweet in tweets:
            payload.append(
                (
                    tweet.raw_tweet.tweet_id,
                    tweet.raw_tweet.username,
                    tweet.raw_tweet.display_name,
                    tweet.raw_tweet.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    tweet.raw_tweet.content,
                    tweet.cleaned_text,
                    json.dumps(tweet.tokens),
                    tweet.sentiment.label,
                    float(tweet.sentiment.score),
                    float(tweet.sentiment.confidence),
                    int(tweet.raw_tweet.like_count),
                    int(tweet.raw_tweet.retweet_count),
                    tweet.raw_tweet.lang,
                    json.dumps(tweet.raw_tweet.raw_payload),
                    updated_at,
                )
            )

        if not payload:
            return 0

        with self.database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO tweets (
                    tweet_id,
                    username,
                    display_name,
                    created_at,
                    content,
                    cleaned_text,
                    tokens,
                    sentiment_label,
                    sentiment_score,
                    sentiment_confidence,
                    like_count,
                    retweet_count,
                    lang,
                    raw_payload,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tweet_id) DO UPDATE SET
                    username = excluded.username,
                    display_name = excluded.display_name,
                    created_at = excluded.created_at,
                    content = excluded.content,
                    cleaned_text = excluded.cleaned_text,
                    tokens = excluded.tokens,
                    sentiment_label = excluded.sentiment_label,
                    sentiment_score = excluded.sentiment_score,
                    sentiment_confidence = excluded.sentiment_confidence,
                    like_count = excluded.like_count,
                    retweet_count = excluded.retweet_count,
                    lang = excluded.lang,
                    raw_payload = excluded.raw_payload,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
        return len(payload)

    def link_tweets_to_run(self, run_id: int, tweet_ids: list[str]) -> None:
        """Associate stored tweets with an ingestion run."""
        if not tweet_ids:
            return
        ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        payload = [(run_id, tweet_id, ingested_at) for tweet_id in tweet_ids]
        with self.database.connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO ingestion_run_tweets (run_id, tweet_id, ingested_at)
                VALUES (?, ?, ?)
                """,
                payload,
            )

    def fetch_time_series(self, granularity: str, run_id: int) -> list[dict]:
        """Return aggregated sentiment metrics for hourly or daily buckets."""
        if granularity not in {"hourly", "daily"}:
            raise ValueError("granularity must be either `hourly` or `daily`.")
        period_expression = (
            "strftime('%Y-%m-%d %H:00:00', t.created_at)"
            if granularity == "hourly"
            else "strftime('%Y-%m-%d 00:00:00', t.created_at)"
        )

        query = f"""
            SELECT
                {period_expression} AS period,
                COUNT(*) AS total_tweets,
                AVG(t.sentiment_score) AS average_sentiment,
                SUM(CASE WHEN t.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
                SUM(CASE WHEN t.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
                SUM(CASE WHEN t.sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral_count
            FROM tweets t
            INNER JOIN ingestion_run_tweets irt ON irt.tweet_id = t.tweet_id
            WHERE irt.run_id = ?
            GROUP BY period
            ORDER BY period
        """
        with self.database.connect() as connection:
            rows = connection.execute(query, (run_id,)).fetchall()
        return [dict(row) for row in rows]

    def fetch_sentiment_distribution(self, run_id: int) -> list[dict]:
        """Return sentiment counts for a run."""
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT t.sentiment_label AS label, COUNT(*) AS count
                FROM tweets t
                INNER JOIN ingestion_run_tweets irt ON irt.tweet_id = t.tweet_id
                WHERE irt.run_id = ?
                GROUP BY t.sentiment_label
                ORDER BY count DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_summary_stats(self, run_id: int) -> dict:
        """Return top-level summary metrics for a run."""
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_tweets,
                    AVG(t.sentiment_score) AS average_sentiment,
                    AVG(t.sentiment_confidence) AS average_confidence,
                    MIN(t.created_at) AS earliest_tweet,
                    MAX(t.created_at) AS latest_tweet
                FROM tweets t
                INNER JOIN ingestion_run_tweets irt ON irt.tweet_id = t.tweet_id
                WHERE irt.run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return dict(row) if row else {}

    def fetch_recent_tweets(self, run_id: int, limit: int = 100) -> list[dict]:
        """Return recently created tweets for a run."""
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.tweet_id,
                    t.created_at,
                    t.username,
                    t.display_name,
                    t.content,
                    t.cleaned_text,
                    t.sentiment_label,
                    t.sentiment_score,
                    t.sentiment_confidence
                FROM tweets t
                INNER JOIN ingestion_run_tweets irt ON irt.tweet_id = t.tweet_id
                WHERE irt.run_id = ?
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
