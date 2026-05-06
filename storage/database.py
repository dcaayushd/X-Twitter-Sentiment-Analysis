"""SQLite database management."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class DatabaseManager:
    """Manage SQLite connections and schema initialization."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection with sane defaults."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create database tables and indexes if needed."""
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tweets (
                    tweet_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    url TEXT NOT NULL DEFAULT '',
                    cleaned_text TEXT NOT NULL,
                    tokens TEXT NOT NULL,
                    sentiment_label TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    sentiment_confidence REAL NOT NULL,
                    like_count INTEGER NOT NULL DEFAULT 0,
                    retweet_count INTEGER NOT NULL DEFAULT 0,
                    lang TEXT NOT NULL,
                    raw_payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_query TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    requested_backend TEXT NOT NULL DEFAULT 'unknown',
                    resolved_backend TEXT NOT NULL DEFAULT 'unknown',
                    served_backend TEXT NOT NULL DEFAULT 'unknown',
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    total_collected INTEGER NOT NULL DEFAULT 0,
                    total_stored INTEGER NOT NULL DEFAULT 0,
                    raw_export_path TEXT,
                    processed_export_path TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS ingestion_run_tweets (
                    run_id INTEGER NOT NULL,
                    tweet_id TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, tweet_id),
                    FOREIGN KEY (run_id) REFERENCES ingestion_runs(id) ON DELETE CASCADE,
                    FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at);
                CREATE INDEX IF NOT EXISTS idx_tweets_sentiment_label ON tweets(sentiment_label);
                CREATE INDEX IF NOT EXISTS idx_runs_source_query ON ingestion_runs(source_query);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON ingestion_runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON ingestion_runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_run_tweets_run_id ON ingestion_run_tweets(run_id);
                CREATE INDEX IF NOT EXISTS idx_run_tweets_tweet_id ON ingestion_run_tweets(tweet_id);
                """
            )
            self._migrate_table_columns(
                connection,
                table_name="tweets",
                column_definitions={
                    "url": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._migrate_table_columns(
                connection,
                table_name="ingestion_runs",
                column_definitions={
                    "requested_backend": "TEXT NOT NULL DEFAULT 'unknown'",
                    "resolved_backend": "TEXT NOT NULL DEFAULT 'unknown'",
                    "served_backend": "TEXT NOT NULL DEFAULT 'unknown'",
                    "fallback_used": "INTEGER NOT NULL DEFAULT 0",
                    "raw_export_path": "TEXT",
                    "processed_export_path": "TEXT",
                },
            )

    @staticmethod
    def _migrate_table_columns(
        connection: sqlite3.Connection,
        table_name: str,
        column_definitions: dict[str, str],
    ) -> None:
        """Add missing columns to an existing table without dropping local data."""
        existing_columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column_name, definition in column_definitions.items():
            if column_name in existing_columns:
                continue
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )
