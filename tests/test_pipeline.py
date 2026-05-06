"""Tests for pipeline dataset selection and run orchestration helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from app.config import (
    AppConfig,
    DashboardConfig,
    DatabaseConfig,
    IngestionConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    ProcessingConfig,
    SearchConfig,
)
from app.pipeline import SentimentTrackingPipeline, build_training_corpus, resolve_training_dataset_paths
from app.schemas import IngestionBatch, RawTweet, SearchParameters, SentimentPrediction
from storage.database import DatabaseManager
from storage.repository import TweetRepository


class _FakeCleaner:
    def clean_text(self, text: str) -> str:
        return text.lower().strip()


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


class _FakeModel:
    def predict(self, text: str) -> SentimentPrediction:
        return SentimentPrediction(label="positive", score=0.7, confidence=0.9)


class _FakeModelSelector:
    def get_model(self, model_name: str) -> _FakeModel:
        return _FakeModel()


class _FakeScraper:
    def resolve_backend_request(self, backend_override: str | None = None) -> tuple[str, str]:
        requested = backend_override or "auto"
        resolved = "twscrape" if requested == "auto" else requested
        return requested, resolved

    def fetch_batch(
        self,
        search_parameters: SearchParameters,
        backend_override: str | None = None,
    ) -> IngestionBatch:
        requested = backend_override or "auto"
        resolved = "twscrape" if requested == "auto" else requested
        return IngestionBatch(
            tweets=[
                RawTweet(
                    tweet_id="tweet-1",
                    created_at=datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
                    username="openai",
                    display_name="OpenAI",
                    content="New release looks strong",
                    url="https://x.com/openai/status/1",
                    like_count=42,
                    retweet_count=5,
                    lang="en",
                    raw_payload={"tweet_id": "tweet-1"},
                )
            ],
            requested_backend=requested,
            resolved_backend=resolved,
            served_backend="news_rss",
            fallback_used=True,
        )


def test_resolve_training_dataset_paths_prefers_all_local_corpora(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    configured = processed_dir / "training_sample.csv"
    pd.DataFrame(
        [
            {"text": "love this", "label": "positive"},
            {"text": "hate this", "label": "negative"},
        ]
    ).to_csv(configured, index=False)

    older = processed_dir / "older_processed.csv"
    pd.DataFrame(
        [
            {"cleaned_text": "this is great", "sentiment_label": "positive"},
            {"cleaned_text": "this is bad", "sentiment_label": "negative"},
        ]
    ).to_csv(older, index=False)

    latest = processed_dir / "latest_processed.csv"
    pd.DataFrame(
        [
            {"cleaned_text": "nice update", "sentiment_label": "positive"},
            {"cleaned_text": "awful update", "sentiment_label": "negative"},
            {"cleaned_text": "okay update", "sentiment_label": "neutral"},
        ]
    ).to_csv(latest, index=False)

    selected = resolve_training_dataset_paths(
        explicit_path=None,
        configured_path=configured,
        processed_output_dir=processed_dir,
    )

    assert selected == [
        configured.resolve(),
        latest.resolve(),
        older.resolve(),
    ]


def test_resolve_training_dataset_paths_respects_explicit_dataset(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    explicit = tmp_path / "custom.csv"
    explicit.write_text("text,label\nhello,positive\nbye,negative\n", encoding="utf-8")

    selected = resolve_training_dataset_paths(
        explicit_path=explicit,
        configured_path=processed_dir / "training_sample.csv",
        processed_output_dir=processed_dir,
    )

    assert selected == [explicit.resolve()]


def test_build_training_corpus_merges_and_deduplicates_frames(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        [
            {"text": "I love this launch", "label": "positive"},
            {"text": "This is awful", "label": "negative"},
        ]
    ).to_csv(first, index=False)
    pd.DataFrame(
        [
            {
                "cleaned_text": "i love this launch",
                "sentiment_label": "positive",
                "sentiment_confidence": 0.8,
                "sentiment_score": 0.7,
            },
            {
                "cleaned_text": "this is awful",
                "sentiment_label": "negative",
                "sentiment_confidence": 0.9,
                "sentiment_score": -0.8,
            },
        ]
    ).to_csv(second, index=False)

    corpus = build_training_corpus([first, second])

    assert len(corpus) == 2
    assert set(corpus["label"]) == {"positive", "negative"}
    assert corpus["sample_weight"].min() > 0


def test_pipeline_run_persists_backend_metadata_and_exports(tmp_path) -> None:
    database = DatabaseManager(tmp_path / "sentiment.db")
    database.initialize()
    repository = TweetRepository(database)
    config = AppConfig(
        project_root=tmp_path,
        ingestion=IngestionConfig(backend="auto"),
        search=SearchConfig(
            keyword="openai",
            hashtags=["AI"],
            start_date=None,
            end_date=None,
            max_results=25,
        ),
        processing=ProcessingConfig(spacy_model="blank_en"),
        model=ModelConfig(
            active_model="vader",
            ml_model_path=tmp_path / "sentiment_model.joblib",
            training_data_path=tmp_path / "training.csv",
            auto_train_if_missing=False,
        ),
        database=DatabaseConfig(path=tmp_path / "sentiment.db"),
        logging=LoggingConfig(),
        paths=PathsConfig(
            raw_output_dir=tmp_path / "raw",
            processed_output_dir=tmp_path / "processed",
        ),
        dashboard=DashboardConfig(title="Test Dashboard"),
    )
    pipeline = SentimentTrackingPipeline(
        config=config,
        repository=repository,
        scraper=_FakeScraper(),
        cleaner=_FakeCleaner(),
        tokenizer=_FakeTokenizer(),
        model_selector=_FakeModelSelector(),
        logger=logging.getLogger("test.pipeline"),
    )

    result = pipeline.run(
        search_parameters=SearchParameters(keyword="openai", hashtags=["AI"], max_results=25),
        model_name="vader",
        backend_override="auto",
    )

    assert result.requested_backend == "auto"
    assert result.resolved_backend == "twscrape"
    assert result.served_backend == "news_rss"
    assert result.fallback_used is True
    assert result.collected_count == 1
    assert result.stored_count == 1
    assert result.raw_export_path is not None
    assert result.processed_export_path is not None

    run_details = repository.fetch_run_details(result.run_id)
    assert run_details["requested_backend"] == "auto"
    assert run_details["resolved_backend"] == "twscrape"
    assert run_details["served_backend"] == "news_rss"
    assert run_details["fallback_used"] == 1
    assert run_details["raw_export_path"]
    assert run_details["processed_export_path"]

    recent_runs = repository.fetch_recent_runs(limit=5)
    assert recent_runs[0]["id"] == result.run_id
    assert recent_runs[0]["served_backend"] == "news_rss"

    source_breakdown = repository.fetch_source_breakdown(result.run_id)
    assert source_breakdown[0]["display_name"] == "OpenAI"
    assert source_breakdown[0]["tweet_count"] == 1
