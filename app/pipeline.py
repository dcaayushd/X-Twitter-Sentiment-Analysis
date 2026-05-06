"""Pipeline orchestration for ingestion, processing, sentiment, and persistence."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.config import AppConfig
from app.schemas import AnalyzedTweet, PipelineRunResult, SearchParameters, SentimentPrediction
from ingestion.scraper import IngestionError, TwitterScraper
from processing.cleaner import TextCleaner
from processing.tokenizer import SpacyTokenizer
from sentiment.ml_model import MLSentimentModel
from sentiment.model_selector import ModelSelector
from storage.repository import TweetRepository


class SentimentTrackingPipeline:
    """Coordinates the end-to-end sentiment tracking workflow."""

    def __init__(
        self,
        config: AppConfig,
        repository: TweetRepository,
        scraper: TwitterScraper,
        cleaner: TextCleaner,
        tokenizer: SpacyTokenizer,
        model_selector: ModelSelector,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.repository = repository
        self.scraper = scraper
        self.cleaner = cleaner
        self.tokenizer = tokenizer
        self.model_selector = model_selector
        self.logger = logger

    def run(
        self,
        search_parameters: SearchParameters,
        model_name: str | None = None,
    ) -> PipelineRunResult:
        """Execute the full pipeline and persist the analyzed tweets."""
        search_parameters.validate()
        selected_model_name = (model_name or self.config.model.active_model).lower()
        model = self.model_selector.get_model(selected_model_name)
        run_id = self.repository.create_ingestion_run(
            source_query=search_parameters.source_query,
            model_name=selected_model_name,
        )

        try:
            self.logger.info("Starting ingestion for %s", search_parameters.source_query)
            raw_tweets = self.scraper.fetch_tweets(search_parameters)
            self.logger.info("Collected %s tweets", len(raw_tweets))

            analyzed_tweets = [self._analyze_tweet(tweet, model.predict) for tweet in raw_tweets]
            stored_count = self.repository.upsert_tweets(analyzed_tweets)
            self.repository.link_tweets_to_run(
                run_id=run_id,
                tweet_ids=[tweet.raw_tweet.tweet_id for tweet in analyzed_tweets],
            )
            self.repository.complete_ingestion_run(
                run_id=run_id,
                total_collected=len(raw_tweets),
                total_stored=stored_count,
            )

            raw_export_path = self._export_raw_tweets(raw_tweets, search_parameters.source_query)
            processed_export_path = self._export_processed_tweets(
                analyzed_tweets,
                search_parameters.source_query,
            )
            self.logger.info(
                "Pipeline completed for run_id=%s with model=%s",
                run_id,
                selected_model_name,
            )
            return PipelineRunResult(
                run_id=run_id,
                source_query=search_parameters.source_query,
                model_name=selected_model_name,
                collected_count=len(raw_tweets),
                stored_count=stored_count,
                raw_export_path=str(raw_export_path) if raw_export_path else None,
                processed_export_path=str(processed_export_path) if processed_export_path else None,
            )
        except Exception as exc:
            self.repository.fail_ingestion_run(run_id, error_message=str(exc))
            if isinstance(exc, IngestionError):
                self.logger.error("Pipeline failed for run_id=%s: %s", run_id, exc)
            else:
                self.logger.exception("Pipeline failed for run_id=%s", run_id)
            raise

    def train_ml_model(self, training_data_path: Path | None = None) -> Path:
        """Train and persist the configured ML sentiment model."""
        model_path = self.config.model.ml_model_path
        dataset_path = resolve_training_dataset_path(
            explicit_path=training_data_path,
            configured_path=self.config.model.training_data_path,
            processed_output_dir=self.config.paths.processed_output_dir,
        )
        training_frame = MLSentimentModel.load_training_frame(dataset_path)
        label_counts = training_frame["label"].value_counts().to_dict()
        self.model_selector.train_ml_model(dataset_path, model_path)
        self.logger.info(
            "Trained ML model from %s (%s rows, labels=%s) and saved it to %s",
            dataset_path,
            len(training_frame),
            label_counts,
            model_path,
        )
        return model_path

    def _analyze_tweet(self, tweet, predict_fn) -> AnalyzedTweet:
        cleaned_text = self.cleaner.clean_text(tweet.content)
        tokens = self.tokenizer.tokenize(cleaned_text)
        sentiment = (
            predict_fn(cleaned_text)
            if cleaned_text
            else SentimentPrediction(
                label="neutral",
                score=0.0,
                confidence=1.0,
                metadata={"reason": "empty_cleaned_text"},
            )
        )
        return AnalyzedTweet(
            raw_tweet=tweet,
            cleaned_text=cleaned_text,
            tokens=tokens,
            sentiment=sentiment,
        )

    def _export_raw_tweets(self, raw_tweets: Iterable, source_query: str) -> Path | None:
        rows = [tweet.raw_payload for tweet in raw_tweets]
        if not rows:
            return None
        target = self._build_export_path(self.config.paths.raw_output_dir, source_query)
        pd.DataFrame(rows).to_csv(target, index=False)
        return target

    def _export_processed_tweets(
        self,
        analyzed_tweets: Iterable[AnalyzedTweet],
        source_query: str,
    ) -> Path | None:
        rows: list[dict[str, object]] = []
        for tweet in analyzed_tweets:
            rows.append(
                {
                    "tweet_id": tweet.raw_tweet.tweet_id,
                    "created_at": tweet.raw_tweet.created_at.astimezone(timezone.utc).isoformat(),
                    "username": tweet.raw_tweet.username,
                    "display_name": tweet.raw_tweet.display_name,
                    "content": tweet.raw_tweet.content,
                    "cleaned_text": tweet.cleaned_text,
                    "tokens": json.dumps(tweet.tokens),
                    "sentiment_label": tweet.sentiment.label,
                    "sentiment_score": tweet.sentiment.score,
                    "sentiment_confidence": tweet.sentiment.confidence,
                    "sentiment_metadata": json.dumps(tweet.sentiment.metadata),
                }
            )
        if not rows:
            return None
        target = self._build_export_path(self.config.paths.processed_output_dir, source_query)
        pd.DataFrame(rows).to_csv(target, index=False)
        return target

    def _build_export_path(self, directory: Path, source_query: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", source_query).strip("_").lower()
        filename = f"{slug[:80] or 'tweets'}_{timestamp}.csv"
        return directory / filename


def resolve_training_dataset_path(
    explicit_path: Path | None,
    configured_path: Path,
    processed_output_dir: Path,
) -> Path:
    """Pick a training dataset, preferring the newest trainable processed export."""
    if explicit_path is not None:
        return explicit_path.resolve()

    candidates = sorted(
        processed_output_dir.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            training_frame = MLSentimentModel.load_training_frame(candidate)
        except Exception:
            continue
        if training_frame["label"].nunique() < 2:
            continue
        return candidate.resolve()

    return configured_path.resolve()
