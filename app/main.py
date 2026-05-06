"""CLI entry point for the Twitter/X sentiment tracking system."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.config import AppConfig, load_config
from app.logger import configure_logging, get_logger
from app.pipeline import SentimentTrackingPipeline
from app.schemas import SearchParameters
from ingestion.scraper import IngestionError, TwitterScraper
from processing.cleaner import TextCleaner
from processing.tokenizer import SpacyTokenizer
from sentiment.model_selector import ModelSelector
from storage.database import DatabaseManager
from storage.repository import TweetRepository


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Twitter/X sentiment tracking pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the ingestion and sentiment pipeline.")
    run_parser.add_argument("--query", help="Keyword query to search for.")
    run_parser.add_argument(
        "--hashtags",
        help="Comma-separated hashtags to include in the search.",
    )
    run_parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format.")
    run_parser.add_argument("--end-date", help="End date in YYYY-MM-DD format.")
    run_parser.add_argument("--max-results", type=int, help="Maximum number of tweets to collect.")
    run_parser.add_argument(
        "--backend",
        choices=["auto", "snscrape", "twscrape", "x_api", "news_rss"],
        help="Override the configured ingestion backend.",
    )
    run_parser.add_argument(
        "--model",
        choices=["vader", "ml"],
        help="Override the configured sentiment model.",
    )

    train_parser = subparsers.add_parser("train-ml", help="Train the scikit-learn sentiment model.")
    train_parser.add_argument(
        "--training-data",
        help="Path to a CSV file with `text` and `label` columns.",
    )

    return parser.parse_args()


def build_search_parameters(config: AppConfig, args: argparse.Namespace) -> SearchParameters:
    """Merge CLI overrides with the configured default search parameters."""
    hashtags = (
        [value.strip() for value in args.hashtags.split(",")]
        if getattr(args, "hashtags", None)
        else config.search.hashtags
    )
    start_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d").date()
        if getattr(args, "start_date", None)
        else config.search.start_date
    )
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d").date()
        if getattr(args, "end_date", None)
        else config.search.end_date
    )
    return SearchParameters(
        keyword=getattr(args, "query", None) or config.search.keyword,
        hashtags=hashtags,
        start_date=start_date,
        end_date=end_date,
        max_results=getattr(args, "max_results", None) or config.search.max_results,
    )


def build_pipeline(config: AppConfig) -> SentimentTrackingPipeline:
    """Construct the fully-wired pipeline."""
    database = DatabaseManager(config.database.path)
    database.initialize()
    repository = TweetRepository(database)
    logger = get_logger("twitter_sentiment.pipeline")
    return SentimentTrackingPipeline(
        config=config,
        repository=repository,
        scraper=TwitterScraper(
            backend=config.ingestion.backend,
            max_retries=config.ingestion.max_retries,
            backoff_factor=config.ingestion.backoff_factor,
            twscrape_accounts_db=config.ingestion.twscrape_accounts_db,
            logger=get_logger("twitter_sentiment.ingestion"),
        ),
        cleaner=TextCleaner(),
        tokenizer=SpacyTokenizer(
            model_name=config.processing.spacy_model,
            logger=get_logger("twitter_sentiment.processing"),
        ),
        model_selector=ModelSelector(config=config, logger=get_logger("twitter_sentiment.sentiment")),
        logger=logger,
    )


def main() -> int:
    """Run the requested CLI command."""
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.logging.level, config.logging.format)
    logger = get_logger("twitter_sentiment.cli")
    pipeline = build_pipeline(config)

    if args.command == "train-ml":
        training_data_path = Path(args.training_data).resolve() if args.training_data else None
        model_path = pipeline.train_ml_model(training_data_path)
        logger.info("ML model saved to %s", model_path)
        return 0

    search_parameters = build_search_parameters(config, args)
    try:
        result = pipeline.run(
            search_parameters=search_parameters,
            model_name=getattr(args, "model", None),
            backend_override=getattr(args, "backend", None),
        )
    except IngestionError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Ingestion run completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
