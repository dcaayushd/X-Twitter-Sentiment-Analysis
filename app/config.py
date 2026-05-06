"""Configuration loader for the sentiment tracking system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion-specific settings."""

    backend: str = "x_api"
    max_retries: int = 3
    backoff_factor: float = 1.5
    twscrape_accounts_db: Path = Path("data/twscrape_accounts.db")


@dataclass(frozen=True)
class SearchConfig:
    """Default search settings for CLI runs."""

    keyword: str | None
    hashtags: list[str]
    start_date: date | None
    end_date: date | None
    max_results: int


@dataclass(frozen=True)
class ProcessingConfig:
    """Text processing settings."""

    spacy_model: str = "blank_en"


@dataclass(frozen=True)
class ModelConfig:
    """Sentiment model settings."""

    active_model: str = "vader"
    ml_model_path: Path = Path("models/sentiment_model.joblib")
    training_data_path: Path = Path("data/processed/training_sample.csv")
    auto_train_if_missing: bool = True


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite connection settings."""

    path: Path = Path("data/twitter_sentiment.db")


@dataclass(frozen=True)
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used for pipeline outputs."""

    raw_output_dir: Path = Path("data/raw")
    processed_output_dir: Path = Path("data/processed")


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard settings."""

    title: str = "Twitter/X Sentiment Tracker"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration object."""

    project_root: Path
    ingestion: IngestionConfig
    search: SearchConfig
    processing: ProcessingConfig
    model: ModelConfig
    database: DatabaseConfig
    logging: LoggingConfig
    paths: PathsConfig
    dashboard: DashboardConfig


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise ConfigError(f"Unsupported date value: {value!r}")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    """Load application configuration from a YAML file."""
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    project_root = config_file.parent.parent.resolve()

    search_config = raw_config.get("search", {})
    model_config = raw_config.get("model", {})
    database_config = raw_config.get("database", {})
    logging_config = raw_config.get("logging", {})
    paths_config = raw_config.get("paths", {})
    processing_config = raw_config.get("processing", {})
    ingestion_config = raw_config.get("ingestion", {})
    dashboard_config = raw_config.get("dashboard", {})

    ingestion_backend = str(ingestion_config.get("backend", "x_api")).lower()
    if ingestion_backend not in {"auto", "snscrape", "twscrape", "x_api", "news_rss"}:
        raise ConfigError(
            "ingestion.backend must be one of: auto, x_api, snscrape, twscrape, news_rss."
        )

    return AppConfig(
        project_root=project_root,
        ingestion=IngestionConfig(
            backend=ingestion_backend,
            max_retries=int(ingestion_config.get("max_retries", 3)),
            backoff_factor=float(ingestion_config.get("backoff_factor", 1.5)),
            twscrape_accounts_db=_resolve_path(
                project_root,
                ingestion_config.get("twscrape_accounts_db", "data/twscrape_accounts.db"),
            ),
        ),
        search=SearchConfig(
            keyword=search_config.get("keyword"),
            hashtags=list(search_config.get("hashtags", [])),
            start_date=_parse_date(search_config.get("start_date")),
            end_date=_parse_date(search_config.get("end_date")),
            max_results=int(search_config.get("max_results", 100)),
        ),
        processing=ProcessingConfig(
            spacy_model=str(processing_config.get("spacy_model", "blank_en"))
        ),
        model=ModelConfig(
            active_model=str(model_config.get("active_model", "vader")).lower(),
            ml_model_path=_resolve_path(
                project_root,
                model_config.get("ml_model_path", "models/sentiment_model.joblib"),
            ),
            training_data_path=_resolve_path(
                project_root,
                model_config.get("training_data_path", "data/processed/training_sample.csv"),
            ),
            auto_train_if_missing=bool(model_config.get("auto_train_if_missing", True)),
        ),
        database=DatabaseConfig(
            path=_resolve_path(project_root, database_config.get("path", "data/twitter_sentiment.db"))
        ),
        logging=LoggingConfig(
            level=str(logging_config.get("level", "INFO")).upper(),
            format=str(
                logging_config.get(
                    "format",
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                )
            ),
        ),
        paths=PathsConfig(
            raw_output_dir=_resolve_path(project_root, paths_config.get("raw_output_dir", "data/raw")),
            processed_output_dir=_resolve_path(
                project_root,
                paths_config.get("processed_output_dir", "data/processed"),
            ),
        ),
        dashboard=DashboardConfig(
            title=str(dashboard_config.get("title", "Twitter/X Sentiment Tracker"))
        ),
    )
