"""Dynamic sentiment model selection based on configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import AppConfig
from sentiment.ml_model import MLSentimentModel
from sentiment.vader_model import VaderSentimentModel


class ModelSelector:
    """Resolve and cache sentiment models at runtime."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._cache: dict[str, object] = {}

    def get_model(self, model_name: str):
        """Return a configured model instance."""
        normalized_name = model_name.lower()
        if normalized_name in self._cache:
            return self._cache[normalized_name]

        if normalized_name == "vader":
            self.logger.info("Using VADER sentiment model.")
            model = VaderSentimentModel(logger=self.logger)
        elif normalized_name == "ml":
            self.logger.info("Using scikit-learn sentiment model.")
            model = self._load_or_train_ml_model()
        else:
            raise ValueError(f"Unsupported sentiment model: {model_name}")

        self._cache[normalized_name] = model
        return model

    def train_ml_model(
        self,
        model_path: Path,
        training_data_path: Path | None = None,
        training_frame: Any | None = None,
    ) -> tuple[Path, MLSentimentModel]:
        """Train and persist a machine-learning sentiment model."""
        model = MLSentimentModel(logger=self.logger)
        if training_frame is not None:
            model.train_from_frame(training_frame)
        elif training_data_path is not None:
            model.train_from_csv(training_data_path)
        else:
            raise ValueError("Either training_data_path or training_frame must be provided.")
        saved_path = model.save(model_path)
        self._cache["ml"] = model
        return saved_path, model

    def _load_or_train_ml_model(self) -> MLSentimentModel:
        model_path = self.config.model.ml_model_path
        if model_path.exists():
            return MLSentimentModel.load(model_path, logger=self.logger)

        if not self.config.model.auto_train_if_missing:
            raise FileNotFoundError(
                f"Configured ML model file does not exist and auto-training is disabled: {model_path}"
            )

        training_data_path = self.config.model.training_data_path
        if not training_data_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found for auto-training: {training_data_path}"
            )

        self.logger.info("Training ML sentiment model from %s", training_data_path)
        self.train_ml_model(training_data_path=training_data_path, model_path=model_path)
        return MLSentimentModel.load(model_path, logger=self.logger)
