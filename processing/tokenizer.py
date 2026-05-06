"""spaCy-based tokenization utilities."""

from __future__ import annotations

import logging

import spacy


class SpacyTokenizer:
    """Thin wrapper around spaCy tokenization."""

    def __init__(self, model_name: str = "blank_en", logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.nlp = self._load_model(model_name)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text using spaCy."""
        if not text:
            return []
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]

    def _load_model(self, model_name: str):
        if model_name == "blank_en":
            return spacy.blank("en")
        try:
            return spacy.load(model_name)
        except OSError:
            self.logger.warning(
                "spaCy model `%s` is unavailable. Falling back to blank English tokenizer.",
                model_name,
            )
            return spacy.blank("en")
