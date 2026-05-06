"""Reusable text cleaning utilities."""

from __future__ import annotations

import re
import string


class TextCleaner:
    """Normalize tweet text before sentiment analysis."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "]",
        flags=re.UNICODE,
    )
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self) -> None:
        self._punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text: str) -> str:
        """Return cleaned, normalized tweet text."""
        value = text or ""
        value = self.URL_PATTERN.sub(" ", value)
        value = self.MENTION_PATTERN.sub(" ", value)
        value = self.HASHTAG_PATTERN.sub(" ", value)
        value = self.EMOJI_PATTERN.sub(" ", value)
        value = value.translate(self._punctuation_table)
        value = value.lower().strip()
        value = self.WHITESPACE_PATTERN.sub(" ", value)
        return value
