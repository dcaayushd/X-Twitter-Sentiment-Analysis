"""Tests for the text processing layer."""

from processing.cleaner import TextCleaner
from processing.tokenizer import SpacyTokenizer


def test_clean_text_removes_noise_and_normalizes_whitespace() -> None:
    cleaner = TextCleaner()

    cleaned = cleaner.clean_text(
        "Hello, @alice! Visit https://example.com #AI 😀 for MORE   updates..."
    )

    assert cleaned == "hello visit for more updates"


def test_spacy_tokenizer_splits_normalized_text() -> None:
    tokenizer = SpacyTokenizer()

    tokens = tokenizer.tokenize("hello data world")

    assert tokens == ["hello", "data", "world"]
