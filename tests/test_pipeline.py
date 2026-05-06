"""Tests for pipeline dataset selection helpers."""

from __future__ import annotations

import pandas as pd

from app.pipeline import resolve_training_dataset_path


def test_resolve_training_dataset_path_prefers_latest_trainable_processed_export(tmp_path) -> None:
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

    selected = resolve_training_dataset_path(
        explicit_path=None,
        configured_path=configured,
        processed_output_dir=processed_dir,
    )

    assert selected == latest.resolve()


def test_resolve_training_dataset_path_respects_explicit_dataset(tmp_path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    explicit = tmp_path / "custom.csv"
    explicit.write_text("text,label\nhello,positive\nbye,negative\n", encoding="utf-8")

    selected = resolve_training_dataset_path(
        explicit_path=explicit,
        configured_path=processed_dir / "training_sample.csv",
        processed_output_dir=processed_dir,
    )

    assert selected == explicit.resolve()
