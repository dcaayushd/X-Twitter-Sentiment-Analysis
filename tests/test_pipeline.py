"""Tests for pipeline dataset selection helpers."""

from __future__ import annotations

import pandas as pd

from app.pipeline import build_training_corpus, resolve_training_dataset_paths


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
