"""Tests for the sentiment analysis layer."""

import pandas as pd

from sentiment.ml_model import MLSentimentModel
from sentiment.vader_model import VaderSentimentModel


def test_vader_model_classifies_obvious_positive_text() -> None:
    model = VaderSentimentModel()

    prediction = model.predict("I absolutely love this product. It is amazing and delightful.")

    assert prediction.label == "positive"
    assert prediction.score > 0
    assert 0.0 <= prediction.confidence <= 1.0


def test_ml_model_can_train_save_load_and_predict(tmp_path) -> None:
    model = MLSentimentModel()
    texts = [
        "I love this",
        "This is fantastic",
        "I hate this",
        "This is awful",
        "It is okay",
        "Nothing special here",
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    model.train(texts, labels)

    model_path = tmp_path / "sentiment_model.joblib"
    model.save(model_path)
    loaded_model = MLSentimentModel.load(model_path)
    prediction = loaded_model.predict("This is fantastic and wonderful")

    assert prediction.label == "positive"
    assert prediction.score > 0
    assert 0.0 <= prediction.confidence <= 1.0


def test_ml_model_load_training_frame_accepts_processed_export_columns(tmp_path) -> None:
    dataset_path = tmp_path / "processed_export.csv"
    pd.DataFrame(
        [
            {"cleaned_text": "love this product", "sentiment_label": "positive"},
            {"cleaned_text": "hate this product", "sentiment_label": "negative"},
            {"cleaned_text": "it is fine", "sentiment_label": "neutral"},
        ]
    ).to_csv(dataset_path, index=False)

    frame = MLSentimentModel.load_training_frame(dataset_path)

    assert list(frame.columns) == ["text", "label"]
    assert frame.to_dict(orient="records") == [
        {"text": "love this product", "label": "positive"},
        {"text": "hate this product", "label": "negative"},
        {"text": "it is fine", "label": "neutral"},
    ]
