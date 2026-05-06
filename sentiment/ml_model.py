"""Machine-learning sentiment model using scikit-learn."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.schemas import SentimentPrediction


class MLSentimentModel:
    """Trainable and serializable scikit-learn sentiment classifier."""

    def __init__(self, pipeline: Pipeline | None = None, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = pipeline or Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Fit the classifier using labeled text examples."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if len(set(labels)) < 2:
            raise ValueError("At least two distinct labels are required for training.")
        self.pipeline.fit(texts, labels)

    @staticmethod
    def load_training_frame(dataset_path: str | Path) -> pd.DataFrame:
        """Load and normalize a supported training CSV into `text`/`label` columns."""
        dataset = pd.read_csv(dataset_path)

        if {"text", "label"}.issubset(dataset.columns):
            text_column = "text"
            label_column = "label"
        elif {"cleaned_text", "sentiment_label"}.issubset(dataset.columns):
            text_column = "cleaned_text"
            label_column = "sentiment_label"
        elif {"content", "sentiment_label"}.issubset(dataset.columns):
            text_column = "content"
            label_column = "sentiment_label"
        else:
            raise ValueError(
                "Training dataset must include either `text`/`label`, "
                "`cleaned_text`/`sentiment_label`, or `content`/`sentiment_label` columns."
            )

        cleaned_dataset = (
            dataset[[text_column, label_column]]
            .rename(columns={text_column: "text", label_column: "label"})
            .dropna(subset=["text", "label"])
            .copy()
        )
        cleaned_dataset["text"] = cleaned_dataset["text"].astype(str).str.strip()
        cleaned_dataset["label"] = cleaned_dataset["label"].astype(str).str.strip().str.lower()
        cleaned_dataset = cleaned_dataset[
            (cleaned_dataset["text"] != "") & (cleaned_dataset["label"] != "")
        ]
        return cleaned_dataset

    def train_from_csv(self, dataset_path: str | Path) -> None:
        """Train the classifier using a normalized labeled CSV dataset."""
        cleaned_dataset = self.load_training_frame(dataset_path)
        texts = cleaned_dataset["text"].astype(str).tolist()
        labels = cleaned_dataset["label"].astype(str).tolist()
        self.train(texts, labels)

    def predict(self, text: str) -> SentimentPrediction:
        """Predict the sentiment label and probability-backed confidence."""
        if not text:
            return SentimentPrediction(label="neutral", score=0.0, confidence=1.0, metadata={})

        probabilities = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        predicted_label = str(labels[best_index])
        confidence = float(probabilities[best_index])

        signed_score = 0.0
        for label, probability in zip(labels, probabilities):
            if label == "positive":
                signed_score += float(probability)
            elif label == "negative":
                signed_score -= float(probability)

        return SentimentPrediction(
            label=predicted_label,
            score=max(min(signed_score, 1.0), -1.0),
            confidence=confidence,
            metadata={
                "probabilities": {
                    str(label): float(probability) for label, probability in zip(labels, probabilities)
                }
            },
        )

    def save(self, model_path: str | Path) -> Path:
        """Persist the fitted pipeline to disk."""
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target)
        return target

    @classmethod
    def load(cls, model_path: str | Path, logger: logging.Logger | None = None) -> "MLSentimentModel":
        """Load a previously trained pipeline from disk."""
        pipeline = joblib.load(model_path)
        return cls(pipeline=pipeline, logger=logger)
