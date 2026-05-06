"""Rule-based sentiment analysis using VADER."""

from __future__ import annotations

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.schemas import SentimentPrediction


class VaderSentimentModel:
    """Wrapper around VADER with a consistent prediction interface."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> SentimentPrediction:
        """Predict sentiment label and confidence for a piece of text."""
        if not text:
            return SentimentPrediction(
                label="neutral",
                score=0.0,
                confidence=1.0,
                metadata={"compound": 0.0, "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0}},
            )

        scores = self.analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        if compound >= 0.05:
            label = "positive"
            confidence = float(scores["pos"])
        elif compound <= -0.05:
            label = "negative"
            confidence = float(scores["neg"])
        else:
            label = "neutral"
            confidence = float(scores["neu"])

        return SentimentPrediction(
            label=label,
            score=compound,
            confidence=confidence,
            metadata={"compound": compound, "scores": scores},
        )
