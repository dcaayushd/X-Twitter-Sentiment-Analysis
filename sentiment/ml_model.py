"""Machine-learning sentiment model using scikit-learn."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from app.schemas import SentimentPrediction


MODEL_ARTIFACT_VERSION = 2
CANONICAL_LABELS = {"positive", "negative", "neutral"}
LABEL_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
}


class MLSentimentModel:
    """Trainable and serializable scikit-learn sentiment classifier."""

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        logger: logging.Logger | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = pipeline or self._build_candidate_pipelines()["word_char_logreg"]
        self.metadata = metadata or {}

    @staticmethod
    def _build_candidate_pipelines() -> dict[str, Pipeline]:
        """Construct candidate pipelines used during model selection."""
        return {
            "word_bigram_logreg": Pipeline(
                steps=[
                    (
                        "features",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            min_df=1,
                            max_df=0.95,
                            strip_accents="unicode",
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            C=2.0,
                        ),
                    ),
                ]
            ),
            "word_char_logreg": Pipeline(
                steps=[
                    (
                        "features",
                        FeatureUnion(
                            transformer_list=[
                                (
                                    "word",
                                    TfidfVectorizer(
                                        ngram_range=(1, 2),
                                        min_df=1,
                                        max_df=0.95,
                                        strip_accents="unicode",
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=(3, 5),
                                        min_df=1,
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2500,
                            class_weight="balanced",
                            C=4.0,
                        ),
                    ),
                ]
            ),
        }

    def train(
        self,
        texts: list[str],
        labels: list[str],
        sample_weights: list[float] | np.ndarray | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fit the classifier using labeled text examples and candidate selection."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if len(set(labels)) < 2:
            raise ValueError("At least two distinct labels are required for training.")

        texts_array = np.asarray(texts, dtype=object)
        labels_array = np.asarray(labels, dtype=object)
        if sample_weights is None:
            weights_array = np.ones(len(texts_array), dtype=float)
        else:
            weights_array = np.asarray(sample_weights, dtype=float)
        if len(weights_array) != len(texts_array):
            raise ValueError("sample_weights must have the same length as texts.")

        candidate_pipelines = self._build_candidate_pipelines()
        candidate_metrics = {
            name: self._evaluate_candidate(name, pipeline, texts_array, labels_array, weights_array)
            for name, pipeline in candidate_pipelines.items()
        }
        selected_name = self._select_best_candidate(candidate_metrics)
        selected_pipeline = clone(candidate_pipelines[selected_name])
        selected_pipeline.fit(
            texts_array.tolist(),
            labels_array.tolist(),
            classifier__sample_weight=weights_array,
        )

        self.pipeline = selected_pipeline
        self.metadata = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "selected_model": selected_name,
            "training_examples": int(len(texts_array)),
            "label_distribution": pd.Series(labels_array).value_counts().to_dict(),
            "weight_summary": {
                "min": float(weights_array.min()),
                "max": float(weights_array.max()),
                "mean": float(weights_array.mean()),
            },
            "candidate_metrics": candidate_metrics,
        }
        if training_metadata:
            self.metadata.update(training_metadata)

    @staticmethod
    def load_training_frame(dataset_path: str | Path) -> pd.DataFrame:
        """Load and normalize a supported training CSV into a weighted frame."""
        dataset_path = Path(dataset_path).resolve()
        dataset = pd.read_csv(dataset_path)

        if {"text", "label"}.issubset(dataset.columns):
            text_column = "text"
            label_column = "label"
            source_kind = "manual_labels"
        elif {"cleaned_text", "sentiment_label"}.issubset(dataset.columns):
            text_column = "cleaned_text"
            label_column = "sentiment_label"
            source_kind = "processed_export"
        elif {"content", "sentiment_label"}.issubset(dataset.columns):
            text_column = "content"
            label_column = "sentiment_label"
            source_kind = "processed_export"
        else:
            raise ValueError(
                "Training dataset must include either `text`/`label`, "
                "`cleaned_text`/`sentiment_label`, or `content`/`sentiment_label` columns."
            )

        frame = (
            dataset[[column for column in [text_column, label_column, "sentiment_confidence", "sentiment_score"] if column in dataset.columns]]
            .rename(columns={text_column: "text", label_column: "label"})
            .dropna(subset=["text", "label"])
            .copy()
        )
        frame["text"] = frame["text"].astype(str).str.strip()
        frame["label"] = (
            frame["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(LABEL_ALIASES)
        )
        frame = frame[
            (frame["text"] != "") & frame["label"].isin(CANONICAL_LABELS)
        ].copy()
        if frame.empty:
            raise ValueError(f"Training dataset has no usable labeled rows: {dataset_path}")

        if source_kind == "manual_labels":
            frame["sample_weight"] = 3.0
        else:
            confidence = pd.to_numeric(
                frame.get("sentiment_confidence", pd.Series([0.5] * len(frame))),
                errors="coerce",
            ).fillna(0.5).clip(0.0, 1.0)
            score = pd.to_numeric(
                frame.get("sentiment_score", pd.Series([0.0] * len(frame))),
                errors="coerce",
            ).fillna(0.0).abs().clip(0.0, 1.0)
            frame["sample_weight"] = 0.4 + (0.9 * confidence)
            frame.loc[frame["label"] != "neutral", "sample_weight"] *= 1.1
            frame.loc[frame["label"] == "neutral", "sample_weight"] *= 0.85
            frame["sample_weight"] *= 0.75 + (0.5 * score)

        frame["sample_weight"] = frame["sample_weight"].clip(0.25, 3.0)
        frame["source_path"] = str(dataset_path)
        frame["source_kind"] = source_kind
        return frame[["text", "label", "sample_weight", "source_path", "source_kind"]]

    @staticmethod
    def merge_training_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple weighted training frames into a deduplicated corpus."""
        if not frames:
            raise ValueError("At least one training frame is required.")

        combined = pd.concat(frames, ignore_index=True)
        combined["normalized_text"] = (
            combined["text"].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        )
        combined = combined[combined["normalized_text"] != ""].copy()
        if combined.empty:
            raise ValueError("Merged training corpus is empty.")

        representative_text = combined.groupby("normalized_text")["text"].agg(
            lambda values: max(values, key=len)
        )
        grouped = (
            combined.groupby(["normalized_text", "label"], as_index=False)
            .agg(
                sample_weight=("sample_weight", "sum"),
                support=("label", "size"),
                source_paths=("source_path", lambda values: sorted(set(values))),
                source_kinds=("source_kind", lambda values: sorted(set(values))),
            )
            .sort_values(
                ["normalized_text", "sample_weight", "support"],
                ascending=[True, False, False],
            )
        )
        selected = grouped.drop_duplicates(subset=["normalized_text"], keep="first").copy()
        selected["text"] = selected["normalized_text"].map(representative_text)
        selected["source_paths"] = selected["source_paths"].apply(list)
        selected["source_kinds"] = selected["source_kinds"].apply(list)
        return selected[
            ["text", "label", "sample_weight", "support", "source_paths", "source_kinds"]
        ].reset_index(drop=True)

    def train_from_frame(self, training_frame: pd.DataFrame) -> None:
        """Train the classifier using a merged weighted training frame."""
        required_columns = {"text", "label", "sample_weight"}
        if not required_columns.issubset(training_frame.columns):
            raise ValueError(
                f"Training frame must include columns: {sorted(required_columns)}"
            )

        cleaned_dataset = training_frame.dropna(subset=["text", "label", "sample_weight"]).copy()
        cleaned_dataset["text"] = cleaned_dataset["text"].astype(str)
        cleaned_dataset["label"] = cleaned_dataset["label"].astype(str)
        cleaned_dataset["sample_weight"] = pd.to_numeric(
            cleaned_dataset["sample_weight"], errors="coerce"
        ).fillna(1.0)
        texts = cleaned_dataset["text"].tolist()
        labels = cleaned_dataset["label"].tolist()
        sample_weights = cleaned_dataset["sample_weight"].tolist()
        metadata = {
            "source_paths": sorted(
                {
                    str(path)
                    for values in cleaned_dataset.get("source_paths", [])
                    for path in (values if isinstance(values, list) else [values])
                    if path
                }
            ),
            "source_kinds": sorted(
                {
                    str(kind)
                    for values in cleaned_dataset.get("source_kinds", [])
                    for kind in (values if isinstance(values, list) else [values])
                    if kind
                }
            ),
            "deduplicated_examples": int(len(cleaned_dataset)),
            "support_sum": int(pd.to_numeric(cleaned_dataset.get("support", 1), errors="coerce").fillna(1).sum()),
        }
        self.train(texts, labels, sample_weights=sample_weights, training_metadata=metadata)

    def train_from_csv(self, dataset_path: str | Path) -> None:
        """Train the classifier using a normalized labeled CSV dataset."""
        training_frame = self.load_training_frame(dataset_path)
        merged_frame = self.merge_training_frames([training_frame])
        self.train_from_frame(merged_frame)

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

        metadata = {
            "probabilities": {
                str(label): float(probability) for label, probability in zip(labels, probabilities)
            }
        }
        if self.metadata:
            metadata["model"] = {
                key: self.metadata[key]
                for key in ["selected_model", "trained_at", "training_examples"]
                if key in self.metadata
            }

        return SentimentPrediction(
            label=predicted_label,
            score=max(min(signed_score, 1.0), -1.0),
            confidence=confidence,
            metadata=metadata,
        )

    def save(self, model_path: str | Path) -> Path:
        """Persist the fitted pipeline and metadata to disk."""
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "pipeline": self.pipeline,
            "metadata": self.metadata,
        }
        joblib.dump(artifact, target)

        metadata_path = target.with_name(f"{target.stem}.metadata.json")
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, model_path: str | Path, logger: logging.Logger | None = None) -> "MLSentimentModel":
        """Load a previously trained pipeline from disk."""
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return cls(
                pipeline=artifact["pipeline"],
                logger=logger,
                metadata=dict(artifact.get("metadata", {})),
            )
        return cls(pipeline=artifact, logger=logger, metadata={})

    def _evaluate_candidate(
        self,
        candidate_name: str,
        pipeline: Pipeline,
        texts: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate a candidate pipeline with stratified cross-validation."""
        label_counts = pd.Series(labels).value_counts()
        min_class_count = int(label_counts.min())
        if len(texts) < 6 or min_class_count < 2:
            return {
                "candidate": candidate_name,
                "folds": 0,
                "macro_f1": None,
                "accuracy": None,
            }

        folds = min(5, min_class_count)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        macro_f1_scores: list[float] = []
        accuracy_scores: list[float] = []

        for train_indices, test_indices in splitter.split(texts, labels):
            model = clone(pipeline)
            model.fit(
                texts[train_indices].tolist(),
                labels[train_indices].tolist(),
                classifier__sample_weight=sample_weights[train_indices],
            )
            predictions = model.predict(texts[test_indices].tolist())
            macro_f1_scores.append(
                float(f1_score(labels[test_indices], predictions, average="macro"))
            )
            accuracy_scores.append(float(accuracy_score(labels[test_indices], predictions)))

        return {
            "candidate": candidate_name,
            "folds": folds,
            "macro_f1": float(np.mean(macro_f1_scores)),
            "accuracy": float(np.mean(accuracy_scores)),
        }

    @staticmethod
    def _select_best_candidate(candidate_metrics: dict[str, dict[str, Any]]) -> str:
        """Choose the best candidate by validation macro F1, then accuracy."""

        def metric_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
            metrics = item[1]
            macro_f1 = metrics.get("macro_f1")
            accuracy = metrics.get("accuracy")
            return (
                float(macro_f1) if macro_f1 is not None else float("-inf"),
                float(accuracy) if accuracy is not None else float("-inf"),
            )

        best_name, best_metrics = max(candidate_metrics.items(), key=metric_key)
        if best_metrics.get("macro_f1") is None:
            return "word_char_logreg"
        return best_name
