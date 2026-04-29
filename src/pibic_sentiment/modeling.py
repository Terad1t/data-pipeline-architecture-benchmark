from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from .config import FeatureConfig, ModelConfig


@dataclass(frozen=True)
class BatchArtifacts:
    pipeline: Pipeline


def build_batch_pipeline(feature_config: FeatureConfig, model_config: ModelConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=feature_config.max_features,
        ngram_range=feature_config.ngram_range,
        min_df=feature_config.min_df,
    )
    if model_config.name == "logreg":
        classifier = LogisticRegression(
            max_iter=model_config.max_iter,
            class_weight=model_config.class_weight,
            random_state=model_config.random_state,
        )
    elif model_config.name == "linear_svm":
        classifier = LinearSVC(class_weight=model_config.class_weight, random_state=model_config.random_state)
    else:
        msg = f"Unsupported model: {model_config.name}. Supported: logreg, linear_svm"
        raise ValueError(msg)

    return Pipeline([
        ("tfidf", vectorizer),
        (model_config.name, classifier),
    ])
