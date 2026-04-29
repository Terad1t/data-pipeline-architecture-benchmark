from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from .config import RunConfig
from .data import load_sentiment_dataset, split_dataset
from .experiment import build_experiment_manifest, save_experiment_manifest
from .evaluation import evaluate_predictions, save_metrics
from .modeling import build_batch_pipeline
from .preprocessing import normalize_text


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["text"] = cleaned["text"].astype(str).map(normalize_text)
    cleaned = cleaned[cleaned["text"].str.len() > 0].reset_index(drop=True)
    return cleaned


def run_batch_baseline(config: RunConfig) -> dict:
    frame = load_sentiment_dataset(config.dataset)
    frame = prepare_frame(frame)
    split = split_dataset(frame, config.dataset)

    pipeline = build_batch_pipeline(config.features, config.model)

    started = perf_counter()
    pipeline.fit(split.train[config.dataset.text_column], split.train[config.dataset.label_column])
    y_pred = pipeline.predict(split.test[config.dataset.text_column])
    elapsed = perf_counter() - started

    result = evaluate_predictions(
        split.test[config.dataset.label_column],
        y_pred,
        elapsed_seconds=elapsed,
        sample_count=len(split.test),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = config.output_dir / "batch_baseline_metrics.csv"
    model_path = config.artifacts_dir / "batch_logreg_pipeline.joblib"
    manifest_path = config.artifacts_dir / "batch_baseline_manifest.json"
    manifest = build_experiment_manifest(config)
    save_metrics(result, metrics_path)
    joblib.dump(pipeline, model_path)
    save_experiment_manifest(manifest, manifest_path)

    return {
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "result": result,
        "train_size": len(split.train),
        "test_size": len(split.test),
    }
