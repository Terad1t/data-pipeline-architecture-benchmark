from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from .config import RunConfig
from .data import load_sentiment_dataset, split_dataset
from .experiment import append_run_log, build_experiment_manifest, build_run_id, save_experiment_manifest
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
    run_id = build_run_id(
        dataset_name=config.dataset.name,
        model_name=config.model.name,
        seed=config.dataset.random_state,
    )

    started = perf_counter()
    pipeline.fit(split.train[config.dataset.text_column], split.train[config.dataset.label_column])
    y_pred = pipeline.predict(split.test[config.dataset.text_column])
    elapsed = perf_counter() - started

    result = evaluate_predictions(
        split.test[config.dataset.label_column],
        y_pred,
        elapsed_seconds=elapsed,
        sample_count=len(split.test),
        model_name=config.model.name,
        dataset_name=config.dataset.name,
        run_id=run_id,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = config.output_dir / f"{run_id}_metrics.csv"
    model_path = config.artifacts_dir / f"{run_id}_pipeline.joblib"
    manifest_path = config.artifacts_dir / f"{run_id}_manifest.json"
    log_path = config.output_dir / "batch_run_log.jsonl"
    manifest = build_experiment_manifest(config)
    manifest["run_id"] = run_id
    manifest["metrics_path"] = str(metrics_path)
    manifest["model_path"] = str(model_path)
    save_metrics(result, metrics_path)
    joblib.dump(pipeline, model_path)
    save_experiment_manifest(manifest, manifest_path)
    append_run_log(
        {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "metrics_path": str(metrics_path),
            "model_path": str(model_path),
            "dataset": config.dataset.name,
            "model": config.model.name,
            "seed": config.dataset.random_state,
            "result": {
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "latency_seconds": result.latency_seconds,
                "processing_seconds": result.processing_seconds,
                "samples_per_second": result.samples_per_second,
            },
        },
        log_path,
    )

    return {
        "run_id": run_id,
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "result": result,
        "train_size": len(split.train),
        "test_size": len(split.test),
    }
