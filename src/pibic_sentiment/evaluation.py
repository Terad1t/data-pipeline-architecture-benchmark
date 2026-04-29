from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_seconds: float
    processing_seconds: float
    samples_per_second: float


def evaluate_predictions(y_true, y_pred, *, elapsed_seconds: float, sample_count: int) -> EvaluationResult:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    throughput = sample_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_seconds=elapsed_seconds,
        processing_seconds=elapsed_seconds,
        samples_per_second=throughput,
    )


def result_to_frame(result: EvaluationResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(result)])


def save_metrics(result: EvaluationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_to_frame(result).to_csv(output_path, index=False)
