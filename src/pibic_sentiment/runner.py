from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DatasetConfig, FeatureConfig, ModelConfig, RunConfig
from .pipeline import run_batch_baseline


def run_experiment_grid(
    *,
    dataset_name: str = "imdb",
    output_dir: str | Path = "metrics",
    artifacts_dir: str | Path = "experiments/runs",
    seeds: list[int] | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    seeds = seeds or [42, 1337, 2026]
    model_names = model_names or ["logreg", "linear_svm"]
    rows = []

    for seed in seeds:
        for model_name in model_names:
            config = RunConfig(
                dataset=DatasetConfig(name=dataset_name, test_size=0.2, random_state=seed),
                features=FeatureConfig(),
                model=ModelConfig(name=model_name, random_state=seed),
                output_dir=Path(output_dir),
                artifacts_dir=Path(artifacts_dir),
            )
            outcome = run_batch_baseline(config)
            row = asdict(outcome["result"])
            row["metrics_path"] = outcome["metrics_path"]
            row["model_path"] = outcome["model_path"]
            row["manifest_path"] = outcome["manifest_path"]
            row["run_id"] = outcome["run_id"]
            row["seed"] = seed
            rows.append(row)

    frame = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "benchmark_summary.csv", index=False)
    return frame


def save_grid_manifest(
    *,
    dataset_name: str,
    seeds: list[int],
    model_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name,
        "seeds": seeds,
        "model_names": model_names,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
