from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import DatasetConfig, RunConfig
from .pipeline import run_batch_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PIBIC batch sentiment baseline.")
    parser.add_argument("--output-dir", default="metrics", help="Directory for CSV metrics.")
    parser.add_argument("--artifacts-dir", default="experiments/runs", help="Directory for serialized models.")
    parser.add_argument("--dataset", default="imdb", help="Initial dataset name.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = RunConfig(
        dataset=DatasetConfig(name=args.dataset, test_size=args.test_size, random_state=args.random_state),
        output_dir=Path(args.output_dir),
        artifacts_dir=Path(args.artifacts_dir),
    )

    outcome = run_batch_baseline(config)
    result = outcome["result"]

    print("Batch baseline completed")
    print(f"Dataset: {config.dataset.name}")
    print(f"Train size: {outcome['train_size']}")
    print(f"Test size: {outcome['test_size']}")
    print(f"Metrics file: {outcome['metrics_path']}")
    print(f"Model file: {outcome['model_path']}")
    print("Metrics:")
    for key, value in asdict(result).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
