from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pibic_sentiment.config import DatasetConfig, ModelConfig, RunConfig
from pibic_sentiment.pipeline import prepare_frame, run_batch_baseline
from pibic_sentiment.runner import run_experiment_grid


class BaselineTests(unittest.TestCase):
    def test_prepare_frame_normalizes_text(self):
        frame = pd.DataFrame(
            {
                "text": ["Hello, WORLD!", "<br />Test   text  ", "   "],
                "label": [1, 0, 1],
            }
        )

        cleaned = prepare_frame(frame)

        self.assertListEqual(cleaned["text"].tolist(), ["hello world", "test text"])

    def test_run_batch_baseline_with_mocked_dataset(self):
        fake_frame = pd.DataFrame(
            {
                "text": [
                    "I loved this movie",
                    "Terrible film",
                    "Amazing acting",
                    "Worst plot ever",
                    "Pretty good overall",
                    "Not my taste",
                    "Fantastic soundtrack",
                    "I disliked the ending",
                ],
                "label": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "metrics"
            artifacts_dir = Path(temp_dir) / "runs"
            config = RunConfig(
                dataset=DatasetConfig(name="imdb", test_size=0.25, random_state=42),
                model=ModelConfig(name="logreg", random_state=42),
                output_dir=output_dir,
                artifacts_dir=artifacts_dir,
            )

            with patch("pibic_sentiment.pipeline.load_sentiment_dataset", return_value=fake_frame):
                outcome = run_batch_baseline(config)

            self.assertTrue(Path(outcome["metrics_path"]).exists())
            self.assertTrue(Path(outcome["model_path"]).exists())
            self.assertTrue(Path(outcome["manifest_path"]).exists())
            self.assertTrue(Path(outcome["log_path"]).exists())
            self.assertGreater(outcome["result"].accuracy, 0.0)

    def test_run_experiment_grid_generates_summary(self):
        fake_frame = pd.DataFrame(
            {
                "text": [
                    "good movie",
                    "bad movie",
                    "excellent plot",
                    "awful acting",
                    "nice pacing",
                    "boring scenes",
                    "great cast",
                    "poor script",
                ],
                "label": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "metrics"
            artifacts_dir = Path(temp_dir) / "runs"

            with patch("pibic_sentiment.pipeline.load_sentiment_dataset", return_value=fake_frame):
                frame = run_experiment_grid(
                    dataset_name="imdb",
                    output_dir=output_dir,
                    artifacts_dir=artifacts_dir,
                    seeds=[42],
                    model_names=["logreg", "linear_svm"],
                )

            self.assertEqual(len(frame), 2)
            self.assertTrue((output_dir / "benchmark_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
