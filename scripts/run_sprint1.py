from __future__ import annotations

from pibic_sentiment.runner import run_experiment_grid, save_grid_manifest


def main() -> None:
    dataset_name = "imdb"
    seeds = [42, 1337, 2026]
    model_names = ["logreg", "linear_svm"]

    save_grid_manifest(
        dataset_name=dataset_name,
        seeds=seeds,
        model_names=model_names,
        output_path="experiments/run_grid_manifest.json",
    )
    frame = run_experiment_grid(dataset_name=dataset_name, seeds=seeds, model_names=model_names)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
