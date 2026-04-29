from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_imdb_dataset() -> pd.DataFrame:
    dataset = load_dataset("imdb")
    train_frame = dataset["train"].to_pandas()
    test_frame = dataset["test"].to_pandas()
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    frame = frame.rename(columns={"label": "label", "text": "text"})
    frame = frame[["text", "label"]].dropna().reset_index(drop=True)
    return frame


def load_sentiment_dataset(config: DatasetConfig) -> pd.DataFrame:
    if config.name.lower() != "imdb":
        msg = f"Unsupported dataset: {config.name}. Only imdb is implemented for the first baseline."
        raise ValueError(msg)
    return load_imdb_dataset()


def split_dataset(frame: pd.DataFrame, config: DatasetConfig) -> DatasetSplit:
    train_frame, test_frame = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame[config.label_column],
    )
    return DatasetSplit(
        train=train_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )
