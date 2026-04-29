from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2


@dataclass(frozen=True)
class ModelConfig:
    max_iter: int = 1_000
    class_weight: str | None = "balanced"
    random_state: int = 42


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: Path = Path("metrics")
    artifacts_dir: Path = Path("experiments/runs")
