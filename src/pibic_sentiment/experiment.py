from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig


def build_experiment_manifest(config: RunConfig) -> dict:
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["artifacts_dir"] = str(config.artifacts_dir)
    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config": config_dict,
    }


def build_run_id(*, dataset_name: str, model_name: str, seed: int) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{dataset_name}_{model_name}_seed{seed}"


def save_experiment_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def append_run_log(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False))
        stream.write("\n")
