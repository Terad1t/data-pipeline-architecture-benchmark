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


def save_experiment_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
