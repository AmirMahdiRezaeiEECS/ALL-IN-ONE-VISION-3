"""
Thin MLflow integration for AIO VISION 3.

We use MLflow directly rather than building a custom experiment-tracking
abstraction. This module only adds the small amount of glue needed to:
  - set the experiment name from config
  - log the full resolved config as params + as a YAML artifact
  - provide a single context manager scripts can wrap their work in
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow
import yaml
from omegaconf import DictConfig

from aio_vision.config import to_container


def _flatten(d: dict, parent_key: str = "") -> dict:
    items: dict = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


@contextmanager
def experiment_run(cfg: DictConfig, run_name: str | None = None) -> Iterator[None]:
    """
    Start an MLflow run configured from `cfg.tracking`.

    Expects config to optionally define:
        tracking:
          experiment_name: "aio-vision-3"
          tracking_uri: "mlruns"        # local dir or remote server URI
    """
    tracking_cfg = cfg.get("tracking", {})
    tracking_uri = tracking_cfg.get("tracking_uri", "mlruns")
    experiment_name = tracking_cfg.get("experiment_name", "aio-vision-3")

    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(experiment_name)

    run_name = run_name or cfg.get("name", None)

    with mlflow.start_run(run_name=run_name):
        config_dict = to_container(cfg)
        mlflow.log_params(_flatten(config_dict))

        # Save the exact resolved config alongside the run for reproducibility.
        tmp_path = Path("resolved_config.yaml")
        tmp_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))
        mlflow.log_artifact(str(tmp_path))
        tmp_path.unlink(missing_ok=True)

        yield
