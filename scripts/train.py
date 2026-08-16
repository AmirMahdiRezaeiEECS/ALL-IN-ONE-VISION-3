#!/usr/bin/env python3
"""
Train a YOLO model using Ultralytics, driven entirely by a config file.

Usage:
    python scripts/train.py configs/experiments/yolo11n_coco8_example.yaml

This script is intentionally thin: Ultralytics does the actual training.
Our job is only to load config, wire up MLflow tracking, and hand
parameters to `model.train(...)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

from aio_vision.config import load_experiment_config, to_container  # noqa: E402
from aio_vision.tracking import experiment_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model from a config file.")
    parser.add_argument("config", type=str, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the last checkpoint."
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    dataset_yaml = cfg.dataset.path
    model_weights = cfg.model.weights
    train_params = to_container(cfg.training)
    train_params["resume"] = args.resume

    with experiment_run(cfg, run_name=cfg.get("name")):
        model = YOLO(model_weights)
        results = model.train(data=dataset_yaml, **train_params)

        # Log the best/last checkpoints so the run is self-contained.
        import mlflow

        save_dir = Path(results.save_dir)
        for ckpt_name in ("best.pt", "last.pt"):
            ckpt_path = save_dir / "weights" / ckpt_name
            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path), artifact_path="weights")

        print(f"\nTraining complete. Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
