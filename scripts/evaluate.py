#!/usr/bin/env python3
"""
Evaluate a trained YOLO model on its configured validation/test split.

Usage:
    python scripts/evaluate.py configs/experiments/yolo11n_coco8_example.yaml \
        --weights runs/train/yolo11n_coco8_example/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO  # noqa: E402

from aio_vision.config import load_experiment_config  # noqa: E402
from aio_vision.tracking import experiment_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    parser.add_argument("config", type=str, help="Path to the experiment YAML config used for training.")
    parser.add_argument("--weights", type=str, required=True, help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset_yaml = cfg.dataset.path

    with experiment_run(cfg, run_name=f"{cfg.get('name')}-eval"):
        model = YOLO(args.weights)
        metrics = model.val(
            data=dataset_yaml,
            split=args.split,
            imgsz=cfg.training.get("imgsz", 640),
            device=cfg.training.get("device", 0),
        )

        import mlflow

        mlflow.log_metrics(
            {
                "map50-95": float(metrics.box.map),
                "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75),
            }
        )
        print(metrics)


if __name__ == "__main__":
    main()
