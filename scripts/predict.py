#!/usr/bin/env python3
"""
Run inference with a trained YOLO model over an image, directory, or video.

Usage:
    python scripts/predict.py \
        --weights runs/train/yolo11n_coco8_example/weights/best.pt \
        --source path/to/images_or_video \
        --imgsz 640 --conf 0.25
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Image, dir, video, or glob.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--project", type=str, default="runs/predict")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        project=args.project,
    )
    print(f"Predicted on {len(results)} item(s). Results saved under {args.project}/")


if __name__ == "__main__":
    main()
