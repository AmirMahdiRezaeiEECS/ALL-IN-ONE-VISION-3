#!/usr/bin/env python3
"""
Export a trained YOLO checkpoint to a deployment format.

Usage:
    python scripts/export.py --weights runs/train/.../best.pt --format onnx
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model for deployment.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="onnx, torchscript, engine (TensorRT), coreml, tflite, openvino, etc. "
        "See Ultralytics export docs for the full list.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 export.")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (ONNX/TensorRT).")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(
        format=args.format, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic
    )
    print(f"Exported model to: {exported_path}")


if __name__ == "__main__":
    main()
