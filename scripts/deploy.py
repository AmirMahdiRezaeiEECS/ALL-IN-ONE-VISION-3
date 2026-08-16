#!/usr/bin/env python3
"""
Serve a trained YOLO model behind a minimal HTTP API for inference.

This is a thin wrapper around Ultralytics + FastAPI/uvicorn (mature,
established tools) — not a custom serving framework. For heavier
production needs, export the model (scripts/export.py) and serve it
through a dedicated inference server (Triton, TorchServe, ONNX Runtime
Server, etc.) instead of this script.

Usage:
    python scripts/deploy.py --weights runs/train/.../best.pt --port 8000

Requires the "serve" extra: pip install -e ".[serve]"
"""

from __future__ import annotations

import argparse
import io

from PIL import Image
from ultralytics import YOLO


def build_app(weights: str, conf: float, device: str):
    from fastapi import FastAPI, File, UploadFile

    app = FastAPI(title="AIO VISION 3 — Inference API")
    model = YOLO(weights)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(source=image, conf=conf, device=device, verbose=False)
        boxes = results[0].boxes
        return {
            "detections": [
                {
                    "class_id": int(cls_),
                    "class_name": model.names[int(cls_)],
                    "confidence": float(conf_),
                    "box_xyxy": box.tolist(),
                }
                for box, conf_, cls_ in zip(boxes.xyxy, boxes.conf, boxes.cls)
            ]
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a YOLO model over HTTP.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = build_app(args.weights, args.conf, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
