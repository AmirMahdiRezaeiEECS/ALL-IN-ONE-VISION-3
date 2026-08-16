#!/usr/bin/env python3
"""
Prepare a dataset for Ultralytics YOLO training.

This script does NOT reimplement dataset conversion logic. It is a thin
wrapper that:
  1. Expects raw data to already be organized as (or converted to) YOLO
     format: images/ + labels/ pairs, split into train/val/(test).
  2. Generates the Ultralytics-style data.yaml that other scripts and
     Ultralytics itself expect.

For converting from other formats (COCO JSON, Pascal VOC, etc.), prefer
an existing mature converter (e.g. Ultralytics' own JSON2YOLO utilities,
FiftyOne, or Roboflow) rather than writing custom conversion code here.
Add a conversion step ahead of this script if/when a real dataset needs one.

Usage:
    python scripts/prepare_dataset.py \
        --name my_dataset \
        --root data/my_dataset \
        --classes person,car,bicycle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

EXPECTED_SPLIT_DIRS = ("train", "val")  # "test" is optional


def _check_split(root: Path, split: str) -> tuple[int, int]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    n_images = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
    n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    return n_images, n_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format dataset and write its data.yaml.")
    parser.add_argument("--name", required=True, help="Dataset name.")
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root, expected to contain images/{train,val[,test]} and labels/{train,val[,test]}.",
    )
    parser.add_argument("--classes", required=True, help="Comma-separated class names, in class-index order.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    for split in EXPECTED_SPLIT_DIRS:
        n_images, n_labels = _check_split(root, split)
        if n_images == 0:
            raise FileNotFoundError(
                f"No images found for split '{split}' under {root / 'images' / split}. "
                "Organize raw data as images/<split>/ and labels/<split>/ before running this script."
            )
        print(f"[{split}] {n_images} images, {n_labels} labels")

    data_yaml = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }
    if (root / "images" / "test").exists():
        data_yaml["test"] = "images/test"

    out_path = root / "data.yaml"
    out_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"\nWrote {out_path}")
    print(
        f"Now add configs/datasets/{args.name}.yaml pointing `path:` at {out_path}, "
        "and reference it (as `dataset: " + args.name + "`) from an experiment config."
    )


if __name__ == "__main__":
    main()
