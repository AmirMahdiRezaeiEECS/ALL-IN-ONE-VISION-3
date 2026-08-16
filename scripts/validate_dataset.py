#!/usr/bin/env python3
"""
Validate a YOLO-format dataset for common integrity issues before training:
  - every image has a matching label file (and vice versa)
  - label files contain valid YOLO-format rows (class_id + 4 normalized floats)
  - class ids are within the declared number of classes
  - no empty splits

Usage:
    python scripts/validate_dataset.py configs/datasets/my_dataset.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _validate_label_file(path: Path, num_classes: int) -> list[str]:
    errors = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (5, 6):  # class + box, optionally + extra field
            errors.append(f"{path}:{i}: expected 5 (or 6) values, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:5]]
        except ValueError:
            errors.append(f"{path}:{i}: non-numeric values")
            continue
        if not (0 <= cls_id < num_classes):
            errors.append(f"{path}:{i}: class id {cls_id} out of range [0, {num_classes})")
        if not all(0.0 <= c <= 1.0 for c in coords):
            errors.append(f"{path}:{i}: coordinates must be normalized to [0, 1]")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO-format dataset config.")
    parser.add_argument("dataset_config", type=str, help="Path to configs/datasets/<name>.yaml")
    args = parser.parse_args()

    ds_cfg_path = Path(args.dataset_config)
    ds_cfg = yaml.safe_load(ds_cfg_path.read_text())

    data_yaml_path = Path(ds_cfg["path"])
    if not data_yaml_path.is_absolute():
        data_yaml_path = ds_cfg_path.parent / data_yaml_path
    data_yaml = yaml.safe_load(data_yaml_path.read_text())

    root = Path(data_yaml["path"])
    num_classes = len(data_yaml["names"])

    all_errors: list[str] = []
    for split_key in ("train", "val", "test"):
        if split_key not in data_yaml:
            continue
        img_dir = root / data_yaml[split_key]
        lbl_dir = root / data_yaml[split_key].replace("images", "labels")

        images = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

        if not images:
            all_errors.append(f"[{split_key}] no images found under {img_dir}")
            continue

        missing_labels = images.keys() - labels.keys()
        missing_images = labels.keys() - images.keys()
        for stem in missing_labels:
            all_errors.append(f"[{split_key}] {stem}: image has no matching label file")
        for stem in missing_images:
            all_errors.append(f"[{split_key}] {stem}: label file has no matching image")

        for lbl_path in labels.values():
            all_errors.extend(_validate_label_file(lbl_path, num_classes))

        print(f"[{split_key}] {len(images)} images, {len(labels)} labels checked")

    if all_errors:
        print(f"\n{len(all_errors)} issue(s) found:")
        for err in all_errors[:50]:
            print(f"  - {err}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more")
        sys.exit(1)

    print("\nDataset OK: no issues found.")


if __name__ == "__main__":
    main()
