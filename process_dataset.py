"""Normalize dataset layout and build the unified training set."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from modules import MergeConfig, ensure_dir, merge_coco_yolo_datasets

ROOT = Path(__file__).resolve().parent

COCO_SRC = ROOT / "coco_dataset"
YOLO_SRC = ROOT / "yolov8_dataset"

RAW_ROOT = ROOT / "data" / "raw"
COCO_RAW = RAW_ROOT / "coco_dataset"
YOLO_RAW = RAW_ROOT / "yolov8_dataset"

PROCESSED_ROOT = ROOT / "data" / "processed" / "unified"

SPLITS: Iterable[str] = ("train", "valid", "test")


def relocate_tree(src: Path, dst: Path) -> Path:
    """Move a directory tree into the data/raw hierarchy."""
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
    elif not dst.exists():
        raise FileNotFoundError(f"Neither source nor destination exists for {src}")
    return dst


def merge_coco_sources() -> None:
    for split in SPLITS:
        annotations = COCO_RAW / split / "_annotations.coco.json"
        images_dir = COCO_RAW / split
        if not annotations.exists():
            continue
        merge_coco_yolo_datasets(
            MergeConfig(
                output_root=PROCESSED_ROOT,
                coco_annotations=annotations,
                coco_images_dir=images_dir,
                overwrite=True,
            )
        )


def merge_yolo_sources() -> None:
    for split in SPLITS:
        labels_dir = YOLO_RAW / split / "labels"
        images_dir = YOLO_RAW / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        merge_coco_yolo_datasets(
            MergeConfig(
                output_root=PROCESSED_ROOT,
                yolo_labels_dir=labels_dir,
                yolo_images_dir=images_dir,
                overwrite=True,
            )
        )


def main() -> None:
    ensure_dir(RAW_ROOT)

    relocate_tree(COCO_SRC, COCO_RAW)
    relocate_tree(YOLO_SRC, YOLO_RAW)

    if PROCESSED_ROOT.exists():
        shutil.rmtree(PROCESSED_ROOT)
    ensure_dir(PROCESSED_ROOT)

    merge_coco_sources()
    merge_yolo_sources()

    print("Raw datasets relocated to:")
    print(f" - {COCO_RAW}")
    print(f" - {YOLO_RAW}")
    print(f"Unified dataset available at: {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
