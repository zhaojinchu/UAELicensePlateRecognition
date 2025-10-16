"""Dataset utilities, augmentation, and dataset merging helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


@dataclass
class MergeConfig:
    """Configuration describing dataset sources and output."""

    output_root: Path
    coco_annotations: Optional[Path] = None
    coco_images_dir: Optional[Path] = None
    yolo_labels_dir: Optional[Path] = None
    yolo_images_dir: Optional[Path] = None
    class_mapping: Optional[Dict[int, int]] = None
    overwrite: bool = False


def merge_coco_yolo_datasets(cfg: MergeConfig) -> None:
    """
    Merge COCO JSON and YOLO text labels into a single dataset directory.

    The resulting layout is compatible with the training pipeline:
        output_root/
            images/
            labels/
    """

    images_out = cfg.output_root / "images"
    labels_out = cfg.output_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if cfg.coco_annotations and cfg.coco_annotations.exists():
        _merge_coco(
            annotations_path=cfg.coco_annotations,
            images_dir=cfg.coco_images_dir,
            images_out=images_out,
            labels_out=labels_out,
            class_mapping=cfg.class_mapping or {},
            overwrite=cfg.overwrite,
        )

    if cfg.yolo_labels_dir and cfg.yolo_labels_dir.exists():
        _merge_yolo(
            labels_dir=cfg.yolo_labels_dir,
            images_dir=cfg.yolo_images_dir,
            images_out=images_out,
            labels_out=labels_out,
            overwrite=cfg.overwrite,
        )


def _merge_coco(
    annotations_path: Path,
    images_dir: Optional[Path],
    images_out: Path,
    labels_out: Path,
    class_mapping: Dict[int, int],
    overwrite: bool,
) -> None:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    id_to_name = {img["id"]: img["file_name"] for img in data.get("images", [])}
    image_meta = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}

    grouped: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        grouped.setdefault(image_id, []).append(ann)

    for image_id, annotations in grouped.items():
        if image_id not in id_to_name:
            continue
        file_name = id_to_name[image_id]
        width, height = image_meta.get(image_id, (None, None))
        if width in (None, 0) or height in (None, 0):
            continue

        stem = Path(file_name).stem
        label_path = labels_out / f"{stem}.txt"
        if label_path.exists() and not overwrite:
            continue

        with label_path.open("w", encoding="utf-8") as handle:
            for ann in annotations:
                category_id = ann["category_id"]
                target_id = class_mapping.get(category_id, category_id)
                x, y, w, h = ann["bbox"]
                x_c = (x + w / 2.0) / width
                y_c = (y + h / 2.0) / height
                w_n = w / width
                h_n = h / height
                handle.write(f"{target_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

        if images_dir:
            _copy_asset(images_dir / file_name, images_out / file_name, overwrite)


def _merge_yolo(
    labels_dir: Path,
    images_dir: Optional[Path],
    images_out: Path,
    labels_out: Path,
    overwrite: bool,
) -> None:
    for label_file in _iter_files(labels_dir, ".txt"):
        target_label = labels_out / label_file.name
        if target_label.exists() and not overwrite:
            continue
        shutil.copy2(label_file, target_label)

        if images_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{label_file.stem}{ext}"
                if candidate.exists():
                    _copy_asset(candidate, images_out / candidate.name, overwrite)
                    break


def _copy_asset(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        return
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_files(directory: Path, suffix: str) -> Iterable[Path]:
    for path in directory.rglob(f"*{suffix}"):
        if path.is_file():
            yield path


def resolve_dataset_paths(cfg: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Resolve dataset paths for the requested split (train/val/test)."""

    root = Path(cfg["root"])
    image_dir = root / cfg.get("image_dir", "images")
    label_dir = root / cfg.get("label_dir", "labels")
    split_key = cfg.get(f"split_{split}")
    split_file = Path(split_key) if split_key else None

    return {
        "name": cfg.get("name", "uae_license_plate"),
        "image_dir": image_dir,
        "label_dir": label_dir,
        "split_file": split_file,
        "class_names": cfg.get("class_names", ["license_plate"]),
        "image_size": tuple(cfg.get("image_size", [224, 224])),
        "letterbox_color": tuple(cfg.get("letterbox_color", [114, 114, 114])),
        "normalization": cfg.get("normalization", {}),
        "augmentations": cfg.get("augmentations", {}),
    }


def prepare_transforms(
    dataset_cfg: Dict[str, Any], *, train: bool
) -> A.Compose:
    """Create Albumentations transforms for training or evaluation."""

    image_size = dataset_cfg["image_size"]
    pad_color = dataset_cfg["letterbox_color"]
    norm_cfg = dataset_cfg["normalization"]
    mean = tuple(norm_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(norm_cfg.get("std", [0.229, 0.224, 0.225]))

    if train:
        aug_cfg = dataset_cfg["augmentations"]
        geometric = aug_cfg.get("geometric", {})
        photometric = aug_cfg.get("photometric", {})
        degradations = aug_cfg.get("degradations", {})
        occlusion = aug_cfg.get("occlusion", {})

        scale_range = geometric.get("scale_range", [0.5, 1.5])
        aspect_range = geometric.get("aspect_ratio", [0.9, 1.1])
        translate = geometric.get("translate", 0.1)
        rotate = geometric.get("rotation", 7.0)
        shear = geometric.get("shear", 5.0)
        perspective = geometric.get("perspective", 0.02)

        brightness = photometric.get("brightness", 0.2)
        contrast = photometric.get("contrast", 0.2)
        hsv_h = photometric.get("hsv_h", 5.0)
        hsv_s = photometric.get("hsv_s", 0.2)
        hsv_v = photometric.get("hsv_v", 0.2)

        motion_cfg = degradations.get("motion_blur", {})
        motion_kernel = tuple(motion_cfg.get("kernel", [3, 7]))
        gaussian_sigma = degradations.get("gaussian_blur_sigma", 1.2)
        noise_sigma = degradations.get("gaussian_noise_sigma", 0.01)
        jpeg_quality = tuple(degradations.get("jpeg_quality", [40, 90]))
        sharpening = degradations.get("sharpening", 0.05)

        cutout_cfg = occlusion.get("cutout", {})
        cutout_patches = cutout_cfg.get("patches", 3)
        cutout_max = cutout_cfg.get("max_size", 0.1)
        edge_occ_prob = occlusion.get("edge_occlusion_probability", 0.1)

        transforms = [
            A.RandomScale(
                scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
                interpolation=cv2.INTER_NEAREST,
                p=0.3,
            ),
            A.Affine(
                scale={
                    "x": (scale_range[0], scale_range[1]),
                    "y": (scale_range[0] * aspect_range[0], scale_range[1] * aspect_range[1]),
                },
                translate_percent={"x": (-translate, translate), "y": (-translate, translate)},
                rotate=(-rotate, rotate),
                shear=(-shear, shear),
                fit_output=False,
                interpolation=cv2.INTER_NEAREST,
                mode=cv2.BORDER_CONSTANT,
                cval=pad_color,
                p=0.7,
            ),
            A.Perspective(scale=(0.0, perspective), interpolation=cv2.INTER_NEAREST, p=0.2),
            A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_NEAREST),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=pad_color,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(hsv_h),
                sat_shift_limit=int(hsv_s * 255),
                val_shift_limit=int(hsv_v * 255),
                p=0.3,
            ),
            A.ColorTemperature(p=0.2),
            A.MotionBlur(blur_limit=motion_kernel, p=motion_cfg.get("probability", 0.2)),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.0, gaussian_sigma), p=0.2),
            A.GaussNoise(var_limit=(0.0, noise_sigma), p=0.2),
            A.ImageCompression(quality_lower=jpeg_quality[0], quality_upper=jpeg_quality[1], p=0.3),
            A.Sharpen(alpha=sharpening, lightness=0.0, p=0.2),
            A.RandomRain(p=0.1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(p=0.1, src_radius=35),
            A.Cutout(
                num_holes=cutout_patches,
                max_h_size=int(cutout_max * image_size[0]),
                max_w_size=int(cutout_max * image_size[1]),
                fill_value=pad_color,
                p=0.4,
            ),
            A.CoarseDropout(
                max_holes=2,
                max_height=int(0.2 * image_size[0]),
                max_width=int(0.3 * image_size[1]),
                fill_value=pad_color,
                p=edge_occ_prob,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_NEAREST),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=pad_color,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.0 if not train else 0.1,
        clip=True,
    )
    return A.Compose(transforms, bbox_params=bbox_params)


class LicensePlateDataset(Dataset):
    """Torch dataset that loads YOLO-formatted label files and applies augmentations."""

    def __init__(
        self,
        items: Sequence[Path],
        label_dir: Path,
        transforms: Optional[A.Compose],
    ) -> None:
        self.items = list(items)
        self.label_dir = label_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.items[index]
        image = self._read_image(image_path)
        boxes, labels = self._read_labels(image_path.stem)

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist(),
            )
            image_tensor = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)
            labels = np.array(transformed["class_labels"], dtype=np.int64)
        else:
            image_tensor = self._to_tensor(image)

        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if labels.size == 0:
            labels = np.zeros((0,), dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": image_path.stem,
            "path": str(image_path),
        }
        return {"image": image_tensor.float(), "target": target}

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_labels(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        label_file = self.label_dir / f"{stem}.txt"
        if not label_file.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes: List[List[float]] = []
        labels: List[int] = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            boxes.append(
                [
                    np.clip(x_c, 0.0, 1.0),
                    np.clip(y_c, 0.0, 1.0),
                    np.clip(w, 0.0, 1.0),
                    np.clip(h, 0.0, 1.0),
                ]
            )
            labels.append(cls_id)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor


def _load_split(split_file: Optional[Path], image_dir: Path) -> List[Path]:
    if not split_file or not split_file.exists():
        return sorted(image_dir.glob("*"))

    items: List[Path] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry:
            continue
        candidate = Path(entry)
        if candidate.is_absolute():
            items.append(candidate)
            continue
        for ext in ("", ".jpg", ".jpeg", ".png"):
            path = image_dir / f"{entry}{ext}"
            if path.exists():
                items.append(path)
                break
    return items


def create_dataloader(
    dataset_cfg: Dict[str, Any],
    *,
    train: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build DataLoader for a given split."""

    split = "train" if train else "val"
    resolved = resolve_dataset_paths(dataset_cfg, split=split)
    items = _load_split(resolved["split_file"], resolved["image_dir"])

    transforms = prepare_transforms(resolved, train=train)
    dataset = LicensePlateDataset(items, resolved["label_dir"], transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    images = torch.stack([item["image"] for item in batch])
    targets: List[Dict[str, Any]] = []
    for idx, item in enumerate(batch):
        target = dict(item["target"])
        target["batch_index"] = idx
        targets.append(target)
    return images, targets
