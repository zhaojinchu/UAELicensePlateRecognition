"""Visualize ground-truth and predicted bounding boxes for a single sample."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from modules import build_model, ensure_dir, load_yaml
from modules.data import LicensePlateDataset, prepare_transforms, resolve_dataset_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detector predictions vs. ground-truth.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to training config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a model checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to sample from.",
    )
    parser.add_argument("--index", type=int, default=None, help="Index of the sample to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --index is omitted.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence required to draw a predicted box.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load EMA weights from the checkpoint when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization (directories will be created).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure with matplotlib (requires GUI backend).",
    )
    return parser.parse_args()


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("CUDA requested but not available.")
    return torch.device("cpu")


def gather_image_paths(image_dir: Path, split_file: Path | None) -> List[Path]:
    valid_exts = ("", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if split_file and split_file.exists():
        items: List[Path] = []
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            entry = raw_line.strip()
            if not entry:
                continue
            candidate = Path(entry)
            if candidate.is_absolute():
                if candidate.exists():
                    items.append(candidate)
                continue
            for ext in valid_exts:
                resolved = image_dir / f"{entry}{ext}"
                if resolved.exists():
                    items.append(resolved)
                    break
        if items:
            return items
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def normalize_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    cls_logits = outputs["cls_logits"]
    boxes = outputs["boxes"]
    scores = torch.sigmoid(cls_logits)
    if scores.ndim == 4:
        scores = scores.squeeze(1)
    flat_scores = scores.view(scores.size(0), -1)
    flat_boxes = boxes.view(boxes.size(0), 4, -1).permute(0, 2, 1)
    return flat_scores, flat_boxes


def yolo_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    cxcy = boxes[:, :2].clamp(0.0, 1.0)
    wh = boxes[:, 2:].clamp(min=1e-6, max=1.0)

    cx = cxcy[:, 0] * width
    cy = cxcy[:, 1] * height
    w = wh[:, 0] * width
    h = wh[:, 1] * height

    x1 = (cx - 0.5 * w).clamp(min=0.0, max=float(width - 1))
    y1 = (cy - 0.5 * h).clamp(min=0.0, max=float(height - 1))
    x2 = (cx + 0.5 * w).clamp(min=0.0, max=float(width - 1))
    y2 = (cy + 0.5 * h).clamp(min=0.0, max=float(height - 1))
    return torch.stack((x1, y1, x2, y2), dim=-1)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    image = image_tensor * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def draw_boxes(
    ax: plt.Axes,
    boxes: Iterable[Sequence[float]],
    color: str,
    label: str,
    scores: Sequence[float] | None = None,
) -> None:
    for idx, coords in enumerate(boxes):
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (float(x1), float(y1)),
            float(width),
            float(height),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        caption = label
        if scores is not None and idx < len(scores):
            caption = f"{caption} {scores[idx]:.2f}"
        ax.text(
            float(x1),
            float(max(0.0, y1 - 5)),
            caption,
            fontsize=8,
            color="white",
            bbox={"facecolor": color, "alpha": 0.6, "pad": 1.5},
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    config = load_yaml(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    device = select_device(args.device)

    model = build_model(model_cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict_key = "ema_state_dict" if args.use_ema and checkpoint.get("ema_state_dict") else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    model.to(device).eval()

    resolved = resolve_dataset_paths(dataset_cfg, split=args.split)
    image_paths = gather_image_paths(resolved["image_dir"], resolved["split_file"])
    if not image_paths:
        raise RuntimeError(f"No images found for split '{args.split}' in {resolved['image_dir']}.")

    transforms = prepare_transforms(resolved, train=False)
    dataset = LicensePlateDataset(image_paths, resolved["label_dir"], transforms)
    num_samples = len(dataset)

    index = args.index if args.index is not None else random.randrange(num_samples)
    index %= num_samples

    sample = dataset[index]
    image_tensor = sample["image"]
    target = sample["target"]
    gt_boxes = target["boxes"]
    max_targets = dataset_cfg.get("max_targets_per_image", 1)
    if max_targets and max_targets > 0 and gt_boxes.numel() > 0:
        areas = gt_boxes[:, 2] * gt_boxes[:, 3]
        topk = torch.topk(areas, k=min(max_targets, gt_boxes.size(0))).indices
        gt_boxes = gt_boxes[topk]

    input_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    score_map, box_map = normalize_outputs(outputs)

    scores = score_map[0]
    boxes = box_map[0]

    keep = scores >= args.score_threshold
    pred_scores = scores[keep]
    pred_boxes = boxes[keep]

    if pred_scores.numel() > 0:
        order = torch.argsort(pred_scores, descending=True)
        pred_scores = pred_scores[order]
        pred_boxes = pred_boxes[order]

    height, width = image_tensor.shape[1:]
    mean = dataset_cfg.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    std = dataset_cfg.get("normalization", {}).get("std", [0.229, 0.224, 0.225])

    vis_image = denormalize_image(image_tensor, mean, std)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(vis_image)
    ax.set_title(f"Split={args.split} | Index={index} | Device={device.type.upper()}")
    ax.axis("off")

    if gt_boxes.numel() > 0:
        gt_xyxy = yolo_to_xyxy(gt_boxes, width, height).cpu().numpy()
        draw_boxes(ax, gt_xyxy, color="#00FF00", label="GT")

    if pred_boxes.numel() > 0:
        pred_xyxy = yolo_to_xyxy(pred_boxes.cpu(), width, height).cpu().numpy()
        draw_boxes(ax, pred_xyxy, color="#FF4C4C", label="Pred", scores=pred_scores.cpu().tolist())

    if args.output:
        ensure_dir(args.output.parent)
        fig.savefig(args.output, bbox_inches="tight", dpi=200)
        print(f"Visualization saved to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
