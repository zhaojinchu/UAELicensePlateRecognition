"""Metric logging utilities for training and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torchvision.ops import box_iou

from .losses import center_to_xyxy


METRIC_FIELDS: Sequence[str] = (
    "run_id",
    "time_epoch_start",
    "time_epoch_end",
    "epoch",
    "steps",
    "lr",
    "train_loss",
    "train_box",
    "train_cls",
    "train_dfl",
    "val_loss",
    "val_box",
    "val_cls",
    "val_dfl",
    "map50",
    "map5095",
    "ap_small",
    "ar_small",
    "fps",
    "gpu_mem_mb",
    "precision",
    "recall",
    "f1",
    "num_params",
    "model_size_mb",
    "grad_norm",
    "ema_on",
    "amp_on",
    "qat_on",
)


@dataclass
class DetectionMetricsLogger:
    """Write metrics to CSV and JSONL for reproducibility."""

    csv_path: Path
    jsonl_path: Path
    header_written: bool = field(default=False, init=False)

    def log(self, record: Dict[str, object]) -> None:
        if not self.header_written:
            self._write_header()
            self.header_written = True

        csv_line = ",".join(str(record.get(field, "")) for field in METRIC_FIELDS)
        with self.csv_path.open("a", encoding="utf-8") as csv_file:
            csv_file.write(f"{csv_line}\n")

        with self.jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            json.dump(record, jsonl_file)
            jsonl_file.write("\n")

    def _write_header(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8") as csv_file:
            csv_file.write(",".join(METRIC_FIELDS) + "\n")
        with self.jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            jsonl_file.write("")


def compute_detection_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    thresholds: Iterable[float],
    score_threshold: float,
    small_area_threshold: float,
) -> Dict[str, float]:
    """Compute simple detection metrics compatible with the project plan."""

    thresholds = list(thresholds)
    stats = {thr: {"tp": 0, "fp": 0, "fn": 0, "tp_small": 0, "fn_small": 0} for thr in thresholds}

    for pred, target in zip(predictions, targets, strict=True):
        pred_scores = pred["scores"].detach().cpu()
        pred_boxes = pred["boxes"].detach().cpu()
        keep = pred_scores >= score_threshold
        pred_scores = pred_scores[keep]
        pred_boxes = pred_boxes[keep]

        gt_boxes = target["boxes"].detach().cpu()
        small_mask = (gt_boxes[:, 2] * gt_boxes[:, 3]) < small_area_threshold if gt_boxes.numel() else torch.zeros(0, dtype=torch.bool)

        if gt_boxes.numel() == 0:
            for thr in thresholds:
                stats[thr]["fp"] += int(pred_boxes.size(0))
            continue

        if pred_boxes.numel() == 0:
            for thr in thresholds:
                stats[thr]["fn"] += int(gt_boxes.size(0))
                stats[thr]["fn_small"] += int(small_mask.sum())
            continue

        pred_xyxy = center_to_xyxy(pred_boxes)
        gt_xyxy = center_to_xyxy(gt_boxes)

        ious = box_iou(pred_xyxy, gt_xyxy)
        order = torch.argsort(pred_scores, descending=True)

        for thr in thresholds:
            matched_gt = set()
            matched_pred = set()
            tp_small = 0

            for idx in order.tolist():
                best_iou, gt_idx = (ious[idx].max(dim=0))
                gt_index = int(gt_idx.item())
                if best_iou >= thr and gt_index not in matched_gt:
                    matched_gt.add(gt_index)
                    matched_pred.add(idx)
                    if small_mask[gt_index]:
                        tp_small += 1

            tp = len(matched_pred)
            fp = int(pred_boxes.size(0)) - tp
            fn = int(gt_boxes.size(0)) - tp
            fn_small = int(small_mask.sum()) - tp_small

            stats[thr]["tp"] += tp
            stats[thr]["fp"] += fp
            stats[thr]["fn"] += fn
            stats[thr]["tp_small"] += tp_small
            stats[thr]["fn_small"] += max(0, fn_small)

    tp50 = stats[0.5]["tp"] if 0.5 in stats else 0
    fp50 = stats[0.5]["fp"] if 0.5 in stats else 0
    fn50 = stats[0.5]["fn"] if 0.5 in stats else 0
    tp_small50 = stats[0.5]["tp_small"] if 0.5 in stats else 0
    fn_small50 = stats[0.5]["fn_small"] if 0.5 in stats else 0

    precision = tp50 / max(1, tp50 + fp50)
    recall = tp50 / max(1, tp50 + fn50)
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-7)
        if precision + recall > 0
        else 0.0
    )

    map_values = [
        stats[thr]["tp"] / max(1, stats[thr]["tp"] + stats[thr]["fn"]) for thr in thresholds
    ]
    map50 = map_values[thresholds.index(0.5)] if 0.5 in thresholds else 0.0
    map5095 = sum(map_values) / len(map_values) if map_values else 0.0

    recall_small = tp_small50 / max(1, tp_small50 + fn_small50)

    return {
        "map50": map50,
        "map5095": map5095,
        "ap_small": recall_small,
        "ar_small": recall_small,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
