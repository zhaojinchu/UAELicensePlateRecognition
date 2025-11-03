"""Post-processing helpers for dense detector heads."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torchvision.ops import nms

from .losses import center_to_xyxy


def suppress_dense_predictions(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
    max_detections: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply score thresholding and NMS to dense predictions from the shared head.

    The function expects `scores` to contain objectness probabilities for each spatial
    location and `boxes` to contain normalized center-width-height box predictions
    matching the same layout. All tensors are processed on their existing device.
    """
    if scores.ndim != 1 or boxes.ndim != 2 or boxes.size(-1) != 4:
        raise ValueError("Expected scores shape (N,) and boxes shape (N, 4) in cxcywh format.")

    if max_detections is not None and max_detections <= 0:
        empty_scores = scores.new_zeros((0,), dtype=scores.dtype)
        empty_boxes = boxes.new_zeros((0, 4), dtype=boxes.dtype)
        return empty_scores, empty_boxes

    keep_mask = scores >= score_threshold
    if not torch.any(keep_mask):
        empty_scores = scores.new_zeros((0,), dtype=scores.dtype)
        empty_boxes = boxes.new_zeros((0, 4), dtype=boxes.dtype)
        return empty_scores, empty_boxes

    filtered_scores = scores[keep_mask]
    filtered_boxes = boxes[keep_mask]

    if filtered_boxes.numel() == 0:
        empty_scores = scores.new_zeros((0,), dtype=scores.dtype)
        empty_boxes = boxes.new_zeros((0, 4), dtype=boxes.dtype)
        return empty_scores, empty_boxes

    boxes_xyxy = center_to_xyxy(filtered_boxes)
    keep_indices = nms(boxes_xyxy, filtered_scores, iou_threshold)
    if max_detections is not None:
        keep_indices = keep_indices[:max_detections]

    return filtered_scores[keep_indices], filtered_boxes[keep_indices]


__all__ = ["suppress_dense_predictions"]
