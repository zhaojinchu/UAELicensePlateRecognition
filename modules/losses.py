"""Unified detection loss combining focal, IoU, and L1 terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


@dataclass
class LossConfig:
    lambda_box: float = 2.0
    lambda_l1: float = 1.0
    lambda_cls: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


class UnifiedDetectionLoss:
    """Implements the shared loss described in the detector spec."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        self.cfg = LossConfig(
            lambda_box=cfg.get("lambda_box", 2.0),
            lambda_l1=cfg.get("lambda_l1", 1.0),
            lambda_cls=cfg.get("lambda_cls", 1.0),
            focal_alpha=cfg.get("focal_alpha", 0.25),
            focal_gamma=cfg.get("focal_gamma", 2.0),
        )

    def __call__(
        self,
        cls_logits: torch.Tensor,
        box_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        box_targets: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Classification/objectness loss with sigmoid focal.
        cls_loss = sigmoid_focal_loss(
            cls_logits.view(-1),
            cls_targets.view(-1),
            alpha=self.cfg.focal_alpha,
            gamma=self.cfg.focal_gamma,
            reduction="mean",
        )

        if positive_mask.any():
            pos_preds = box_preds.permute(0, 2, 3, 1)[positive_mask]
            pos_targets = box_targets.permute(0, 2, 3, 1)[positive_mask]
            iou_loss = ciou_loss(pos_preds, pos_targets).mean()
            l1_loss = F.l1_loss(pos_preds, pos_targets, reduction="mean")
        else:
            iou_loss = cls_loss.new_zeros(())
            l1_loss = cls_loss.new_zeros(())

        total = (
            self.cfg.lambda_cls * cls_loss
            + self.cfg.lambda_box * iou_loss
            + self.cfg.lambda_l1 * l1_loss
        )

        return {
            "total": total,
            "cls": cls_loss,
            "box": iou_loss,
            "l1": l1_loss,
            "dfl": cls_loss.new_zeros(()),
        }


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete IoU loss between predicted and target boxes.

    Boxes are expected in center-width-height format normalized to [0, 1].
    """

    pred_xyxy = center_to_xyxy(pred_boxes)
    target_xyxy = center_to_xyxy(target_boxes)

    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union = pred_area + target_area - inter_area + 1e-7
    iou = inter_area / union

    center_pred = pred_boxes[:, :2]
    center_target = target_boxes[:, :2]
    center_dist = torch.sum((center_pred - center_target) ** 2, dim=-1)

    enc_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enc_w = enc_x2 - enc_x1
    enc_h = enc_y2 - enc_y1
    enc_diag = enc_w**2 + enc_h**2 + 1e-7

    v = _aspect_ratio_term(pred_boxes[:, 2:], target_boxes[:, 2:])
    with torch.no_grad():
        alpha = v / (1.0 - iou + v)

    ciou = iou - (center_dist / enc_diag) - alpha * v
    return 1.0 - ciou


def center_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert cx, cy, w, h boxes (normalized) to x1, y1, x2, y2."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _aspect_ratio_term(pred_wh: torch.Tensor, target_wh: torch.Tensor) -> torch.Tensor:
    """Aspect ratio penalty used in CIoU."""
    pred_w, pred_h = pred_wh.unbind(-1)
    target_w, target_h = target_wh.unbind(-1)

    pred_ratio = torch.atan(pred_w / (pred_h + 1e-7))
    target_ratio = torch.atan(target_w / (target_h + 1e-7))
    v = (4.0 / (torch.pi**2)) * (pred_ratio - target_ratio) ** 2
    return v
