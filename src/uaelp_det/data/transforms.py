"""Transformation factories describing augmentation strategy."""

from __future__ import annotations

from typing import Any, Dict, List


def build_train_transforms(cfg: Dict[str, Any]) -> List[Any]:
    """
    Return a placeholder list representing the training augmentation pipeline.

    Actual augmentation ops (letterbox, jitter, color, blur, occlusion, etc.) should
    be implemented here using a library such as Albumentations or torchvision once
    dependencies are available.
    """

    _ = cfg  # Reserved for future use.
    return []


def build_eval_transforms(cfg: Dict[str, Any]) -> List[Any]:
    """Return placeholder evaluation transforms (typically just letterbox + normalize)."""
    _ = cfg
    return []
