"""Loss function definitions shared across detector backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LossConfig:
    """Hyperparameters for the unified detection loss."""

    lambda_box: float = 2.0
    lambda_l1: float = 1.0
    lambda_cls: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


class UnifiedDetectionLoss:
    """Placeholder detection loss; wire up actual PyTorch ops later."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, predictions: Any, targets: Any) -> Dict[str, float]:
        _ = predictions, targets
        raise NotImplementedError("Implement detection loss computation.")


def build_loss(cfg: Dict[str, Any]) -> UnifiedDetectionLoss:
    """Instantiate the unified detection loss."""
    config = LossConfig(
        lambda_box=cfg.get("lambda_box", 2.0),
        lambda_l1=cfg.get("lambda_l1", 1.0),
        lambda_cls=cfg.get("lambda_cls", 1.0),
        focal_alpha=cfg.get("focal_alpha", 0.25),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )
    return UnifiedDetectionLoss(config)
