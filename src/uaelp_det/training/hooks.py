"""Training hooks for EMA, gradient clipping, and quantization phases."""

from __future__ import annotations

from typing import Any


class TrainingHook:
    """Base hook interface."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:  # noqa: D401
        """Override to run logic at the beginning of each epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        """Override to run logic at the end of each epoch."""


class EMAHook(TrainingHook):
    """Placeholder for EMA weight updates."""

    def __init__(self, decay: float = 0.9999) -> None:
        self.decay = decay


class GradClipHook(TrainingHook):
    """Placeholder for gradient clipping."""

    def __init__(self, max_norm: float) -> None:
        self.max_norm = max_norm


class QATHook(TrainingHook):
    """Placeholder hook to enable QAT after a given epoch."""

    def __init__(self, start_epoch: int) -> None:
        self.start_epoch = start_epoch
