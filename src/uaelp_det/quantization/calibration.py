"""Post-training quantization calibration helpers."""

from __future__ import annotations

from typing import Any, Iterable


def calibrate_model(model: Any, data_loader: Iterable[Any], cfg: dict) -> None:
    """
    Run post-training INT8 calibration.

    Use a 200-500 image in-distribution calibration set with preprocessing aligned
    to the final deployment pipeline. Store calibration cache files in
    `data/calibration`.
    """
    _ = model, data_loader, cfg
    raise NotImplementedError("Implement calibration logic.")
