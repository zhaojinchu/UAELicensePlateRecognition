"""Quantization-aware training setup."""

from __future__ import annotations

from typing import Any, Dict


def setup_qat(model: Any, cfg: Dict[str, Any]) -> Any:
    """
    Prepare the model for quantization-aware training.

    Configure fake quantization modules, fuse supported layers, and ensure TensorRT
    compatibility (e.g., Conv+BN+Activation fusion and channel alignment).
    """
    _ = cfg
    raise NotImplementedError("Implement QAT preparation for the model.")
