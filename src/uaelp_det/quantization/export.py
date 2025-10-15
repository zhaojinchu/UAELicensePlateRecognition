"""Export helpers for quantized models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def export_to_onnx(model: Any, output_path: Path, cfg: Dict[str, Any]) -> None:
    """
    Export a trained model to ONNX.

    Ensure the exported graph sticks to TensorRT-friendly operators and optionally
    includes Q/DQ nodes when QAT is used.
    """
    _ = model, output_path, cfg
    raise NotImplementedError("Implement ONNX export logic.")
