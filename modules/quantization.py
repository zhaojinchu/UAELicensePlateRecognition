"""Quantization helpers aligning with the deployment roadmap."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable

import torch


def prepare_qat(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Prepare the model for quantization-aware training.

    Fuses Conv+BN+Act layers and inserts fake-quant observers configured for the
    requested backend (default: fbgemm for x86 / TensorRT compatibility).
    """

    backend = cfg.get("backend", "fbgemm")
    torch.backends.quantized.engine = backend
    was_training = model.training

    # Module fusion requires evaluation mode to freeze BatchNorm statistics.
    model.eval()

    if hasattr(model, "fuse_model"):
        model.fuse_model()

    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    # Switch back to training for QAT preparation so observers/fake-quant modules are active.
    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)

    if not was_training:
        model.eval()
    return model


def calibrate_int8(
    model: torch.nn.Module,
    data_loader: Iterable[Any],
    cfg: Dict[str, Any],
) -> torch.nn.Module:
    """
    Run post-training INT8 calibration using a copy of the provided model.

    Returns the quantized model instance ready for export as ONNX/TensorRT plan.
    """

    backend = cfg.get("backend", "fbgemm")
    torch.backends.quantized.engine = backend

    model_copy = copy.deepcopy(model).eval()
    if hasattr(model_copy, "fuse_model"):
        model_copy.fuse_model()

    qconfig = torch.ao.quantization.get_default_qconfig(backend)
    model_copy.qconfig = qconfig
    prepared = torch.ao.quantization.prepare(model_copy, inplace=False)

    device = next(model.parameters()).device
    max_batches = cfg.get("max_batches")

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            prepared(images.to(device))
            if max_batches and (batch_idx + 1) >= max_batches:
                break

    quantized = torch.ao.quantization.convert(prepared, inplace=False)
    return quantized
