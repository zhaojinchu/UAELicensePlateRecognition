"""Lightweight model registry with QAT-ready toy detector placeholder."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict

import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator used to register model builder functions."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the requested model."""
    name = cfg.get("name")
    if not name:
        raise ValueError("Model configuration must include a 'name'.")
    if name not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Model '{name}' is not registered. Use register_model to add it."
        )
    return MODEL_REGISTRY[name](cfg)


class ConvBNAct(nn.Sequential):
    """Conv-BN-Activation block with fusion helper."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("act", nn.SiLU(inplace=True)),
                ]
            )
        )

    def fuse_model(self) -> None:
        if isinstance(self.act, nn.SiLU):
            # Fuser mappings do not cover SiLU yet; fall back to Conv+BN fusion.
            torch.ao.quantization.fuse_modules(self, [["conv", "bn"]], inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, [["conv", "bn", "act"]], inplace=True)


@register_model("toy_cnn")
def _build_toy_cnn(cfg: Dict[str, Any]) -> nn.Module:
    """Very small CNN placeholder so training loop can be smoke-tested."""
    num_classes = cfg.get("num_classes", 1)
    return ToyDetector(num_classes=num_classes)


class ToyDetector(nn.Module):
    """
    Minimal detector skeleton that emits dummy box predictions.

    Replace with a real detection architecture (YOLO, RT-DETR, etc.) when ready.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)
        self.block1 = ConvBNAct(32, 64, kernel_size=3, stride=2)
        self.block2 = ConvBNAct(64, 128, kernel_size=3, stride=2)
        self.block3 = ConvBNAct(128, 128, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(128, num_classes)
        self.box_head = nn.Linear(128, 4)
        self.dequant = DeQuantStub()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.quant(images)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = self.dequant(x)

        cls_logits = self.cls_head(x)
        if cls_logits.shape[-1] == 1:
            cls_logits = cls_logits.squeeze(-1)
        bbox_raw = self.box_head(x)
        boxes = torch.sigmoid(bbox_raw)
        return {"cls_logits": cls_logits, "boxes": boxes}

    def fuse_model(self) -> None:
        """Fuse Conv+BN+Act layers prior to (Q)AT or export."""
        self.stem.fuse_model()
        self.block1.fuse_model()
        self.block2.fuse_model()
        self.block3.fuse_model()
