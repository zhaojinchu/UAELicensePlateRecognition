"""Model registry with QAT-aware detector backbones."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Sequence

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        groups: int = 1,
        activation: nn.Module | None = None,
        bias: bool = False,
    ) -> None:
        padding = kernel_size // 2
        act_module = activation if activation is not None else nn.SiLU(inplace=True)
        if activation is None:
            act_module = nn.SiLU(inplace=True)
        modules = OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                    ),
                ),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("act", act_module if act_module is not None else nn.Identity()),
            ]
        )
        super().__init__(modules)

    def fuse_model(self) -> None:
        act = getattr(self, "act", None)
        if isinstance(act, (nn.SiLU, nn.Identity)):
            # PyTorch does not fuse SiLU yet; fall back to Conv+BN.
            torch.ao.quantization.fuse_modules(self, [["conv", "bn"]], inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, [["conv", "bn", "act"]], inplace=True)


class Bottleneck(nn.Module):
    """Standard YOLO bottleneck layer."""

    def __init__(self, channels: int, shortcut: bool = True, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3)
        self.use_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.use_shortcut:
            out = out + x
        return out

    def fuse_model(self) -> None:
        self.cv1.fuse_model()
        self.cv2.fuse_model()


class C2fBlock(nn.Module):
    """C2f block from YOLOv8/v10 (split-transform-merge)."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(in_channels, out_channels, 1)
        self.cv2 = ConvBNAct(in_channels, out_channels, 1)
        self.blocks = nn.ModuleList(
            [Bottleneck(out_channels, shortcut=False, expansion=0.5) for _ in range(num_layers)]
        )
        self.cv3 = ConvBNAct(out_channels * (num_layers + 2), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x), self.cv2(x)]
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv3(torch.cat(y, dim=1))

    def fuse_model(self) -> None:
        self.cv1.fuse_model()
        self.cv2.fuse_model()
        for block in self.blocks:
            block.fuse_model()
        self.cv3.fuse_model()


class CSPBlock(nn.Module):
    """CSP bottleneck stack used in YOLOv5 nano variants."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.cv2 = ConvBNAct(in_channels, hidden, 1)
        self.blocks = nn.Sequential(
            *[Bottleneck(hidden, shortcut=True, expansion=1.0) for _ in range(num_layers)]
        )
        self.cv3 = ConvBNAct(hidden * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))

    def fuse_model(self) -> None:
        self.cv1.fuse_model()
        self.cv2.fuse_model()
        for block in self.blocks:
            block.fuse_model()
        self.cv3.fuse_model()


class GhostConv(nn.Module):
    """Ghost convolution used by lightweight YOLO variants."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 1)
        self.primary = ConvBNAct(in_channels, hidden, kernel_size, stride=stride)
        self.cheap = ConvBNAct(hidden, hidden, 3, stride=1, groups=hidden)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        z = self.cheap(y)
        out = torch.cat([y, z], dim=1)
        return out[:, : self.out_channels]

    def fuse_model(self) -> None:
        self.primary.fuse_model()
        self.cheap.fuse_model()


class SPPF(nn.Module):
    """Spatial pyramid pooling fast module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

    def fuse_model(self) -> None:
        self.cv1.fuse_model()
        self.cv2.fuse_model()


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for mobile backbones."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pw = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

    def fuse_model(self) -> None:
        self.dw.fuse_model()
        self.pw.fuse_model()


class BaseDetector(nn.Module):
    """Common scaffolding for detector backbones."""

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.quant = QuantStub()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dequant = DeQuantStub()
        self.cls_head = nn.Linear(feature_dim, num_classes)
        self.box_head = nn.Linear(feature_dim, 4)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return backbone features (to be provided by subclasses)."""
        raise NotImplementedError

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Convert backbone output to a flat embedding."""
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        elif features.ndim == 3:
            features = features.mean(dim=1)
        return features

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.quant(images)
        features = self.encode(x)
        embedding = self.project_features(features)
        embedding = self.dequant(embedding)

        cls_logits = self.cls_head(embedding)
        if self.num_classes == 1 and cls_logits.ndim == 2:
            cls_logits = cls_logits.squeeze(-1)
        boxes = torch.sigmoid(self.box_head(embedding))
        return {"cls_logits": cls_logits, "boxes": boxes}

    def fuse_model(self) -> None:  # pragma: no cover - to be overridden
        """Fuse Conv+BN(+Act) where applicable (implemented by subclasses)."""
        return


class ToyDetector(BaseDetector):
    """Minimal CNN baseline."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, feature_dim=128)
        self.stem = ConvBNAct(3, 32, 3, stride=2)
        self.block1 = ConvBNAct(32, 64, 3, stride=2)
        self.block2 = ConvBNAct(64, 128, 3, stride=2)
        self.block3 = ConvBNAct(128, 128, 3, stride=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def fuse_model(self) -> None:
        self.stem.fuse_model()
        self.block1.fuse_model()
        self.block2.fuse_model()
        self.block3.fuse_model()


class YOLOv8Nano(BaseDetector):
    """Anchor-free CNN inspired by YOLOv8-n."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, feature_dim=256)
        self.stem = ConvBNAct(3, 32, 3, stride=2)
        self.stage1 = C2fBlock(32, 64, num_layers=1)
        self.down1 = ConvBNAct(64, 128, 3, stride=2)
        self.stage2 = C2fBlock(128, 128, num_layers=2)
        self.down2 = ConvBNAct(128, 256, 3, stride=2)
        self.stage3 = C2fBlock(256, 256, num_layers=2)
        self.sppf = SPPF(256, 256, kernel_size=5)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.sppf(x)
        return x

    def fuse_model(self) -> None:
        self.stem.fuse_model()
        self.stage1.fuse_model()
        self.down1.fuse_model()
        self.stage2.fuse_model()
        self.down2.fuse_model()
        self.stage3.fuse_model()
        self.sppf.fuse_model()


class YOLOv5Nano(BaseDetector):
    """Anchor-based CNN inspired by YOLOv5-n backbone."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, feature_dim=256)
        self.stem = ConvBNAct(3, 32, 3, stride=2)
        self.stage1 = CSPBlock(32, 64, num_layers=1)
        self.down1 = ConvBNAct(64, 128, 3, stride=2)
        self.stage2 = CSPBlock(128, 128, num_layers=2)
        self.down2 = ConvBNAct(128, 256, 3, stride=2)
        self.stage3 = CSPBlock(256, 256, num_layers=2)
        self.sppf = SPPF(256, 256, kernel_size=5)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.sppf(x)
        return x

    def fuse_model(self) -> None:
        self.stem.fuse_model()
        self.stage1.fuse_model()
        self.down1.fuse_model()
        self.stage2.fuse_model()
        self.down2.fuse_model()
        self.stage3.fuse_model()
        self.sppf.fuse_model()


class YOLOv10Nano(BaseDetector):
    """NMS-free CNN skeleton inspired by YOLOv10-n."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, feature_dim=256)
        self.stem = ConvBNAct(3, 32, 3, stride=2)
        self.stage1 = GhostConv(32, 64, 3)
        self.down1 = ConvBNAct(64, 128, 3, stride=2)
        self.stage2 = C2fBlock(128, 128, num_layers=2)
        self.down2 = ConvBNAct(128, 256, 3, stride=2)
        self.stage3 = C2fBlock(256, 256, num_layers=2)
        self.context = ConvBNAct(256, 256, 3, stride=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.context(x)
        attn = self.attn(x)
        return x * attn

    def fuse_model(self) -> None:
        self.stem.fuse_model()
        self.stage1.fuse_model()
        self.down1.fuse_model()
        self.stage2.fuse_model()
        self.down2.fuse_model()
        self.stage3.fuse_model()
        self.context.fuse_model()
        # self.attn contains pooling/conv only; nothing to fuse.


class RTDETRTiny(BaseDetector):
    """Tiny RT-DETR style hybrid with convolutional stem + transformer encoder."""

    def __init__(self, num_classes: int) -> None:
        embed_dim = 192
        super().__init__(num_classes, feature_dim=embed_dim)
        self.patch_embed = nn.Sequential(
            ConvBNAct(3, 64, 3, stride=2),
            ConvBNAct(64, 128, 3, stride=2),
            ConvBNAct(128, embed_dim, 3, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        n, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(n, h * w, c)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        return encoded

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=1)

    def fuse_model(self) -> None:
        for module in self.patch_embed:
            if hasattr(module, "fuse_model"):
                module.fuse_model()
        # Transformer layers are not fused.


class PPPicoDetS(BaseDetector):
    """PP-PicoDet-S inspired mobile backbone."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, feature_dim=160)
        self.stages = nn.ModuleList(
            [
                ConvBNAct(3, 32, 3, stride=2),
                DepthwiseSeparableConv(32, 64, stride=2),
                DepthwiseSeparableConv(64, 96, stride=2),
                DepthwiseSeparableConv(96, 128, stride=1),
                DepthwiseSeparableConv(128, 160, stride=1),
            ]
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        for stage in self.stages:
            x = stage(x)
        return x

    def fuse_model(self) -> None:
        for stage in self.stages:
            stage.fuse_model()


def _build_model_helper(cfg: Dict[str, Any], key: str, cls: Callable[[int], nn.Module]) -> nn.Module:
    num_classes = cfg.get("num_classes", 1)
    return cls(num_classes=num_classes)


@register_model("toy_cnn")
def _build_toy_cnn(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "toy_cnn", ToyDetector)


@register_model("yolov8n")
def _build_yolov8n(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "yolov8n", YOLOv8Nano)


@register_model("yolov5n")
def _build_yolov5n(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "yolov5n", YOLOv5Nano)


@register_model("yolov10n")
def _build_yolov10n(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "yolov10n", YOLOv10Nano)


@register_model("rtdetr_tiny")
def _build_rtdetr_tiny(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "rtdetr_tiny", RTDETRTiny)


@register_model("pp_picodet_s")
def _build_pp_picodet_s(cfg: Dict[str, Any]) -> nn.Module:
    return _build_model_helper(cfg, "pp_picodet_s", PPPicoDetS)
