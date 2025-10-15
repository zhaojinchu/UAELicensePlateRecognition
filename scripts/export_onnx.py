"""Export a trained detector to ONNX or TensorRT engines."""

from __future__ import annotations

from pathlib import Path


def main(checkpoint_path: str, export_config: str) -> None:
    """Stub for model export pipeline."""
    _ = checkpoint_path, export_config
    raise NotImplementedError("Implement export workflow.")


if __name__ == "__main__":
    cfg = Path("configs/export/onnx.yaml")
    ckpt = Path("runs/latest/checkpoints/best_map5095.pth")
    main(str(ckpt), str(cfg))
