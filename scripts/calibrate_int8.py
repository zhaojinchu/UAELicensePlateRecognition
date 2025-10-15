"""Calibrate INT8 TensorRT engine using PTQ."""

from __future__ import annotations

from pathlib import Path


def main(checkpoint_path: str, calibration_config: str) -> None:
    """Run calibration over a held-out dataset subset."""
    _ = checkpoint_path, calibration_config
    raise NotImplementedError("Implement calibration workflow.")


if __name__ == "__main__":
    cfg = Path("configs/quantization/ptq.yaml")
    ckpt = Path("runs/latest/checkpoints/best_map5095.pth")
    main(str(ckpt), str(cfg))
