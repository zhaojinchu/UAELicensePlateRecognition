"""Evaluate trained detector checkpoints."""

from __future__ import annotations

from pathlib import Path


def main(config_path: str, checkpoint_path: str) -> None:
    """Run evaluation using stored metrics configuration."""
    _ = config_path, checkpoint_path
    raise NotImplementedError("Implement evaluation entrypoint.")


if __name__ == "__main__":
    default_cfg = Path("configs/training/default.yaml")
    dummy_ckpt = Path("runs/latest/checkpoints/last.pth")
    main(str(default_cfg), str(dummy_ckpt))
