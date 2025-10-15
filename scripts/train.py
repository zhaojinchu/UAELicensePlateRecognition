"""Entry point for training detector models."""

from __future__ import annotations

from pathlib import Path


def main(config_path: str) -> None:
    """Load configs, build components, and launch training."""
    _ = config_path
    raise NotImplementedError("Implement training entrypoint.")


if __name__ == "__main__":
    default_cfg = Path("configs/training/default.yaml")
    main(str(default_cfg))
