"""Utility helpers for configuration, logging, and distributed training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - dependency resolved later
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def create_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger writing to stdout and optional file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def is_distributed_available() -> bool:
    """Return True if torch.distributed is ready."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - dependency resolved later
        return False
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    """Synchronize processes if distributed training is active."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
