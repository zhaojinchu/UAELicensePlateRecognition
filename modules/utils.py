"""Generic helpers used across the detector project."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import numpy as np
import torch
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Create a basic logger that logs to stdout and optionally a file."""
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
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_dataset_id(paths: Sequence[Path]) -> str:
    """Return a stable hash for the dataset based on file paths."""
    canonical = sorted(str(path).replace("\\", "/") for path in paths)
    data = "\n".join(canonical).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def get_git_commit(repo_root: Path) -> str:
    """Best-effort retrieval of the current git commit hash."""
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        ref_path = repo_root / ".git" / ref.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return ref[:40] if ref else "unknown"


def atomic_save(obj: Any, path: Path) -> None:
    """Atomically write serialized tensors/objects to disk."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(obj, tmp.name)
        tmp.flush()
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


class AverageMeter:
    """Track average values for timing/metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute global L2 norm of gradients."""
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.data.norm(2) ** 2)
    return math.sqrt(total)


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            model_params = dict(model.named_parameters())
            for name, ema_param in ema_params.items():
                model_param = model_params[name]
                ema_param.copy_(ema_param * self.decay + (1.0 - self.decay) * model_param.detach())

            ema_buffers = dict(self.ema.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name, buffer in ema_buffers.items():
                buffer.copy_(model_buffers[name])

    @contextlib.contextmanager
    def apply_shadow(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily swap model parameters with EMA weights for evaluation."""
        param_backup: List[torch.Tensor] = []
        buffer_backup: List[torch.Tensor] = []
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters(), strict=True):
                param_backup.append(model_param.data.clone())
                model_param.data.copy_(ema_param.data)
            for ema_buffer, model_buffer in zip(self.ema.buffers(), model.buffers(), strict=True):
                buffer_backup.append(model_buffer.data.clone())
                model_buffer.data.copy_(ema_buffer.data)
        try:
            yield
        finally:
            with torch.no_grad():
                for model_param, backup in zip(model.parameters(), param_backup, strict=True):
                    model_param.data.copy_(backup)
                for model_buffer, backup in zip(model.buffers(), buffer_backup, strict=True):
                    model_buffer.data.copy_(backup)


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Persist configuration dictionary as JSON for reproducibility."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
