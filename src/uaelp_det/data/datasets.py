"""Dataset builders and registry for detector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.data import Dataset  # type: ignore
except ImportError:  # pragma: no cover - torch installed later
    Dataset = object  # noqa: N806  (retain class-style name)


@dataclass
class DatasetConfig:
    """Lightweight description of a dataset split."""

    name: str
    root: Path
    annotations_file: Optional[Path] = None
    cache_dir: Optional[Path] = None


class LicensePlateDataset(Dataset):
    """Placeholder dataset exposing basic Dataset API for future implementation."""

    def __init__(self, config: DatasetConfig, transforms: Optional[Any] = None) -> None:
        self.config = config
        self.transforms = transforms
        self._items: list[Any] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Implement dataset item loading logic.")


def build_dataset(cfg: Dict[str, Any]) -> LicensePlateDataset:
    """Construct the dataset instance from a configuration dictionary."""
    dataset_cfg = DatasetConfig(
        name=cfg["name"],
        root=Path(cfg["root"]),
        annotations_file=Path(cfg["annotations"]) if cfg.get("annotations") else None,
        cache_dir=Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None,
    )
    transforms = cfg.get("transforms")
    return LicensePlateDataset(dataset_cfg, transforms=transforms)
