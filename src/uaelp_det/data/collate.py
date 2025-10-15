"""Batch collation utilities for detection targets."""

from __future__ import annotations

from typing import Any, List, Tuple


def detection_collate(batch: List[Any]) -> Tuple[Any, Any]:
    """Placeholder collate function aligning with training targets."""
    _ = batch
    raise NotImplementedError("Implement detection batch collation.")
