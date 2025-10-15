"""Simple registry to keep model builders discoverable."""

from __future__ import annotations

from typing import Any, Callable, Dict

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a model builder."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a model builder by name."""
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Model '{name}' is not registered.") from exc
    return builder(**kwargs)
