"""Model registry and factory functions for supported detector backbones."""

from .registry import MODEL_REGISTRY, register_model, build_model

__all__ = ["MODEL_REGISTRY", "register_model", "build_model"]
