"""Smoke tests for configuration loader."""

from pathlib import Path

import pytest

from src.uaelp_det.utils import load_config


def test_missing_yaml_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure helpful error when PyYAML is unavailable."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("key: value\n", encoding="utf-8")

    monkeypatch.setitem(load_config.__globals__, "yaml", None)

    with pytest.raises(ImportError):
        load_config(yaml_file)
