"""Stable filesystem anchors for source-checkout and installed web use."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_app_root() -> Path:
    explicit = os.environ.get("POSETESTBOT_APP_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / "pyproject.toml").is_file():
        return source_candidate
    return Path.cwd().resolve()


APP_ROOT = resolve_app_root()
