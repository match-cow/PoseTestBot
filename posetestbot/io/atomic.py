"""Crash-resistant helpers for replace-in-place text and JSON artifacts."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text through a same-directory temporary file and atomically replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    default: Any = None,
) -> Path:
    """Serialize JSON without allowing non-standard NaN/Infinity values."""

    text = json.dumps(
        value,
        indent=indent,
        sort_keys=sort_keys,
        default=default,
        allow_nan=False,
    )
    return atomic_write_text(path, f"{text}\n")


def replace_directory(staging: str | Path, destination: str | Path) -> Path:
    """Promote a complete sibling directory while preserving the old one on error."""

    source = Path(staging)
    target = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"Staging directory does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
    moved_existing = False
    try:
        if target.exists():
            os.replace(target, backup)
            moved_existing = True
        os.replace(source, target)
    except Exception:
        if moved_existing and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    return target


def replace_directories(
    promotions: Iterable[tuple[str | Path, str | Path]],
) -> list[Path]:
    """Promote several sibling directories as one rollback-capable operation."""

    pairs = [(Path(source), Path(target)) for source, target in promotions]
    if not pairs:
        return []
    sources = [source.resolve() for source, _target in pairs]
    targets = [target.resolve() for _source, target in pairs]
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise ValueError("Directory promotion sources and destinations must be unique")
    for source, target in pairs:
        if not source.is_dir():
            raise FileNotFoundError(f"Staging directory does not exist: {source}")
        if source.parent.resolve() != target.parent.resolve():
            raise ValueError(
                f"Staging directory must be a sibling of its destination: {source}, {target}"
            )

    backups = [
        target.with_name(f".{target.name}.{uuid.uuid4().hex}.bak")
        for _source, target in pairs
    ]
    moved_existing: list[int] = []
    promoted: list[int] = []
    try:
        for index, ((_source, target), backup) in enumerate(zip(pairs, backups)):
            if target.exists():
                os.replace(target, backup)
                moved_existing.append(index)
        for index, (source, target) in enumerate(pairs):
            os.replace(source, target)
            promoted.append(index)
    except Exception:
        for index in reversed(promoted):
            source, target = pairs[index]
            if target.exists() and not source.exists():
                os.replace(target, source)
        for index in reversed(moved_existing):
            _source, target = pairs[index]
            backup = backups[index]
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups:
            if backup.exists():
                shutil.rmtree(backup)
    return [target for _source, target in pairs]
