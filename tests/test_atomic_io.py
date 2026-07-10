from __future__ import annotations

import json
from pathlib import Path

import pytest

from posetestbot.io import atomic


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"old":true}\n')

    result = atomic.atomic_write_json(path, {"new": [1, 2, 3]})

    assert result == path
    assert json.loads(path.read_text()) == {"new": [1, 2, 3]}
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_write_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("original\n")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic.atomic_write_text(path, "replacement\n")

    assert path.read_text() == "original\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_json_rejects_nonstandard_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        atomic.atomic_write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_replace_directories_promotes_complete_batch(tmp_path: Path) -> None:
    destinations = [tmp_path / "one", tmp_path / "two"]
    stagings = [tmp_path / ".one.stage", tmp_path / ".two.stage"]
    for index, destination in enumerate(destinations):
        destination.mkdir()
        (destination / "old.txt").write_text(str(index))
        stagings[index].mkdir()
        (stagings[index] / "new.txt").write_text(str(index))

    assert atomic.replace_directories(zip(stagings, destinations)) == destinations

    for index, destination in enumerate(destinations):
        assert (destination / "new.txt").read_text() == str(index)
        assert not (destination / "old.txt").exists()
