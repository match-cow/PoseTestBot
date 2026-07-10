"""JSON-friendly status snapshots for external PoseTestBot runtimes."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

SCHEMA_VERSION = "runtime_status.v1"


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    ok: bool
    value: str | bool | None = None
    hint: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeStatus:
    runtime_id: str
    display_name: str
    category: str
    required_for: str
    available: bool
    checks: list[RuntimeCheck]
    hint: str | None = None

    def as_dict(self) -> dict:
        value = asdict(self)
        value["checks"] = [check.as_dict() for check in self.checks]
        return value


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _truthy_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser()


def _path_check(name: str, path: Path | None, *, hint: str) -> RuntimeCheck:
    exists = bool(path and path.exists())
    return RuntimeCheck(
        name=name,
        ok=exists,
        value=path.as_posix() if path else None,
        hint=None if exists else hint,
    )


def _which_check(
    name: str,
    executable: str,
    *,
    which: Callable[[str], str | None],
    hint: str,
) -> RuntimeCheck:
    path = which(executable)
    return RuntimeCheck(
        name=name,
        ok=path is not None,
        value=path,
        hint=None if path else hint,
    )


def _module_check(module_name: str, *, hint: str) -> RuntimeCheck:
    available = module_available(module_name)
    return RuntimeCheck(
        name=f"module:{module_name}",
        ok=available,
        value=available,
        hint=None if available else hint,
    )


def _runtime(
    *,
    runtime_id: str,
    display_name: str,
    category: str,
    required_for: str,
    checks: list[RuntimeCheck],
    hint: str | None = None,
) -> RuntimeStatus:
    return RuntimeStatus(
        runtime_id=runtime_id,
        display_name=display_name,
        category=category,
        required_for=required_for,
        available=all(check.ok for check in checks),
        checks=checks,
        hint=hint,
    )


def blenderproc_status(
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> RuntimeStatus:
    return _runtime(
        runtime_id="blenderproc",
        display_name="BlenderProc",
        category="renderer",
        required_for="BlenderProc ground-truth rendering",
        checks=[
            _which_check(
                "executable:blenderproc",
                "blenderproc",
                which=which,
                hint="Install BlenderProc or expose the blenderproc executable on PATH.",
            )
        ],
        hint="Dry-run render planning does not require BlenderProc, execution does.",
    )


def zed_sdk_status() -> RuntimeStatus:
    return _runtime(
        runtime_id="zed_sdk_python",
        display_name="Stereolabs ZED SDK Python",
        category="camera_sdk",
        required_for="ZED 2i capture",
        checks=[
            _module_check(
                "pyzed.sl",
                hint=(
                    "Install the Stereolabs ZED SDK and Python module outside "
                    "ordinary PyPI/uv dependency resolution."
                ),
            )
        ],
        hint="The ZED SDK Python module is provided by Stereolabs, not PyPI.",
    )


def collect_runtime_status(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> dict:
    runtimes = [
        blenderproc_status(which=which),
        zed_sdk_status(),
    ]
    available_count = sum(1 for runtime in runtimes if runtime.available)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available_count": available_count,
        "runtime_count": len(runtimes),
        "all_available": available_count == len(runtimes),
        "runtimes": [runtime.as_dict() for runtime in runtimes],
    }
