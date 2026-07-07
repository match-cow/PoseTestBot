"""JSON-friendly status snapshots for external PoseTestBot runtimes."""

from __future__ import annotations

import importlib.util
import os
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


def foundationpose_status(
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> RuntimeStatus:
    env = env or os.environ
    home = home or Path.home()
    configured_root = _truthy_path(env.get("FOUNDATIONPOSE_ROOT"))
    checkout = configured_root or home / "FoundationPose"
    run_demo = checkout / "run_demo.py"
    run_demo_no_tracking = checkout / "run_demo_no_tracking.py"
    return _runtime(
        runtime_id="foundationpose",
        display_name="FoundationPose",
        category="estimator",
        required_for="FoundationPose estimator execution",
        checks=[
            _which_check(
                "executable:docker",
                "docker",
                which=which,
                hint="Install Docker and make it available on PATH.",
            ),
            _path_check(
                "checkout",
                checkout,
                hint=(
                    "Set FOUNDATIONPOSE_ROOT or clone FoundationPose at "
                    "~/FoundationPose."
                ),
            ),
            _path_check(
                "run_demo.py",
                run_demo,
                hint="Expected FoundationPose run_demo.py in the checkout root.",
            ),
            _path_check(
                "run_demo_no_tracking.py",
                run_demo_no_tracking,
                hint=(
                    "Expected FoundationPose run_demo_no_tracking.py in the "
                    "checkout root."
                ),
            ),
        ],
        hint=(
            "The status check does not start the foundationpose Docker container; "
            "it only checks local prerequisites."
        ),
    )


def bop_toolkit_status(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> RuntimeStatus:
    env = env or os.environ
    cwd = cwd or Path.cwd()
    configured_root = _truthy_path(env.get("BOP_TOOLKIT_ROOT"))
    root = configured_root or cwd / "bop_toolkit"
    eval_script = root / "scripts" / "eval_bop19_pose.py"
    return _runtime(
        runtime_id="bop_toolkit",
        display_name="BOP Toolkit",
        category="evaluation",
        required_for="BOP Toolkit pose evaluation",
        checks=[
            _path_check(
                "checkout",
                root,
                hint="Set BOP_TOOLKIT_ROOT or clone bop_toolkit under the repo.",
            ),
            _path_check(
                "eval_bop19_pose.py",
                eval_script,
                hint="Expected scripts/eval_bop19_pose.py in the BOP Toolkit root.",
            ),
        ],
        hint="Dry-run BOP evaluation planning does not require the checkout.",
    )


def _configured_wrapper(
    *,
    env: Mapping[str, str],
    cwd: Path,
    env_var: str,
    default_relative_path: str,
) -> Path:
    configured = _truthy_path(env.get(env_var))
    if configured is not None:
        return configured
    return cwd / default_relative_path


def megapose_status(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> RuntimeStatus:
    env = env or os.environ
    cwd = cwd or Path.cwd()
    wrapper = _configured_wrapper(
        env=env,
        cwd=cwd,
        env_var="MEGAPOSE_WRAPPER",
        default_relative_path="scripts/megapose_wrapper.py",
    )
    return _runtime(
        runtime_id="megapose",
        display_name="MegaPose",
        category="estimator",
        required_for="MegaPose estimator execution",
        checks=[
            _path_check(
                "wrapper_script",
                wrapper,
                hint=(
                    "Set MEGAPOSE_WRAPPER to the installed wrapper script or "
                    "provide scripts/megapose_wrapper.py."
                ),
            )
        ],
        hint="Dry-run MegaPose planning does not require the wrapper script.",
    )


def sam6d_status(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> RuntimeStatus:
    env = env or os.environ
    cwd = cwd or Path.cwd()
    wrapper = _configured_wrapper(
        env=env,
        cwd=cwd,
        env_var="SAM6D_WRAPPER",
        default_relative_path="scripts/sam6d_wrapper.py",
    )
    return _runtime(
        runtime_id="sam6d",
        display_name="SAM6D",
        category="estimator",
        required_for="SAM6D estimator execution",
        checks=[
            _path_check(
                "wrapper_script",
                wrapper,
                hint=(
                    "Set SAM6D_WRAPPER to the installed wrapper script or "
                    "provide scripts/sam6d_wrapper.py."
                ),
            )
        ],
        hint="Dry-run SAM6D planning does not require the wrapper script.",
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
        foundationpose_status(env=env, home=home, which=which),
        megapose_status(env=env, cwd=cwd),
        sam6d_status(env=env, cwd=cwd),
        bop_toolkit_status(env=env, cwd=cwd),
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
