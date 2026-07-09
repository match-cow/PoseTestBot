# PoseTestBot Installation

PoseTestBot is acquisition-first: the repository captures, calibrates,
synchronizes, optionally prepares ground-truth/masks, and exports BOP datasets.
Use `uv` for Python environment management and keep the default robot path fake
unless you intentionally target the physical iiwa.

## Quick Setup

From the repository root:

```bash
bash scripts/install.sh
```

This safe project bootstrap:

- ensures `uv` is available,
- runs `uv sync --all-groups`,
- checks required Python imports,
- checks optional acquisition runtimes,
- lists registered sensor adapters without opening hardware.

If `UV_CACHE_DIR` is unset, the installer uses `/tmp/uv-cache`.
Browser binaries for Playwright UI tests are not installed by default.

Use check-only mode to inspect an already configured environment without
installing or syncing:

```bash
bash scripts/install.sh --check-only
```

## Lab Host Setup

For an Ubuntu lab host where system packages and optional BlenderProc should be
installed by the script:

```bash
bash scripts/install.sh --with-system-packages --with-blenderproc
```

`--with-system-packages` installs common Ubuntu packages for local development,
USB inspection, OpenCV runtime support, and build tooling. It does not install
vendor camera SDKs or proprietary packages.

`--with-blenderproc` installs BlenderProc as a `uv` tool when the
`blenderproc` executable is missing.

`--with-playwright-browsers` installs Chromium for Playwright browser UI tests
after the uv environment has been synchronized. Keep this opt-in on lab hosts
unless you are actively running browser coverage:

```bash
bash scripts/install.sh --with-playwright-browsers
```

## Manual Prerequisites

### uv

The installer can install `uv` through the official Astral installer when it is
missing. To install it manually, follow Astral's current uv installation
instructions, then verify:

```bash
uv --version
uv sync --all-groups
```

Run project scripts through `uv`:

```bash
uv run python scripts/robot_status.py --json
```

### RealSense D435

The Python package `pyrealsense2` is declared in `pyproject.toml` and installed
by `uv sync`. Physical RealSense discovery may still require USB access and
RealSense udev rules on the lab host.

Check visibility:

```bash
uv run python scripts/sensor_status.py --expected realsense_d435=3 --check-expected
```

### OAK-D Pro

DepthAI v3 is declared in `pyproject.toml` and installed by `uv sync`. Physical
OAK-D Pro discovery may still require USB access and udev permissions.

Check visibility:

```bash
uv run python scripts/sensor_status.py --expected oak_d_pro=1 --check-expected
```

### ZED 2i

The Stereolabs ZED SDK and `pyzed.sl` Python module are not ordinary PyPI
dependencies, so they are not installed by `uv sync` or by `scripts/install.sh`.
Install them with Stereolabs' SDK installer for the lab host and Python version,
then verify:

```bash
uv run python scripts/runtime_status.py --json
uv run python scripts/sensor_status.py --expected zed_2i=1 --check-expected
```

### BlenderProc

BlenderProc is only needed for non-dry-run optional GT/mask rendering. Dry-run
render planning and ordinary acquisition checks do not require it.

Install through the PoseTestBot installer:

```bash
bash scripts/install.sh --with-blenderproc
```

Verify:

```bash
uv run python scripts/runtime_status.py --json
```

### Playwright Browser Tests

The Python Playwright package is a dev dependency installed by
`uv sync --all-groups`, but browser binaries are intentionally optional. Install
Chromium only when running browser UI coverage:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium
```

Then run the sensor preview browser test:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
```

## Fake Robot Default

The default robot profile is fake. Use it for setup and hardware-free smoke
tests:

```bash
uv run python iiwa/fake_iiwa_controller.py --receiver-ip 127.0.0.1
uv run python scripts/pose_receiver_udp_json.py /tmp/posetestbot_fake_run --test
```

Use the real robot only intentionally:

```bash
POSETESTBOT_ROBOT_MODE=real uv run python scripts/pose_receiver_udp_json.py working_data/test_run
```

## Validation

Recommended local validation:

```bash
bash -n scripts/install.sh
bash scripts/install.sh --help
bash scripts/install.sh --check-only
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runtime_status.py tests/test_hardware_status.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
git diff --check
```

Hardware-free acquisition-to-BOP smoke:

```bash
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_fake_bop_smoke --overwrite
uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_fake_bop_smoke \
  --gate rewrite_fake_acquisition_to_bop.v1 --write
```

## Troubleshooting

- `uv` missing: run `bash scripts/install.sh` without `--check-only`, or install
  `uv` manually and rerun `uv sync --all-groups`.
- Python import smoke fails: rerun `uv sync --all-groups`; add or update
  dependencies with `uv add ...` rather than hand-editing lock files.
- `pyzed.sl` missing: install the Stereolabs ZED SDK and Python bindings outside
  uv, then rerun `uv run python scripts/runtime_status.py --json`.
- Camera SDK imports succeed but devices are missing: check USB cabling, power,
  device permissions, and vendor udev rules on the lab host.
- BlenderProc missing: install it with `bash scripts/install.sh --with-blenderproc`
  or keep using dry-run render planning.
- Playwright reports a missing Chromium executable: run
  `UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium`, or use
  `bash scripts/install.sh --with-playwright-browsers`.
- Real robot commands should only be run with deliberate real-mode selection.
