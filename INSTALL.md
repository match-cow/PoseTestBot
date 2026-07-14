# PoseTestBot Installation

PoseTestBot is acquisition-first: the repository captures, calibrates,
synchronizes, optionally prepares ground-truth/masks, and exports BOP datasets.
Use `uv` for Python environment management. PoseTestBot targets only the
physical lab iiwa; normal setup and validation never execute capture automatically.

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
- lists registered sensor adapters without opening hardware,
- verifies that the self-contained operator-console build is bundled.

The required Python environment also includes `aiortc`, `aiohttp`, and direct
`av` support for the UGREEN room monitor's video-only WebRTC stream. The
worker binds its offer/answer signaling service to an ephemeral loopback port;
the Flask API proxies signaling while browsers exchange media directly over
the trusted lab LAN. No STUN or TURN service is configured.

If `UV_CACHE_DIR` is unset, the installer uses `/tmp/uv-cache`.
Browser binaries for Playwright UI tests are not installed by default.
Bun is not required for normal Python installation or runtime because the
locked production build is committed and packaged in the wheel.

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

Use `--with-web-build` only when changing the React/shadcn frontend. It requires
Bun, installs exactly the versions in `frontend/bun.lock`, removes stale build
output, and regenerates the bundled Flask assets. The Cell bundle includes
Three.js, React Three Fiber, and Drei; installed operation still requires
neither Bun nor a network connection:

```bash
bash scripts/install.sh --with-web-build
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
uv run posetestbot-web
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
Explicit objectless render plans also skip BlenderProc completely.

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

The room-monitor coverage in that file uses an in-process synthetic aiortc
video track. It does not open the UGREEN camera or any RGB-D acquisition
device.

### UGREEN Room Monitor

The UGREEN USB camera (`0c45:2283`) is reserved by the queued room-monitor
worker as `monitoring_camera:0c45:2283`. It requests MJPEG 640×480 at 30 fps
with a one-frame V4L2 buffer, publishes VP8-preferred WebRTC video, and has no
JPEG fallback. The generic RGB-D sensor preview controls remain JPEG-based.

Starting or retrying the monitor from the dashboard is safe with respect to
the robot: it queues only the monitor worker and never runs an acquisition
pipeline or robot command. A physical monitor smoke test still requires
explicit operator authorization because it opens the USB camera.

### Operator Console Development

The frontend lives in `frontend/` and follows the shadcn Vite layout with
React, TypeScript, Tailwind, Radix primitives, HashRouter, TanStack Query,
React Hook Form, Zod, Three.js, React Three Fiber, and Drei. Its production output is
`posetestbot/web/static/ui/`.

```bash
cd frontend
bun install --frozen-lockfile
bun run typecheck
bun run lint
bun run build
```

Run the Flask server in another terminal when using `bun run dev`; Vite proxies
the existing API routes to `127.0.0.1:5000`. Never point browser tests at lab
hardware: use the mocked Playwright fixtures.

## Real Robot Profile

Robot status is read-only:

```bash
uv run python scripts/robot_status.py --json
```

Create and inspect a physical capture plan without executing it:

```bash
uv run python scripts/create_run_config.py working_data/test_run
uv run python scripts/run_pipeline_sequence.py working_data/test_run \
  --sequence real_full_capture_validation --plan-only
```

## Validation

Recommended local validation:

```bash
bash -n scripts/install.sh
bash scripts/install.sh --help
bash scripts/install.sh --check-only
cd frontend && bun run typecheck && bun run lint && bun run build
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runtime_status.py tests/test_hardware_status.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
UV_CACHE_DIR=/tmp/uv-cache uv build
git diff --check
```

## Troubleshooting

- `uv` missing: run `bash scripts/install.sh` without `--check-only`, or install
  `uv` manually and rerun `uv sync --all-groups`.
- Python import smoke fails: rerun `uv sync --all-groups`; add or update
  dependencies with `uv add ...` rather than hand-editing lock files.
- Room-monitor signaling is unavailable: confirm the queued
  `monitor-webrtc:ugreen` job is running, the `monitor_webrtc.v1` status says
  `signaling_ready`, and local firewall rules allow WebRTC host-candidate
  traffic on the trusted lab LAN. The loopback signaling port is intentionally
  not exposed to browsers.
- Plan the isolated UGREEN hardware smoke without opening the camera with
  `uv run python scripts/run_monitor_webrtc_smoke.py --plan-only`. Physical
  execution additionally requires explicit operator authorization and all
  three command acknowledgements: `--operator-authorized --allow-cameras
  --allow-real-robot`. Despite the shared lab safety gate, this monitor-only
  command contains no robot or acquisition-pipeline action.
- `pyzed.sl` missing: install the Stereolabs ZED SDK and Python bindings outside
  uv, then rerun `uv run python scripts/runtime_status.py --json`.
- Camera SDK imports succeed but devices are missing: check USB cabling, power,
  device permissions, and vendor udev rules on the lab host.
- BlenderProc missing: install it with `bash scripts/install.sh --with-blenderproc`
  or keep using dry-run render planning.
- Playwright reports a missing Chromium executable: run
  `UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium`, or use
  `bash scripts/install.sh --with-playwright-browsers`.
- Bundled web assets are missing: restore the committed
  `posetestbot/web/static/ui/` files or run
  `bash scripts/install.sh --with-web-build` on a machine with Bun.
- Real robot commands require deliberate `--allow-real-robot` and camera
  execution requires `--allow-cameras`.
