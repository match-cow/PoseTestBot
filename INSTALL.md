# PoseTestBot Installation

PoseTestBot is acquisition-first: the repository captures, calibrates,
synchronizes, optionally prepares ground-truth/masks, and exports BOP datasets.
Use Python 3.12 and `uv` for Python environment management. The project requires
`>=3.12,<3.13`; `uv` installs the matching interpreter when necessary. PoseTestBot targets only the
physical lab iiwa; normal setup and validation never execute capture automatically.

## Quick Setup

From the repository root:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
```

This safe project bootstrap:

- ensures `uv` is available,
- runs `uv sync --all-groups`,
- initializes and verifies the exact, clean PoseGridGen source submodule,
- initializes and verifies the exact, clean PoseTemplateCreator source submodule,
- checks required Python imports,
- checks optional acquisition runtimes,
- lists registered sensor adapters without opening hardware,
- verifies that the self-contained operator-console build is bundled.

The required Python environment also includes Matplotlib for reproducible,
headless calibration teaching plots plus `aiortc`, `aiohttp`, `aioice`, and
direct `av` support for the UGREEN room monitor's video-only WebRTC stream.
The worker binds offer/answer signaling to an ephemeral loopback port and runs
a local STUN binding responder on UDP port 3478. The Flask API proxies only
signaling; browsers use the advertised STUN port to obtain numeric candidates
and exchange media directly over the trusted lab LAN. Set
`POSETESTBOT_MONITOR_STUN_PORT` to use a different UDP port. TURN and Internet
NAT traversal remain out of scope.

If `UV_CACHE_DIR` is unset, the installer uses `/tmp/uv-cache`.
Browser binaries for Playwright UI tests are not installed by default.
Bun is not required for normal Python installation or runtime because the
locked production build is committed and packaged in the wheel.

Omit `--with-posegridgen` when this checkout only needs to consume existing
`calibration_target.v1/v2` files. The Calibration Targets generator is then
reported unavailable, while calibration readers and the bundled UI continue
to work.

Omit `--with-posetemplatecreator` when this checkout only needs to browse,
classify, archive, restore, export, or metadata-import existing Workpiece
Catalogue entries and to read immutable pose-template bundles and run
selections. Client-side catalogue PLY previews continue to work. New CAD
inspection/conversion, exact slicing, pose-template preview, and PDF generation
are disabled until the pinned checkout is available.

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
The installed `v4l-utils` command is also used by the managed UGREEN monitor to
discover the camera's brightness range before a browser-requested automatic
brightness calibration; calibration remains unavailable if that control cannot
be inspected.

`--with-blenderproc` installs the validated BlenderProc 2.8.0 as a `uv` tool
when the `blenderproc` executable is missing.

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

### PoseGridGen Calibration Targets

Printable target generation is source-checkout-only. Initialize the committed
submodule at `third_party/PoseGridGen` and verify the pinned revision through
the installer:

```bash
bash scripts/install.sh --with-posegridgen
bash scripts/install.sh --check-only \
  --with-posegridgen --with-posetemplatecreator
```

The required revision is
`ad152e369e8d2746d0cf66cb1455f2371b0ec0f0`. Generation is disabled if the
checkout is missing, dirty, at another revision, lacks the required backend
files, or cannot provide the renderer/OpenCV capabilities. PoseTestBot loads
only PoseGridGen's backend models, errors, fitting, scene, and rendering modules
under a private namespace; FastAPI and Uvicorn are not runtime dependencies.

Use the operator console's **Calibration Targets** page to preview, fit, and
generate immutable source/spec/PDF bundles, then select one for a configured
run. The complete artifact and placement contract is documented in
[`docs/POSEGRIDGEN_CALIBRATION_TARGETS.md`](docs/POSEGRIDGEN_CALIBRATION_TARGETS.md).

### Workpiece Catalogue and PoseTemplateCreator Object Ground Truth

The bundled **Workpiece Catalogue** page stores the test-object manifest and
UUID-addressed CAD assets below `working_data/object_catalog/`. Existing
entries, editable metadata, archive/restore controls, JSON metadata
import/export, bounded isometric thumbnails, and the selected object's single
interactive bounded 3D preview do not require an additional service or
database.

Importing a new PLY/STL/OBJ (with an optional PNG texture), inspecting and
converting its mesh, and generating printable pose templates use the source
checkout at `third_party/PoseTemplateCreator`:

```bash
bash scripts/install.sh --with-posetemplatecreator
bash scripts/install.sh --check-only --with-posetemplatecreator
```

The required revision is
`97ddb9b7b756912deb8c2d2d6dde186b461e5d9d`. PoseTestBot refuses generation
when the checkout is missing, dirty, or at another revision. It privately loads
only the upstream constants, models, secure mesh parser, stable-orientation and
bounded-preview extraction, exact contour slicer, scene, and PDF renderer; the
upstream FastAPI server and React application are never imported or embedded.
The Python environment includes NetworkX for the pinned Trimesh stable-pose
path. Existing immutable bundles remain readable when the optional checkout is
unavailable. Catalogue storage, lifecycle, metre/mm geometry revisions, API,
and metadata-portability details are documented in
[`docs/WORKPIECE_CATALOGUE.md`](docs/WORKPIECE_CATALOGUE.md). The printable
template and object-GT workflow is documented in
[`docs/POSETEMPLATECREATOR_OBJECT_GT.md`](docs/POSETEMPLATECREATOR_OBJECT_GT.md).

### RealSense D435

The Python package `pyrealsense2` is declared in `pyproject.toml` and installed
by `uv sync`. Physical RealSense discovery may still require USB access and
RealSense udev rules on the lab host.

Check visibility:

```bash
uv run python scripts/sensor_status.py --json
```

For the separate three-RealSense service/full-capture maintenance milestone,
require all three SDK-addressable devices explicitly:

```bash
uv run python scripts/sensor_status.py --expected realsense_d435=3 --check-expected
```

The expected count is based on cameras addressable through `pyrealsense2`.
RealSense devices seen only through USB descriptors remain in the status output
for troubleshooting, but do not pass capture-readiness checks. SDK-enumerated
cameras with a known `usb_type_descriptor` below USB 3 also fail readiness and
capture-plan preflight. Older status records that do not contain transport
metadata remain readable; a fresh status check is required before real capture.

Before capture, require every enabled/selected serial to be SDK-addressable and
to report a 3.x-or-newer descriptor when the transport version is known. All
three configured serials are required only for the separate three-camera
service/full-capture milestone; a disabled serial remains recorded but is
excluded from the current run. A USB2 fallback can be caused by a
marginal/non-SuperSpeed cable, port, connector, hub power, or an overcommitted
USB controller. Reseat or power-cycle only the affected USB connection without
moving its camera mount, use known-good SuperSpeed paths, and rerun the status
command. `lsusb -t` is useful read-only topology evidence, but the SDK
descriptor and successful stream warmup remain the capture gates.

When supported by the installed SDK and camera, status also records
`firmware_version` and the SDK's `recommended_firmware_version`. A numeric
difference produces a troubleshooting warning only; it does not weaken USB
readiness or prove that firmware caused a transport failure. PoseTestBot never
flashes camera firmware. Any persistent firmware change requires a separately
reviewed maintenance procedure and explicit device-specific authorization.

The **Devices** page labels each camera **Capture-ready**, **Not
capture-ready**, or **Disconnected** and shows the readiness reason. A camera
that is not ready cannot start a preview or snapshot or be newly selected for a
run; if it was already selected, it can still be deselected. In **Workflow →
Run Setup**, the **Enabled for capture and calibration** checkbox retains a
disabled camera's identity and metadata while excluding it from work. Keep at
least one camera enabled, then regenerate capture-plan and preflight artifacts
after any enable/disable change.

#### Combined static and robot-mounted D435 triggering

`run_config.v3` supports one explicit mixed-mount hardware mode:
`capture_synchronization.v1` with `mode=hardware_trigger`,
`implementation=realsense_inter_cam_sync`, and `scope=depth_exposure`. Every
enabled camera in such a run must be an exact-ID D435. The group needs at least
one `static` and one `eye_in_hand` camera, exactly one master, and one or more
subordinates. The normal `timestamp_aligned` mode remains available for all
other sensor combinations.

The Python environment and installer need no extra package for this mode, but
the physical sync harness is an operator-qualified hardware prerequisite.
Build and inspect it against Intel's D400 multi-camera guidance, including the
documented sync/ground pins and electrical limits:

https://dev.realsenseai.com/docs/multiple-depth-cameras-configuration/

Before research acquisition, retain evidence that the exact mounted devices
and harness pass continuity/electrical inspection, master/subordinate SDK
configuration and read-back, global-time metadata checks, and a shared-view
pulsed LED visible in the depth/IR stream, or an equivalent exposure-timing
qualification, at the intended frame rate. Plan-only validation cannot
establish any of those physical facts.
Record the passed external evidence without opening hardware:

```bash
uv run python scripts/record_hardware_sync_qualification.py \
  working_data/combined_run \
  --operator OPERATOR_ID \
  --method pulsed_light \
  --observed-max-depth-timestamp-skew-ms 0.8 \
  --evidence path/to/pulse-trace.csv \
  --confirm-passed
```

The resulting `hardware_sync_qualification.json` is hash-bound to the exact
resolution, FPS, synchronization policy, camera IDs, mounts, and roles.
Preflight and later dataset stages reject missing, tampered, copied-to-another-
run, or stale evidence. The recorder copies files only and never accesses a
camera or robot. Publish it before acquisition. After any capture status,
report, log, raw camera data, or raw robot-pose evidence exists, qualification
publication and replacement are blocked; use a new run root for another
qualification or contract. Treat both the configured and observed maximum skew
as the full earliest-to-latest depth timestamp span across every camera in a
complete group.
No physical hardware-sync capture was performed as part of the software
implementation.

The supported claim is deliberately **depth exposure only**. D435 color uses a
rolling shutter and is not certified by this contract as the same RGB exposure
across cameras. PoseTestBot records the associated RGB frame and its limitation
instead of upgrading the claim. Do not directly connect or add the current USB
OAK-D Pro or USB ZED 2i to this trigger group: their present lab interfaces
cannot participate in the supported D435 inter-camera synchronization
contract.

During supervised capture, camera metadata has a liveness deadline independent
of the robot UDP first/inter-packet timeouts. Its default is 12 planned frame
periods clamped to 2–5 seconds; an explicit override cannot exceed five
seconds. A stalled, truncated, rewritten, or non-monotonic stream aborts local
capture while preserving all raw evidence. The successful full-capture report
binds the exact configuration and qualification hashes after immediate
pre-receiver revalidation. Authoritative complete groups and BOP frame sets
carry that binding, and `rewrite_bop_export_readiness.v1` rejects disagreement
with the current qualification or capture report.

### OAK-D Pro

DepthAI v3 is declared in `pyproject.toml` and installed by `uv sync`. Physical
OAK-D Pro discovery may still require USB access and udev permissions.

Check visibility:

```bash
uv run python scripts/sensor_status.py --expected oak_d_pro=1 --check-expected
```

The operator console supports an OAK-D Pro RGB preview at 640×480/6 fps. It
uses a non-blocking DepthAI v3 queue with a single latest frame. The Snapshot
control remains a one-frame aligned 1280×720 RGB-D acquisition, matching the
RealSense snapshot contract.

The current PoseTestBot contract does not implement or qualify the USB OAK-D
Pro as a member of the supported D435 group, so it is accepted only in
`timestamp_aligned` runs. Luxonis documents external frame-sync connectors on
other hardware, such as the PoE model, but that is a different electrical
interface and is not interchangeable with the implemented D435 sync contract:

https://docs.luxonis.com/hardware/products/OAK-D%20Pro%20PoE

### ZED 2i

The Stereolabs ZED SDK and `pyzed.sl` Python module are not ordinary PyPI
dependencies, so they are not installed by `uv sync` or by `scripts/install.sh`.
Install them with Stereolabs' SDK installer for the lab host and Python version,
then verify:

```bash
uv run python scripts/runtime_status.py --json
uv run python scripts/sensor_status.py --expected zed_2i=1 --check-expected
```

The current USB ZED 2i is likewise accepted only in `timestamp_aligned` runs.
Stereolabs documents hardware triggering for ZED X/GMSL through ZED Link, not
for the lab's USB ZED 2i:

https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera#frame-synchronization

### BlenderProc

BlenderProc is only needed for non-dry-run optional GT/mask rendering. Dry-run
render planning and ordinary acquisition checks do not require it.
Explicit objectless render plans also skip BlenderProc completely.
Pose-template duplicate-instance GT is validated with BlenderProc 2.8.0. The
renderer rejects other versions before producing derived GT evidence.

Install through the PoseTestBot installer:

```bash
bash scripts/install.sh --with-blenderproc
```

Verify:

```bash
uv run python scripts/runtime_status.py --json
```

### Calibration Teaching Plot

Matplotlib is a direct project dependency installed by `uv sync`. Regenerating
the committed iiwa Workbench teaching SVG and PNG is headless and does not open
the robot or cameras:

```bash
MPLCONFIGDIR=/tmp/posetestbot-mpl UV_CACHE_DIR=/tmp/uv-cache \
  uv run python scripts/plot_iiwa_calibration_teaching_plan.py
```

The script validates `iiwa/calibration_teaching_plan.v2.json` and writes
`docs/images/iiwa_calibration_teaching_plan.svg` plus the corresponding PNG.
The procedure and printable sign-off sheet are linked from
`docs/IIWA_CALIBRATION_VARIANCE_PROPOSAL.md`.

### Playwright Browser Tests

The Python Playwright package is a dev dependency installed by
`uv sync --all-groups`, but browser binaries are intentionally optional. Install
Chromium only when running browser UI coverage:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium
```

Then run the operator-console and sensor-preview browser tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
```

The default `uv run pytest` selection excludes these marked browser modules, so
ordinary tests do not require the optional Chromium download.

The room-monitor coverage in that file uses an in-process synthetic aiortc
video track. It does not open the UGREEN camera or any RGB-D acquisition
device.

### UGREEN Room Monitor

The UGREEN USB camera (`0c45:2283`) is owned by a hidden managed service using
the resource `monitoring_camera:0c45:2283`. The service starts lazily and does
not open the V4L2 node until a browser requests WebRTC media. It requests MJPEG
640×480 at 30 fps with a one-frame V4L2 buffer, publishes VP8-preferred WebRTC
video, and releases the camera after 15 seconds without a connected peer. VP8
payloads are capped at 1100 bytes so the complete RTP datagram remains below
the 1280-byte Tailscale interface MTU after transport overhead. It has no JPEG
fallback. The generic RGB-D sensor preview controls remain latest-frame JPEG
streams.

Managed services are excluded from the normal Jobs list and held-resource
banner. Use `GET /jobs?include_services=1` for diagnostics. Monitor health is
persisted as `monitor_webrtc.v2` with one-second heartbeats, camera state,
capture/media counters, frame timestamps, peer counts, STUN port, and a
concrete failure reason. The status also records `vp8_packet_max_bytes` for
transport diagnostics. Legacy v1 artifacts remain readable but are never reused
as live state.

Starting or retrying the monitor from the dashboard is safe with respect to
the robot: it queues only the monitor worker and never runs an acquisition
pipeline or robot command. A physical monitor smoke test still requires
explicit operator authorization because it opens the USB camera.

All commands queued by any supported web entry point run behind a persisted
process supervisor. On graceful web shutdown, workload groups receive SIGTERM
and have five seconds to exit before SIGKILL. On a forced web-app SIGKILL,
Linux parent-death signaling wakes each supervisor, which verifies the owner
PID/start time and terminates the complete workload descendant group.

### Operator Console Development

The frontend lives in `frontend/` and follows the shadcn Vite layout with
React, TypeScript, Tailwind, Radix primitives, HashRouter, TanStack Query,
React Hook Form, Zod, Three.js, React Three Fiber, and Drei. Its production
output is `posetestbot/web/static/ui/`. The selected Workpiece Catalogue detail
loads the exact canonical PLY with those bundled client-side Three.js
dependencies. Compact catalogue cards use bounded, topology-scored recognition
meshes generated with the pinned `fast-simplification==0.1.13` runtime
dependency; the pose-template editor retains its smaller interaction-oriented
preview tier. No server rendering service or database is required.

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
bash scripts/install.sh --check-only \
  --with-posegridgen --with-posetemplatecreator
cd frontend && bun run typecheck && bun run lint && bun run build
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runtime_status.py tests/test_hardware_status.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
UV_CACHE_DIR=/tmp/uv-cache uv build
git diff --check
```

## Troubleshooting

- `uv` missing: run `bash scripts/install.sh` without `--check-only`, or install
  `uv` manually and rerun `uv sync --all-groups`.
- Python import smoke fails: rerun `uv sync --all-groups`; add or update
  dependencies with `uv add ...` rather than hand-editing lock files.
- Calibration target generation is unavailable: run
  `git submodule update --init --checkout third_party/PoseGridGen`, confirm the
  checkout is clean at the pinned revision, then run
  `bash scripts/install.sh --check-only --with-posegridgen`.
- Pose-template inspection or generation is unavailable: run
  `git submodule update --init --checkout third_party/PoseTemplateCreator`,
  confirm the checkout is clean at the pinned revision, then run
  `bash scripts/install.sh --check-only --with-posetemplatecreator`. Existing
  Workpiece Catalogue records/assets, immutable bundles, and run selections
  remain readable without the source checkout; only new CAD conversion and
  pose-template geometry/rendering actions are unavailable.
- Workpiece catalogue metadata import reports `skipped_missing_assets`: this is
  expected when matching UUID-addressed CAD assets are not installed locally.
  JSON import/export never embeds or restores CAD, canonical PLY, or PNG bytes;
  preserve or migrate the complete managed asset tree separately when binary
  portability is required.
- Room-monitor signaling is unavailable: inspect
  `GET /jobs?include_services=1`, confirm the managed `monitor-webrtc:ugreen`
  service is running, and inspect its `monitor_webrtc.v2` error reason. Allow
  the configured STUN UDP port (3478 by default) plus WebRTC media on the
  trusted lab LAN. The loopback signaling port is intentionally not exposed to
  browsers.
- Room-monitor diagnostics show packets arriving but zero received/decoded
  frames: confirm the active worker status reports
  `vp8_packet_max_bytes: 1100`. A worker started before the MTU fix must be
  restarted before retrying the browser connection.
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
