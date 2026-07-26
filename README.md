<picture>
  <source media="(prefers-color-scheme: dark)" srcset="posetestbot/web/static/cow_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="posetestbot/web/static/cow_light.png">
  <img src="posetestbot/web/static/cow_light.png" alt="PoseTestBot cow logo" width="96" align="right">
</picture>

# PoseTestBot

PoseTestBot records robot-mounted and static RGB-D data, calibrates and
synchronizes it without changing the raw capture, and exports an inspectable
BOP dataset. Optional BlenderProc stages add GT and masks.

Pose estimators, result conversion, BOP evaluation, and metric reporting are
intentionally outside this repository. They consume PoseTestBot's BOP output
from a separate project.

## Start Here

PoseTestBot uses Python 3.12 and `uv`. The installer initializes the supported
optional generators and validates the local environment:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
uv run posetestbot-web
```

The web server is unauthenticated and exposes deliberate real-robot controls.
Its default bind address is intended only for the trusted lab network; use
`POSETESTBOT_WEB_HOST=127.0.0.1` for a local-only session.

In the console:

1. choose the active run in the top bar;
2. open **Workflow** and choose **Camera calibration** or **Object dataset**;
3. follow the numbered required steps;
4. resolve the single readiness check; and
5. authorize physical capture only when the cell and operator are ready.

Library pages prepare reusable inputs but do not silently advance a run:

| Page | Purpose | Handoff |
| --- | --- | --- |
| Devices | Discover cameras; save aliases, mounts, and orientation | Workflow step 1 |
| Calibration Targets | Generate or select an immutable printed ArUco target | Camera calibration step 2 |
| Workpiece Catalogue | Manage canonical CAD identity and metadata | Pose Templates |
| Pose Templates | Publish immutable printable object layouts | Object dataset step 2 |
| Cell | Inspect geometry, trajectories, and provenance | Back to Workflow for changes |
| Jobs | Monitor background work, logs, locks, and cancellation | Back to the originating step |

See [Operator Workflows](docs/OPERATOR_WORKFLOWS.md) for the complete operator
contract.

## Safety and Hardware Boundary

The sole robot profile is the lab KUKA LBR iiwa at
`172.31.1.147:30300`. Physical capture requires explicit operator
authorization and both execution gates. IIWA `STOP` is not a safety stop,
cannot interrupt active motion, and exits the waiting calibration application;
do not use it between calibration captures.

Current sensor inventory:

- 3 Intel RealSense D435-class cameras;
- 1 Luxonis OAK-D Pro; and
- 1 Stereolabs ZED 2i.

Read status without commanding the robot or starting capture:

```bash
uv run python scripts/robot_status.py --json
uv run python scripts/sensor_status.py --json
uv run python scripts/sensor_adapters.py --json
uv run python scripts/runtime_status.py --json
```

Create and inspect a run without touching hardware:

```bash
uv run python scripts/create_run_config.py working_data/example_run
uv run python scripts/run_pipeline_sequence.py working_data/example_run \
  --sequence real_full_capture_validation --plan-only
```

Raw capture data is preserved. Synchronization, rectification, rendering, and
export write derived artifacts, normally below `processed/` or `bop/`.

`timestamp_aligned` is the general synchronization mode. The only supported
hardware-trigger group is a qualified mixed-mount, exact-ID D435 group using
RealSense inter-camera **depth-exposure** sync. It does not certify simultaneous
RGB exposure, and OAK-D Pro or ZED 2i cannot join that group.

## Outputs

Each run is centred on:

```text
run_config.json
run_preflight_report.json
raw_robot_ee_poses.json
<sensor>/rgb/ + depth/ + frame_metadata.jsonl
processed/synchronized/
processed/calibration/
calibration_profiles.json
bop/bop_export_manifest.json
```

The BOP export always includes standard RGB-D scenes, per-image camera
parameters, the selected object models in `models/`, evaluator-compatible
geometry in `models_eval/`, and compact PoseTestBot provenance sidecars.
Annotation-free exports omit `scene_gt.json`,
`scene_gt_info.json`, masks, and GT instance maps. Pose-template exports retain
a populated `test_targets_bop19.json` derived from the confirmed object
inventory because it is needed for target-driven pose estimation and later
evaluation. They are not evaluation-ready until an explicit BlenderProc
annotation export adds GT, masks, and instance identity.

Raw capture folders intentionally remain outside `bop/`. They preserve
pre/post-motion evidence, while the BOP scenes contain only synchronized
capture-motion frames.

Run the acquisition-only gates with:

```bash
uv run python scripts/run_rewrite_gate.py working_data/example_run \
  --gate rewrite_full_capture.v1 --write
uv run python scripts/run_rewrite_status.py working_data/example_run --write
```

The active gates are `rewrite_full_capture.v1`,
`rewrite_calibration_validation.v1`, and
`rewrite_bop_export_readiness.v1`.

## Development

The Flask package serves the checked-in Vite build from
`posetestbot/web/static/ui/`. Rebuild it; never edit hashed assets directly:

```bash
cd frontend
bun install --frozen-lockfile
bun run typecheck
bun run lint
bun run build
```

Validate Python and browser contracts with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
git diff --check
```

The default pytest selection excludes Playwright. Install Chromium only when
explicitly needed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium
```

## Documentation

- [Installation and SDK expectations](INSTALL.md)
- [Operator workflows](docs/OPERATOR_WORKFLOWS.md)
- [Workpiece Catalogue](docs/WORKPIECE_CATALOGUE.md)
- [Pose templates and object GT](docs/POSETEMPLATECREATOR_OBJECT_GT.md)
- [Calibration targets](docs/POSEGRIDGEN_CALIBRATION_TARGETS.md)
- [Rewrite progress](docs/REWRITE_PROGRESS.md)
- [Remaining work and real-lab acceptance](docs/REWRITE_REMAINING_WORK.md)
- [Agent and development rules](AGENTS.md)
