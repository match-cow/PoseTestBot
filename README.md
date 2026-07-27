<picture>
  <source media="(prefers-color-scheme: dark)" srcset="posetestbot/web/static/cow_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="posetestbot/web/static/cow_light.png">
  <img src="posetestbot/web/static/cow_light.png" alt="PoseTestBot cow logo" width="96" align="right">
</picture>

# PoseTestBot

PoseTestBot records robot-mounted and static RGB-D data, calibrates and
synchronizes it without changing the raw capture, and exports an inspectable
BOP dataset. An optional guided job uses BlenderProc 2.8.0 to validate the
placed-object/camera scene and derive pose GT; its full mode then uses the
pinned official BOP Toolkit to add BOP masks and visibility evidence against
the captured depth.

Pose estimators and result conversion remain outside this repository. A narrow
**Inspect → BOP Evaluation** feature validates an already exported,
annotation-bearing dataset by running official BOP19 metrics on a compatible
result CSV or on a deterministic test-only slight perturbation of GT. It does
not execute an estimator or add an evaluation pipeline stage.

## Start Here

PoseTestBot uses Python 3.12 and `uv`. The installer initializes the supported
optional generators and validates the local environment:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator \
  --with-bop-toolkit
uv run posetestbot-web
```

The web server is unauthenticated and exposes deliberate real-robot controls.
Its default bind address is intended only for the trusted lab network; use
`POSETESTBOT_WEB_HOST=127.0.0.1` for a local-only session.

The run picker accepts folders only below the server-approved roots. The
defaults are `<repository>/working_data` and the acquisition SSD at
`/mnt/working_data_ssd`. Add other absolute roots with the colon-separated
`POSETESTBOT_WEB_RUN_ROOTS` environment variable; choose an initial folder with
`POSETESTBOT_WEB_DEFAULT_RUN_ROOT`. The initial folder must remain inside an
approved root.

In the console:

1. choose the active run in the top bar;
2. open **Workflow** and choose **Camera calibration** or **Object dataset**;
3. follow the numbered required steps;
4. resolve the single readiness check; and
5. authorize physical capture only when the cell and operator are ready.

The **Dashboard** is the live supervision surface: it keeps the room monitor
prominent, polls selected-run disk capacity, shows all active jobs and the
latest failures, and derives its five-step calibration or six-step dataset
overview from the saved run configuration and durable artifacts. Dataset step
1 reuses a prior calibration; that input does not turn the dataset journey into
a calibration run.

Library pages prepare reusable inputs but do not silently advance a run:

| Page | Purpose | Handoff |
| --- | --- | --- |
| Dashboard | Monitor the workcell, run storage, jobs, and saved workflow evidence | Exact current guided step |
| Devices | Discover cameras; save aliases, mounts, and orientation | Workflow step 1 |
| Calibration Targets | Generate or select an immutable printed ArUco target | Camera calibration step 2 |
| Workpiece Catalogue | Manage canonical CAD identity and metadata | Pose Templates |
| Pose Templates | Publish immutable printable object layouts | Object dataset step 2 |
| Cell | Inspect geometry, trajectories, and provenance | Back to Workflow for changes |
| BOP Evaluation | Compare registered method results or a test-only GT simulation with official BOP19 metrics | Read-only derived inspection; monitor the queued run in Jobs |
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
processed/bop_annotations/
processed/bop_evaluation/
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
evaluation.

In **Workflow → Object dataset → Export the BOP dataset**, choose one explicit
run-owned annotation product:

- **Plain pose ground truth** writes each instance's standard `scene_gt.json`
  rotation and millimetre translation, derived through model, pose-template
  placement, robot pose, and selected camera calibration. It intentionally
  omits visibility evidence and is not BOP19 evaluation-ready.
- **Pose + object masks and ROI** additionally writes standard full-frame
  `mask/` and `mask_visib/` PNGs plus `scene_gt_info.json` `bbox_obj`,
  `bbox_visib`, pixel counts, and visibility fractions. This is the
  evaluation-compatible product.

The background job is recoverable in **Jobs**, rewrites only derived annotation
and BOP export evidence, and never mutates raw frames, robot poses, template
snapshots, or calibration snapshots.
New annotation-bearing exports build the BOP19 target counts from GT instances
with `visib_fract >= 0.1`, matching the official localization policy. Inspect
warns when an older export's target inventory differs; its scores describe
that exported list but are not leaderboard-comparable.

Install the pinned official toolkit and its isolated locked runtime with
`bash scripts/install.sh --with-bop-toolkit`. In **Inspect → BOP Evaluation**,
select the active run, import one or more canonical BOP result files, and choose
which method/result to evaluate. Compatible files use the standard
`scene_id,im_id,obj_id,score,R,t,time` header and BOP filename convention.
For dataset-format testing before a real estimator exists, the page can instead
create a deterministic, very slightly offset result from GT; simulated results
are labelled test-only. The queued job reports overall Average Recall, AR VSD,
AR MSSD, AR MSPD, timing, and immutable toolkit/dataset/result provenance.
Registered inputs and reports live only below `processed/bop_evaluation/`.

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
