<picture>
  <source media="(prefers-color-scheme: dark)" srcset="posetestbot/web/static/cow_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="posetestbot/web/static/cow_light.png">
  <img src="posetestbot/web/static/cow_light.png" alt="PoseTestBot cow logo" width="96" align="right">
</picture>

# PoseTestBot

PoseTestBot is a local lab data-acquisition and BOP dataset export tool for
robot-mounted and static RGB-D sensors. Its finish line is a synchronized,
calibrated, inspectable BOP-format dataset with optional BlenderProc-generated
GT/masks.

This repository does not run pose-estimation runtimes, convert estimator output
to BOP result CSVs, run BOP Toolkit evaluation, or export metric reports. Those
downstream workflows should consume this repo's BOP output from a separate
project.

## What Is In Scope

- Fixed real lab iiwa profile with explicit execution safety gates.
- Capture planning, preflight, and supervised capture execution.
- RealSense, OAK-D Pro, and ZED 2i sensor registry/status/capture contracts.
- Run-scoped camera enable/disable controls that retain configured camera
  identity and calibration metadata while excluding disabled cameras from work.
- Non-destructive synchronization under `processed/synchronized/`.
- Sync quality reporting.
- Pinned PoseGridGen target preview/generation, immutable target bundles,
  legacy ArUcoGridGen import, split marker detection/pose solving,
  factory-vs-OpenCV RealSense color intrinsic comparison, explicit
  hand-eye/known-grid
  extrinsic solving, selection-gated promotion, and derived RGB-D rectification.
- Pinned PoseTemplateCreator CAD catalog, exact full-pose slicing, immutable
  printable pose templates, run placement, and per-instance object GT.
- BlenderProc preparation/render planning for optional GT and masks.
- BOP dataset export, model metadata, targets, frame maps, and optional
  multiview/COCO sidecars.
- React/shadcn operator console backed by the Flask API and local job runner.

## Quick Setup

Install the Python 3.12 dependencies and initialize the pinned target generator:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
```

Run Python entry points through `uv`:

```bash
uv run python scripts/robot_status.py --json
```

## Hardware Profile

Current lab expectations:

- 3 Intel RealSense D435-class cameras.
- 1 Luxonis OAK-D Pro.
- 1 Stereolabs ZED 2i.
- KUKA LBR iiwa at `172.31.1.147:30300`.
- Lab receiver IP on the robot subnet: `172.31.1.169`.
- Normal network IP on the same interface: `10.145.8.132`.

The sole robot profile is the real lab iiwa. Inspect it without sending UDP commands:

```bash
uv run python scripts/robot_status.py
uv run python scripts/robot_status.py --json
```

Check sensors and acquisition runtimes:

```bash
uv run python scripts/sensor_status.py --json
uv run python scripts/sensor_adapters.py --json
uv run python scripts/runtime_status.py --json
```

`runtime_status.py` checks acquisition-relevant external tools only:
BlenderProc for optional GT rendering and the ZED SDK Python module.

## Run Configs

Create an operator intent artifact:

```bash
uv run python scripts/create_run_config.py working_data/example_run
```

Defaults:

- real lab robot profile,
- current lab sensor list,
- objectless dataset mode until an immutable pose template is selected,
- `real_full_capture_validation` as the saved sequence,
- `plan_only=true`.

New run configs also make the robot stream frames explicit:
`robot_flange -> template_base` with `kuka_abc_radians`; optional fixed edges
can describe flange-to-TCP or template-base-to-physical-base transforms. Older
real-profile configs without frame metadata remain readable and receive a
`legacy_frames_inferred` warning; fake-profile configs are rejected.
New runs are objectless by default. Object-bearing runs use the Pose Templates
page or pose-template CLIs to select an immutable bundle, resolve its physical
instances, and retain stable catalog UUID and BOP `obj_id` provenance.
For the managed pose-template workflow, create the run with
`--dataset-mode pose_template`, then select and confirm an immutable template
in the console's **Workflow → Ground Truth** phase. See
[`docs/POSETEMPLATECREATOR_OBJECT_GT.md`](docs/POSETEMPLATECREATOR_OBJECT_GT.md).

The **Workflow → Run Setup** camera list has an **Enabled for capture and
calibration** checkbox for each configured camera. Disabling a camera retains
its identity, alias, mounting/orientation metadata, and calibration-profile
selection in `run_config.json`; it excludes that camera from capture planning
and preflight, calibration, rewrite-gate expectations, and the Cell scene. At
least one camera must remain enabled. Regenerate any already-written capture
plan and preflight after changing this selection so their evidence matches the
current run configuration.

For example, a measured flange-to-TCP edge can be recorded at creation time:

```bash
uv run python scripts/create_run_config.py working_data/example_run \
  --fixed-transform-json '{"from":"robot_flange","to":"tcp","rotation_quaternion_wxyz":[1,0,0,0],"translation_mm":[0,0,125]}'
```

Preview the default physical validation workflow without executing it:

```bash
uv run python scripts/create_run_config.py working_data/real_run \
  --sequence real_full_capture_validation \
  --print-sequence-plan
```

Write and queue preflight-aware sequence plans:

```bash
uv run python scripts/run_preflight.py working_data/example_run --write
uv run python scripts/run_pipeline_sequence.py working_data/example_run \
  --sequence sync_to_bop_dry_run --plan-only
```

## Capture

Plan capture startup commands without opening hardware:

```bash
uv run python scripts/run_capture_plan_stage.py working_data/example_run
uv run python scripts/run_capture_plan_preflight.py working_data/example_run
uv run python scripts/run_capture_execution_plan.py working_data/example_run
```

Physical execution is explicitly gated and must be operator-triggered:

```bash
uv run python scripts/run_capture_execution_plan.py working_data/real_run \
  --allow-cameras --allow-real-robot --include-sensors
uv run python scripts/run_capture_execution_stage.py working_data/real_run \
  --allow-cameras --allow-real-robot --include-sensors
```

The supervised receiver gets those acknowledgements only in its runtime
command; they are never stored in `capture_plan.json` or a plan-only sequence.
Its default first-packet and inter-packet timeouts are 120 and 60 seconds, and
the supervisor default is 300 seconds. Before receiver bind or `START`, every
selected camera has 15 seconds to publish at least three valid, committed
`frame_metadata.jsonl` records. Override these bounds with `--startup-wait`,
`--receive-start-timeout-s`, `--receive-idle-timeout-s`, and `--timeout-s` when
the reviewed motion program requires different bounds.
Direct `start_iiwa.py` and `scripts/pose_receiver_udp_json.py` invocations also
require both fresh acknowledgement flags. Prefer the supervised capture stage
for coordinated camera startup and cleanup. The receiver refuses to bind or
send START when `raw_robot_ee_poses.json` already exists and preserves failed
or canceled streams as unique `raw_robot_ee_poses.partial.*.json` evidence.

## Synchronization And Quality

Synchronize without mutating raw capture folders:

```bash
uv run python scripts/sync_run_non_destructive.py working_data/example_run
uv run python scripts/run_sync_quality.py working_data/example_run
```

Derived sync folders are written below:

```text
processed/synchronized/<sensor>/
```

Synchronization is transactional and emits `sync_report.v2` per sensor plus
`sync_quality_report.v2` at run level. Synchronized metadata names the derived
frame paths, retains source-frame provenance, and records any timestamp-source
fallback. Raw frames and raw robot poses are never overwritten; start a new run
root when capture preflight reports existing raw data.

## Calibration

The preferred operator path is **Workflow → Calibration**. One form selects
exactly one geometry, one or more captured cameras, and one of two authoritative
modes:

- **Robot-mounted camera (eye-in-hand):** the target is stationary relative to
  `template_base`; the primary result is `camera → robot_flange`.
- **Static camera (eye-to-hand):** the target is attached to `robot_flange`;
  the primary result is `camera → template_base`.

**Run calibration** queues one CPU/disk parent job. It never opens cameras or
commands the robot. Only selected captured camera folders are synchronized into
an immutable derived attempt under
`processed/calibration/<attempt_id>/`. The four displayed phases are Prepare
data, Estimate target poses, Compare robot-camera solutions, and Validate and
rank. Every PnP/extrinsic combination and failure remains available for review.

The default **Auto compare — recommended** policy compares IPPE, ITERATIVE, and
SQPNP using a common robust point mask and LM refinement, then compares Tsai,
Park, Horaud, Andreff, Daniilidis, Shah, and Li robot-camera solutions with
deterministic robust-closure outlier rejection and leave-one-pose-out
validation. Each attempt also compares the captured RealSense factory color
projection with a manual OpenCV intrinsic fit. Both candidates and their
matrix/distortion deltas remain in `intrinsic_comparison.json`; the captured
factory profile is retained whenever its projection is OpenCV-compatible, and
the manual fit remains comparison evidence. RealSense
`inverse_brown_conrady` is forward-OpenCV-compatible only when every recorded
distortion coefficient is finite and exactly zero, where both directions
reduce to the same pinhole projection. Nonzero inverse coefficients are never
passed to OpenCV as forward distortion. When the factory projection is
unusable, the manual fit activates only after its 15-view, 6/9 coverage, 3 px
per-view, 1.5 px RMS, five-view held-out, and parameter-plausibility gates pass;
factory and manual evidence are retained either way.

For RealSense calibration, synchronization pairs each color frame's sensor
exposure timestamp with the robot pose's host-wall timestamp. Every camera
timestamp must be present in the SDK `global_time` domain, every robot pose must
have `host_wall_timestamp_ns`, the manual offset is fixed at zero, timestamp
fallback is forbidden, and frames farther than 20 ms from the nearest pose are
excluded with evidence.

Target poses require at least 12 common corner inliers, 50% support, four
markers with at least three supported corners each spanning two target rows and
columns, and at most 3 px whole-board mean reprojection error. Across accepted
views, at least 50% of target markers and 60% of target rows and columns must be
covered (`calib00`: 18/35 markers, 3/5 rows, 5/7 columns). Final passing also
requires at least 15 accepted views, 6/9 image-centroid cells, four distinct
motion poses, at least 20 mm translation span and 5° rotation span, rotations
of at least 2° with rotation-axis second/first singular ratio at least 0.15,
at least six hand-eye inliers, at most 10 mm mean translation residual, at most
5° mean rotation residual, at most 25% motion-balanced outliers, and at most
25% outliers within any repeated motion. Raw outlier density remains evidence,
not a promotion gate. Solver fitting uses at most five evenly spaced frames per
motion, then validates every accepted frame with motion-balanced
residual/outlier evidence. For two or more cameras estimating the same
stationary companion frame, ranking evaluates only complete bundles that use
the same PnP and extrinsic method for every camera. Every individual candidate
must pass and the maximum pairwise companion closure must be at most 10 mm and
5°. The best mean individual score establishes a quality baseline; bundles
within 0.01 normalized score are treated as quality-equivalent and ranked by
their normalized pairwise companion closure. Bundles outside that band remain
ordered by individual quality, so a poor per-camera solution cannot win on
closure alone. Deterministic numeric and method tie-breaks finish the ordering.
Normalized ranking values are rounded to six decimals before comparison; this
already corresponds to only 0.00001 mm for a translation-only difference or
0.000005° for a rotation-only difference. Smaller solver dust therefore falls
through to the canonical PnP/extrinsic method order.
If no common bundle passes, every recommendation and promotion fails closed;
an explicit override must still select one complete recorded passing bundle.
Single-camera ranking is unchanged. Recommendations remain inactive until the
operator accepts them.

The 2026-07-21 `calib00` campaign retained RealSense `033422071805` as disabled
configuration and completed physical acquisition and calibration with the two
enabled cameras. Immutable attempt `3c4a0b7b765f44bd9cc37fffc48fb321`
promoted the complete `IPPE + Horaud` bundle. Camera `825412070181` uses
profile `825412070181_eye_in_hand_IPPE_horaud_3c4a0b7b` with camera-to-flange
translation `[5.062439, -251.370725, 74.046213]` mm and WXYZ quaternion
`[0.68773742, -0.10903413, -0.13392413, -0.70511923]`; camera
`923322072633` uses profile
`923322072633_eye_in_hand_IPPE_horaud_3c4a0b7b` with translation
`[8.952679, 260.870281, 74.983681]` mm and quaternion
`[0.70485080, 0.11024180, 0.07467657, -0.69674638]`. Their independently
estimated stationary-target companion transforms close within 3.612 mm and
0.277 degrees. Held-out mean residuals are 3.129 mm / 0.491 degrees and
3.332 mm / 0.427 degrees, with 652/652 and 655/656 inliers respectively.

The factory color projections were retained because their recorded
`inverse_brown_conrady` coefficients are exactly zero. The manual OpenCV fits
remain immutable comparison evidence: factory/manual held-out RMS was
1.194/0.967 px for `825412070181` and 1.322/1.013 px for `923322072633`.
Depth scale and depth-to-color alignment remain factory SDK provenance, not a
depth recalibration. A saved-data depth-plane diagnostic found a range-dependent
metric-depth anomaly on `923322072633`; use the promoted RGB eye-in-hand
extrinsic, but keep metric depth from that camera explicitly unvalidated until
a later cable/firmware and depth-specific check.

The **Calibration Targets** page previews and fits PoseGridGen ArUco boards and
stores immutable bundles below
`working_data/calibration_targets/<target_id>/`. Saved targets remain browsable
and selectable when PoseGridGen generation is unavailable. Generation never
selects a target automatically.

Placement choices are unknown, identity-aligned to `template_base`, or the
PoseGridGen board-to-base pose converted to PoseTestBot's millimetre/WXYZ frame
contract. Once target-dependent calibration or BOP artifacts exist, changing
the target or placement is blocked and a new run is required. See
[`docs/POSEGRIDGEN_CALIBRATION_TARGETS.md`](docs/POSEGRIDGEN_CALIBRATION_TARGETS.md)
for bundle, provenance, API, and recovery details.

The calibration import stage prefers that run-config selection. For an older
run it still accepts an exact legacy ArUcoGridGen 1.0 JSON or a PoseGridGen 2.0
source manifest:

```bash
uv run python scripts/run_calibration_target_import.py working_data/example_run \
  --source working_data/example_run/aruco_grid_config.json \
  --aligned-to-template-base
uv run python scripts/run_aruco_detection_stage.py working_data/example_run
uv run python scripts/run_intrinsic_calibration_stage.py working_data/example_run \
  --mode calibrate
uv run python scripts/run_aruco_pose_stage.py working_data/example_run
uv run python scripts/run_calibration_observations.py working_data/example_run \
  --target-spec calibration_target.json
```

The stage-level commands remain available as advanced diagnostics and retain
their existing behavior:

```bash
uv run python scripts/run_calibration_solver.py working_data/example_run \
  --mode compare
uv run python scripts/run_calibration_validation.py working_data/example_run \
  --select-profile realsense_123=PROFILE_ID
```

The intent-level promotion transaction mirrors accepted evidence into the
canonical calibration artifacts, merges exactly one valid `calibration.v2`
profile for each accepted camera, preserves unrelated profiles, and records
attempt, operator, target, PnP, and extrinsic provenance. The diagnostic CLI
promotion path remains explicit as well:

```bash
uv run python scripts/run_calibration_validation.py working_data/example_run \
  --select-profile realsense_123=PROFILE_ID --promote
uv run python scripts/run_camera_rectification.py working_data/example_run
```

Rectification writes only below `processed/rectified/<sensor>/`, using linear
RGB and nearest-neighbor aligned-depth remapping. BlenderProc preparation and
BOP export prefer that tree when present. `run_aruco_stage.py` remains the
factory-intrinsics compatibility wrapper.

## BOP Dataset Export

Prepare optional BlenderProc inputs and render plan:

```bash
uv run python scripts/run_blenderproc_prepare_stage.py working_data/example_run \
  --calibration-profiles working_data/example_run/calibration_profiles.json
uv run python scripts/run_blenderproc_render_stage.py working_data/example_run --dry-run
```

Export BOP dataset structure:

```bash
uv run python scripts/run_bop_export_stage.py working_data/example_run \
  --calibration-profiles working_data/example_run/calibration_profiles.json
```

The transactional export emits `bop_export_manifest.v3` and uses standard
BOP-scenewise paths:

```text
bop/
├── dataset_info.json
├── posetestbot_bop_frame_map.json
├── posetestbot_pose_template.json       # pose-template runs
├── posetestbot_instance_map.json        # pose-template runs
├── models/
└── <split>/<scene_id>/
    ├── rgb/
    ├── depth/
    ├── scene_camera.json
    ├── scene_gt.json
    ├── scene_gt_info.json
    ├── mask/          # optional
    └── mask_visib/    # optional
```

The preparation and export CLIs accept the same repeatable `--object-name` /
`--objectless` choice. Objectless rendering writes a successful skipped plan
without invoking BlenderProc. Objectless BOP output retains RGB, depth, and
camera metadata, writes empty GT and targets, and contains no models or masks.

The export preserves:

- scene RGB/depth,
- `scene_camera.json`,
- explicit empty or imported `scene_gt*.json`,
- optional `mask/` and `mask_visib/`,
- `bop_export_manifest.json`,
- `posetestbot_bop_frame_map.json`,
- `models/obj_XXXXXX.ply`,
- `models/models_info.json`,
- `test_targets_bop19.json`.

Pose-template runs load every physical instance independently in BlenderProc
2.8.0. Duplicate instances share one stable numeric `obj_id` and one exported
model, while their masks and `scene_gt` rows retain distinct immutable instance
UUIDs in PoseTestBot sidecars.

## Pipeline Sequences

Useful presets:

- `real_full_capture_validation`
- `sync_aruco`
- `sync_aruco_calibration_observations`
- `sync_aruco_calibration_candidates`
- `sync_aruco_calibration_solver`
- `sync_aruco_calibration_validation`
- `aruco_grid_full_calibration`
- `calibrated_capture_to_bop_dataset_dry_run`
- `sync_to_bop_dry_run`
- `sync_to_bop_calibrated_dry_run`
- `capture_to_bop_dataset_dry_run`

List current sequences and stages through the Flask API:

```bash
curl http://127.0.0.1:5000/pipeline/stages
curl http://127.0.0.1:5000/pipeline/sequences
```

## Web UI

Start the Flask-backed operator console:

```bash
uv run posetestbot-web
# source-checkout compatibility entrypoint:
uv run python web_interface.py
```

The web server intentionally defaults to unauthenticated `0.0.0.0` for the
trusted lab LAN and still exposes deliberate real-robot controls. Do not expose
it to an untrusted network. Web run paths are confined to `working_data` by
default; add path-list entries with `POSETESTBOT_WEB_RUN_ROOTS`. Read-only
external inputs default to `scripts/default_data`; extend them with
`POSETESTBOT_WEB_INPUT_ROOTS`. Symlink escapes and output paths outside
the selected run are rejected. Installed deployments may set
`POSETESTBOT_APP_ROOT`; otherwise a source checkout is detected automatically
and an installed command uses its current working directory. CLI tools continue
to accept explicit paths.

The bundled console has desktop routes for Dashboard, Devices, Cell,
Calibration Targets, Pose Templates, Workflow, and Jobs. The read-only Cell
page renders the HRI template, base/flange/TCP proxies, calibrated cameras,
pose-template instances, calibration target, and exact recorded trajectories
in right-handed Z-up millimetres.
Missing frame edges remain visibly unresolved, and a component/provenance list
remains available without WebGL. For each resolved camera that list includes
the exact calibration profile identity, camera-to-parent 4 × 4 matrix, WXYZ
quaternion, millimetre translation, stationary-target companion transform when
estimated, quality, solver, common-bundle promotion, target, and intrinsic and
synchronization provenance. It remembers the selected run,
system/light/dark theme, and manual IIWA target in the browser. The Devices
page distinguishes **Capture-ready**, **Not capture-ready**, and
**Disconnected** cameras and shows a human-readable readiness reason. It
blocks preview, snapshot, and new run selection for cameras that are not ready,
while still allowing a previously selected unavailable camera to be
deselected.

Physical capture remains separate from
ordinary stage forms: current preflight evidence is required, camera previews
are stopped first, and two fresh acknowledgements send `allow_cameras` and
`allow_real_robot` together in that one request. A non-plan-only capture
sequence is rejected by `/pipeline/run-config`; use Advanced Capture instead.
Manual **Start IIWA** controls also require a fresh target confirmation plus
both execution acknowledgements. **Stop IIWA** remains available without
motion-start gates, but it is not a safety stop and cannot interrupt active
motion. In the operator-reported running calibration program it exits the
waiting program and requires a manual application restart, so do not use it
between calibration captures.

The checked-in `posetestbot/web/static/ui/` build is what Flask and installed
wheels serve, so Bun is not a runtime dependency. Frontend development uses the
Bun-locked Vite project:

```bash
cd frontend
bun install --frozen-lockfile
bun run typecheck
bun run lint
bun run build
```

The production build clears the previous output before writing hashed assets.
Use `bash scripts/install.sh --with-web-build` to perform the locked install and
build through the project installer.

Important endpoints:

- `GET /ui/bootstrap`
- `GET /ui/runs`
- `GET /ui/cell-scene`
- `GET /ui/cell-scene/timeline`
- `GET /robot/status`
- `POST /run-command`
- `GET /sensors/status`
- `GET /runtime/status`
- `GET|POST /monitoring/webcam`
- `POST /monitoring/webcam/<job_id>/brightness/autocalibrate`
- `GET /calibration-targets/status`
- `GET /calibration-targets/capabilities`
- `POST /calibration-targets/fit`
- `POST /calibration-targets/preview`
- `GET /calibration-targets/bundles`
- `DELETE /calibration-targets/bundles/<target_id>`
- `POST /calibration-targets/bundles/<target_id>/select`
- `GET /calibration/setup`
- `GET|POST /calibration/attempts`
- `GET /calibration/attempts/<attempt_id>`
- `POST /calibration/attempts/<attempt_id>/promote`
- `GET /pose-templates/status`
- `GET /pose-templates/catalog`
- `POST /pose-templates/catalog/upload`
- `POST /pose-templates/preview`
- `POST /pose-templates/generate`
- `GET /pose-templates/library`
- `GET|POST /pose-templates/runs/selection`
- `POST /hardware/status`
- `GET|POST /run-config`
- `GET|POST /pipeline/preflight`
- `POST /pipeline/run-config`
- `GET /pipeline/recommendations`
- `GET /capture/jobs`
- `GET /capture/status`
- `POST /capture/jobs/<job_id>/stop`

## Rewrite Gates

Current gates:

- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

Run all gate status summaries:

```bash
uv run python scripts/run_rewrite_status.py working_data/example_run --write
```

## Removed Legacy Entry Points

The destructive multi-camera sync/capture wrappers, `realsense_multi.py`, the
duplicate `main.py` web launcher, legacy `aruco_pose_estimation.py`, shell
BlenderProc wrapper/preparation script, ROI generators, and superseded
transform scripts were removed. Use the supported replacements:

- capture: `run_capture_plan_stage.py`, `run_capture_plan_preflight.py`, and
  `run_capture_execution_stage.py`;
- sync: `sync_run_non_destructive.py` followed by `run_sync_quality.py`;
- BlenderProc: `run_blenderproc_prepare_stage.py` and
  `run_blenderproc_render_stage.py`;
- calibration transforms: the calibration observation, solver, validation, and
  explicit promotion stages.

## Validation

Recommended local validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
git diff --check
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_console_playwright.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
```

The single source of truth for unfinished rewrite work is
[`docs/REWRITE_REMAINING_WORK.md`](docs/REWRITE_REMAINING_WORK.md).
