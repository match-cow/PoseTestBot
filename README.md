<img src="assets/cow200.png" alt="PoseTestBot cow logo" width="96" align="right">

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

- Fake-first and real iiwa robot profile selection.
- Capture planning, preflight, and supervised capture execution.
- RealSense, OAK-D Pro, and ZED 2i sensor registry/status/capture contracts.
- Non-destructive synchronization under `processed/synchronized/`.
- Sync quality reporting.
- Calibration target detection support via ArUco/ChArUco/checkerboard-style
  outputs.
- Calibration observation extraction, candidate generation, solving,
  validation, and explicit promotion.
- BlenderProc preparation/render planning for optional GT and masks.
- BOP dataset export, model metadata, targets, frame maps, and optional
  multiview/COCO sidecars.
- Flask transition UI/API for local lab operation.

## Quick Setup

Install dependencies with `uv`:

```bash
uv sync
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

The default robot profile is fake. Inspect it without sending UDP commands:

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

## Fake Acquisition To BOP Smoke

The hardware-free acquisition smoke exercises fake iiwa capture, synthetic
RGB-D fixture generation, non-destructive sync, sync quality, BlenderProc
planning, BOP export, and the fake acquisition gate:

```bash
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_fake_bop_smoke --overwrite
uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_fake_bop_smoke \
  --gate rewrite_fake_acquisition_to_bop.v1 --write
```

## Run Configs

Create an operator intent artifact:

```bash
uv run python scripts/create_run_config.py working_data/example_run
```

Defaults:

- fake robot profile,
- current lab sensor list,
- `object_models` as the object registry,
- `sync_to_bop_dry_run` as the saved sequence,
- `plan_only=true`.

Use real robot mode only intentionally:

```bash
uv run python scripts/create_run_config.py working_data/real_run \
  --robot-mode real \
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

Run the safe fake pose-only path:

```bash
uv run python scripts/run_capture_execution_stage.py working_data/example_run \
  --mode pose_only_fake
```

Full camera execution remains explicitly gated:

```bash
uv run python scripts/run_capture_execution_plan.py working_data/real_run \
  --mode full --allow-cameras --allow-real-robot --include-sensors
uv run python scripts/run_capture_execution_stage.py working_data/real_run \
  --mode full --allow-cameras --allow-real-robot --include-sensors
```

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

## Calibration

Generate target detections and calibration inputs:

```bash
uv run python scripts/run_aruco_stage.py working_data/example_run
uv run python scripts/run_aruco_coverage_stage.py working_data/example_run
uv run python scripts/run_calibration_observations.py working_data/example_run
```

Create candidate/solved profiles and validate them:

```bash
uv run python scripts/run_calibration_candidates.py working_data/example_run
uv run python scripts/run_calibration_solver.py working_data/example_run \
  --holdout-fraction 0.2 --compare-hand-eye-methods
uv run python scripts/run_calibration_validation.py working_data/example_run
```

Promotion to `calibration_profiles.json` is explicit:

```bash
uv run python scripts/run_calibration_validation.py working_data/example_run --promote
```

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
  --calibration-profiles working_data/example_run/calibration_profiles.json \
  --object-folder object_models
```

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

## Pipeline Sequences

Useful presets:

- `fake_capture_rehearsal`
- `fake_capture_execution`
- `real_full_capture_validation`
- `sync_aruco`
- `sync_aruco_calibration_observations`
- `sync_aruco_calibration_candidates`
- `sync_aruco_calibration_solver`
- `sync_aruco_calibration_validation`
- `sync_to_bop_dry_run`
- `sync_to_bop_calibrated_dry_run`
- `capture_to_bop_dataset_dry_run`
- `fake_capture_to_bop_dataset_dry_run`

List current sequences and stages through the Flask API:

```bash
curl http://127.0.0.1:5000/pipeline/stages
curl http://127.0.0.1:5000/pipeline/sequences
```

## Web UI

Start the transition Flask app:

```bash
uv run python web_interface.py
```

Important endpoints:

- `GET /robot/status`
- `POST /run-command`
- `GET /sensors/status`
- `GET /runtime/status`
- `POST /hardware/status`
- `GET|POST /run-config`
- `GET|POST /pipeline/preflight`
- `POST /pipeline/run-config`
- `GET /pipeline/recommendations`
- `GET /artifacts`
- `GET /artifacts/preview`
- `GET /artifacts/file`
- `GET /artifacts/bop-scene`
- `GET /artifacts/bop-frame`
- `GET /artifacts/bop-frame-overlay`
- `GET /capture/jobs`
- `GET /capture/status`
- `POST /capture/jobs/<job_id>/stop`

The artifact browser is scoped to acquisition artifacts, job logs, calibration
reports, BOP scene/frame inspection, and GT/mask overlays.

## Rewrite Gates

Current gates:

- `rewrite_fake_acquisition_to_bop.v1`
- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

Run all gate status summaries:

```bash
uv run python scripts/run_rewrite_status.py working_data/example_run --write
```

## Validation

Recommended local validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git diff --check
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_fake_bop_smoke --overwrite
uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_fake_bop_smoke \
  --gate rewrite_fake_acquisition_to_bop.v1 --write
```
