# PoseTestBot Agent Notes

These notes are for Codex and other coding agents working in this repository.
PoseTestBot is now acquisition-first: capture, calibration, synchronization,
optional GT/mask generation, and BOP dataset export are the repo boundary.
Downstream pose-estimator execution, BOP result conversion, evaluator bridges,
and metric reporting belong in a separate consumer repo.

## Operating Rules

- Use `uv` for Python environment and package management.
- Run scripts as `uv run python ...`.
- Add dependencies with `uv add ...`; do not hand-edit dependency locks unless
  a tool-generated update is impossible.
- Browser UI regressions should use Playwright tests. Keep Playwright in the dev
  dependency group, and install browser binaries only when explicitly requested.
- Production frontend builds and localhost-only Playwright regressions are
  standing-authorized outside the sandbox when sandbox stream-descriptor or
  loopback-socket restrictions prevent them. Keep those commands scoped to
  `bun run build` in `frontend/` and this repository's Playwright pytest files.
  This authorization does not include dependency or browser installation,
  external network services, camera or robot access, or physical capture.
- Keep `INSTALL.md` and `scripts/install.sh` current when dependency lists,
  SDK/runtime expectations, setup commands, or validation checks change.
- Prefer running or checking `scripts/install.sh` before adding ad hoc setup
  instructions.
- The lab KUKA iiwa is the sole robot profile. Never execute physical capture
  without explicit operator authorization and both execution safety gates.
- Do not add blocking request handlers for long-running or hardware-touching
  work. Queue them through `posetestbot.jobs.runner.LocalJobRunner` and declare
  resources.
- Preserve raw capture data. Synchronization/export work should create derived
  artifacts, usually under `processed/`, rather than renaming or deleting the
  only copy of frames.
- Keep completed status current in `docs/REWRITE_PROGRESS.md` and unfinished
  work in `docs/REWRITE_REMAINING_WORK.md`.

## Current Lab Hardware

- 3 Intel RealSense D435-class cameras.
- 1 Luxonis OAK-D Pro.
- 1 Stereolabs ZED 2i.
- KUKA LBR iiwa at `172.31.1.147:30300`.
- Lab receiver IP on the robot subnet: `172.31.1.169`.
- Normal network IP on the same interface: `10.145.8.132`.

Robot status is read-only:

```bash
uv run python scripts/robot_status.py --json
```

Plan physical capture without executing it:

```bash
uv run python scripts/create_run_config.py working_data/test_run
uv run python scripts/run_pipeline_sequence.py working_data/test_run \
  --sequence real_full_capture_validation --plan-only
```

Read-only status commands:

```bash
uv run python scripts/robot_status.py --json
uv run python scripts/sensor_status.py --json
uv run python scripts/sensor_adapters.py --json
uv run python scripts/runtime_status.py --json
```

Runtime status is acquisition-only. It checks BlenderProc for optional GT/mask
rendering and the Stereolabs ZED SDK Python module. Camera visibility remains
owned by sensor status.

## Current Architecture Boundary

Keep or extend these areas:

- `posetestbot.pipeline.capture_plan`,
  `posetestbot.pipeline.capture_plan_preflight`,
  and `posetestbot.pipeline.capture_execution`.
- `posetestbot.sensors.*` adapters, registry, status, discovery, and frame
  writer contracts.
- `posetestbot.sync.non_destructive` and `posetestbot.sync.quality`.
- `posetestbot.calibration.*` profile validation, preflight, observations,
  target import, intrinsic/rectification, frame graph, candidates, explicit
  extrinsic modes, and validation/promotion.
- `scripts/run_aruco_stage.py` and `posetestbot.aruco.coverage` as calibration
  target support.
- BlenderProc preparation/render planning for optional dataset GT/masks.
- `posetestbot.pose_templates.*` catalog, exact slicing, immutable bundle,
  run-selection, and object-instance preparation contracts.
- `scripts/run_bop_export_stage.py` and `posetestbot.bop.writer`.
- Flask operator APIs for jobs, capture status, hardware/sensor/runtime
  status, run config, preflight, calibration, sync quality, and pipeline
  sequence submission.

Do not reintroduce downstream estimator/evaluator behavior here:

- No FoundationPose/MegaPose/SAM6D stages or wrappers.
- No BOP19 result CSV conversion stage.
- No BOP Toolkit evaluation bridge.
- No legacy accuracy or metric-report export stage.

## Important Artifacts

- Raw robot pose artifact: `raw_robot_ee_poses.json`.
- Matched robot pose artifact: `match_robot_ee_poses.json`.
- Frame timestamp sidecar: `frame_metadata.jsonl`.
- Run manifest artifact: `dataset_manifest.json`.
- Run configuration artifact: `run_config.json`.
- Run preflight artifact: `run_preflight_report.json`.
- Hardware snapshot artifact: `hardware_status_report.json`.
- Capture artifacts: `capture_plan.json`,
  `capture_plan_preflight_report.json`, `capture_execution_plan.json`,
  `capture_execution_status.json`, `capture_execution_report.json`,
  and `capture_execution_logs/`.
- Derived sync report: `sync_report.json`.
- Run-level sync quality report: `sync_quality_report.json`.
- Calibration artifacts: `calibration_preflight_report.json`,
  `calibration_target.json`, `intrinsic_calibration_profiles.json`,
  per-sensor `aruco_detections.json`, `camera_rectification_report.json`,
  `calibration_observations.json`, `calibration_candidates.json`,
  `calibration_profiles_from_observations.json`,
  `calibration_solver_report.json`, `calibration_profiles_solved.json`,
  `calibration_validation_report.json`, and promoted
  `calibration_profiles.json` (`calibration.v2`; v1 remains loadable).
- Intent-level calibration attempts live under
  `processed/calibration/<attempt_id>/` and retain `request.json`,
  `progress.json`, `pnp_candidates.json`, `extrinsic_candidates.json`,
  `ranking.json`, `checks.json`, `candidate_profiles.json`, the selected target
  bundle, and explicit promotion evidence.
- BlenderProc render plan artifact: `blenderproc_render_plan.json`.
- Pose-template artifacts: global `object_catalog/object_catalog.json`, global
  immutable `pose_templates/<uuid>/pose_template_bundle.json`, run-owned
  `pose_template_selection.json`, and `object_instances.json`.
- BOP export artifacts: `bop/bop_export_manifest.json`,
  `bop/posetestbot_bop_frame_map.json`, `bop/test_targets_bop19.json`,
  `bop/models/models_info.json`, pose-template
  `bop/posetestbot_pose_template.json` and `bop/posetestbot_instance_map.json`, optional
  `bop/posetestbot_multiview_targets.json`, and optional
  `bop/posetestbot_coco_annotations.json`.

## Sensor Contracts

`posetestbot.sensors.registry` is the static single source of truth for
supported RGB-D sensor families, display names, SDK module names, capture
scripts, folder prefixes, supported resolutions, and mounting modes. It does
not open hardware. Update it first when adding or renaming a sensor adapter.

`posetestbot.sensors.frame_writer` owns shared capture output:

- legacy `rgb/` and `depth/` PNG files,
- compact `frame_metadata.jsonl` records,
- camera sidecars via `write_legacy_camera_sidecars`.

RealSense, OAK-D Pro, and ZED 2i capture scripts should write frames through
`write_legacy_rgbd_frame` or `write_aligned_rgbd_frame`.

## Pipeline Sequences

Current acquisition sequences include:

- `real_full_capture_validation`
- `sync_aruco`
- `sync_aruco_calibration_observations`
- `sync_aruco_calibration_candidates`
- `sync_aruco_calibration_solver`
- `sync_aruco_calibration_validation`
- `sync_to_bop_dry_run`
- `sync_to_bop_calibrated_dry_run`
- `capture_to_bop_dataset_dry_run`
- `aruco_grid_full_calibration`
- `calibrated_capture_to_bop_dataset_dry_run`

Keep `sync_quality` immediately after `sync_run` in reusable sequences unless
there is a clear operator-facing reason to bypass that gate.

## Rewrite Gates

The acquisition-only rewrite gates are:

- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

Run them with:

```bash
uv run python scripts/run_rewrite_gate.py <run> --gate rewrite_full_capture.v1 --write
uv run python scripts/run_rewrite_status.py <run> --write
```

## Validation

Use `uv` for tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git diff --check
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium  # only if browser binaries are missing
```
