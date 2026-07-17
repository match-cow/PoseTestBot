# PoseTestBot Acquisition Baseline

This document describes the current target architecture after the
acquisition-only restructure. PoseTestBot is a local lab webapp and Python
toolkit for collecting robot/RGB-D data, calibrating sensors, synchronizing
frames, optionally generating dataset GT/masks, and exporting BOP-format
datasets.

The BOP dataset is the boundary. Estimator execution, result conversion,
evaluator bridges, and metric reporting are intentionally out of scope for this
repository.

## Core Responsibilities

PoseTestBot owns:

- fixed real lab robot profile and iiwa UDP command contracts,
- capture planning, preflight, and supervised capture execution,
- RGB-D sensor registry/status/capture adapters,
- non-destructive frame/pose synchronization,
- sync quality reporting,
- calibration profile validation, observation extraction, solving, validation,
  and promotion,
- optional BlenderProc preparation/render planning for GT/masks,
- BOP dataset export,
- local job orchestration and transition Flask APIs.

PoseTestBot does not own:

- FoundationPose/MegaPose/SAM6D execution,
- BOP19 result CSV conversion,
- BOP Toolkit evaluation,
- legacy accuracy dashboards or metric-report export.

## Package Layout

- `posetestbot.config`: fixed real lab robot profile configuration and overrides.
- `posetestbot.robot`: read-only robot status and UDP helper contracts.
- `posetestbot.sensors`: static registry, discovery/status helpers, frame
  writer contracts, and testable RealSense, OAK-D Pro, and ZED 2i capture
  support.
- `posetestbot.blenderproc`: transactional preparation and render orchestration
  for optional GT/mask artifacts.
- `posetestbot.jobs`: local job runner with resource conflict rejection,
  persistent job records, logs, and cancellation.
- `posetestbot.pipeline`: run config, preflight, capture planning/execution,
  hardware snapshots, typed stage registry, sequences, recommendations, and
  rewrite gates.
- `posetestbot.sync`: non-destructive sync plus run-level quality reporting.
- `posetestbot.calibration`: profile schema/migration, preflight, target
  observations, attempt-scoped planar PnP, two-geometry robot-camera solving,
  deterministic ranking, and transactional validation/promotion.
- `posetestbot.aruco`: calibration target coverage summaries.
- `posetestbot.bop`: BOP writer and geometry helpers.
- `posetestbot.io`: atomic artifact/directory promotion, artifact constants,
  and manifest helpers.

Scripts under `scripts/` should stay thin wrappers over importable modules.

## Data Flow

1. Create `run_config.json` from operator intent.
2. Write `run_preflight_report.json` to snapshot config, robot, sensor, and
   acquisition-runtime readiness.
3. Write `capture_plan.json` and `capture_plan_preflight_report.json`.
4. Write `capture_execution_plan.json`.
5. Execute supervised capture when explicitly requested.
6. Preserve raw sensor folders and `raw_robot_ee_poses.json`.
7. Transactionally create derived synchronized folders under
   `processed/synchronized/` and emit `sync_report.v2`.
8. Write `sync_quality_report.json`.
9. Select captured cameras, a saved target, and eye-in-hand or eye-to-hand mode.
10. Queue one non-hardware calibration parent job. It writes every calculation
    below `processed/calibration/<attempt_id>/`, compares supported PnP and
    robot-camera methods, and ranks passing results per camera.
11. Explicitly accept recommendations or passing per-camera overrides. The
    promotion transaction preserves unrelated profiles and updates canonical
    calibration artifacts plus selected-camera mounting metadata.
12. Optionally prepare/render BlenderProc GT/mask artifacts.
13. Transactionally export standard `bop/<split>/<scene_id>/` scenes, model
    metadata, targets, root frame provenance, and `bop_export_manifest.v2`.
14. Inspect gate status through Flask or CLI and inspect run files directly.

The reusable synchronization, observation, and solver internals retain their
run-wide defaults for existing CLIs/APIs, while also accepting explicit sensor
or target-pose subsets and alternate derived output roots for attempt-scoped
orchestration. Those alternate roots never require moving or deleting raw
capture evidence.

## Artifact Contracts

Important root artifacts:

- `dataset_manifest.json`
- `run_config.json`
- `run_preflight_report.json`
- `hardware_status_report.json`
- `capture_plan.json`
- `capture_plan_preflight_report.json`
- `capture_execution_plan.json`
- `capture_execution_status.json`
- `capture_execution_report.json`
- `raw_robot_ee_poses.json`
- `sync_quality_report.json`
- `pipeline_sequence_plan.json`
- `rewrite_gate_report.json`
- `rewrite_status_report.json`

Sensor folder artifacts:

- `rgb/*.png`
- `depth/*.png`
- `frame_metadata.jsonl`
- `cam_K.txt`
- `depthscale.txt`
- `camera.json`
- `camera_data.json`

Calibration artifacts:

- immutable `processed/calibration/<attempt_id>/request.json`, `progress.json`,
  per-frame PnP candidates, extrinsic candidates, ranking/checks, candidate
  profiles, and promotion evidence,
- `calibration_preflight_report.json`
- `calibration_observations.json`
- `calibration_candidates.json`
- `calibration_profiles_from_observations.json`
- `calibration_solver_report.json`
- `calibration_profiles_solved.json`
- `calibration_validation_report.json`
- `calibration_profiles.json`

BOP artifacts:

- `bop/bop_export_manifest.json`
- `bop/dataset_info.json`
- `bop/posetestbot_bop_frame_map.json`
- scene `bop/<split>/<scene_id>/rgb/`, `depth/`, `scene_camera.json`,
  `scene_gt.json`, `scene_gt_info.json`, optional `mask/`, and optional
  `mask_visib/`
- `bop/models/obj_XXXXXX.ply`
- `bop/models/models_info.json`
- `bop/test_targets_bop19.json`
- optional `bop/posetestbot_multiview_targets.json`
- optional `bop/posetestbot_coco_annotations.json`

## Runtime Readiness

Runtime status is lightweight and read-only. It checks only acquisition
dependencies:

- `blenderproc` executable for non-dry-run BlenderProc rendering.
- `pyzed.sl` for ZED 2i capture support.

Sensor SDK/device visibility remains in `posetestbot.sensors.status`.
Preflight warns about missing optional runtimes, and only errors when a
selected non-dry-run stage requires an unavailable runtime.

## Pipeline Stage Registry

The typed stage registry includes:

- `rewrite_gate`
- `rewrite_status`
- `hardware_status`
- `run_preflight`
- `capture_plan`
- `capture_plan_preflight`
- `capture_execution_plan`
- `capture_execution`
- `realsense_capture_smoke`
- `calibration_preflight`
- `calibration_observations`
- `calibration_candidates`
- `calibration_solver`
- `calibration_validation`
- `sync_run`
- `sync_quality`
- `aruco`
- `aruco_coverage`
- `blenderproc_prepare`
- `blenderproc_render`
- `bop_export`

Stage specs build command arrays rather than shell strings and declare
resources before job submission. Path parameters also declare run, output,
external-input, or repository scope for web submission validation.

## Pipeline Sequences

The sequence registry includes:

- `real_full_capture_validation`
- `sync_aruco`
- `sync_aruco_calibration_observations`
- `sync_aruco_calibration_candidates`
- `sync_aruco_calibration_solver`
- `sync_aruco_calibration_validation`
- `sync_to_bop_dry_run`
- `sync_to_bop_calibrated_dry_run`
- `capture_to_bop_dataset_dry_run`

Sequences that begin with non-destructive sync should run `sync_quality`
immediately after `sync_run`.

## Rewrite Gates

Current gates:

- `rewrite_full_capture.v1`: proves intentional real robot/camera capture with
  command planning, status snapshots, supervised execution, and raw frames.
- `rewrite_calibration_validation.v1`: proves validation and explicit
  promotion of valid calibration profiles.
- `rewrite_bop_export_readiness.v1`: proves a BOP dataset has exported scenes,
  targets, and model metadata.

## Web/API Boundary

The Flask transition app exposes local operator endpoints for:

- jobs and capture jobs,
- robot/sensor/runtime/hardware status,
- run config and preflight,
- capture planning and execution planning,
- calibration stages,
- sync quality,
- typed pipeline stages and sequences,
- recommendations,
- artifact listing/preview/download,
- BOP scene/frame detail and GT/mask overlays.

It no longer exposes metric dashboards or BOP result CSV inspection.

The app remains intentionally unauthenticated and LAN-facing. Web run roots
default to `working_data` and external inputs to `object_models` plus
`scripts/default_data`; environment path lists can add trusted roots. All web
paths resolve symlinks before containment checks. Job records are anchored at
the repository `working_data/jobs` directory, and camera resources use
hierarchical device-specific locks.

## Validation

Baseline validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
git diff --check
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_preview_playwright.py
```
