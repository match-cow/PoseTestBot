# Rewrite Progress

Last updated: 2026-07-10

PoseTestBot has been refocused as an acquisition, calibration,
synchronization, and BOP dataset export repository. The BOP dataset export is
the finish line for this repo.

## Current Scope

In scope:

- fake-first and real robot profile handling,
- sensor registry/status/capture contracts,
- capture planning, preflight, and supervised execution,
- non-destructive synchronization,
- sync quality reports,
- calibration profile preflight, observations, candidates, solver, validation,
  and explicit promotion,
- ArUco/target detection support for calibration,
- BlenderProc preparation/render planning for optional GT/masks,
- BOP dataset export,
- artifact browsing and BOP scene/frame/overlay inspection,
- local Flask transition UI/API and job runner.

Out of scope in this repository:

- pose-estimator runtime orchestration,
- estimator output conversion to BOP result CSVs,
- BOP Toolkit evaluation,
- metric report/export dashboards.

## Completed In This Acquisition-Only Pass

- Completed the rewrite hardening audit in `docs/REWRITE_HARDENING_PLAN.md`:
  - atomic JSON/report/manifest writes and rollback-capable directory promotion,
  - raw-capture overwrite refusal and validated paired frame writes,
  - process-group capture cancellation plus hierarchical job resource locks,
  - transactional `sync_report.v2` / `sync_quality_report.v2` with timestamp
    provenance and source-fallback gates,
  - stricter finite/unique calibration validation and merge-preserving
    promotion,
  - importable transactional BlenderProc preparation/render orchestration,
  - standard transactional BOP-scenewise export with
    `bop_export_manifest.v2`, root dataset/frame metadata, cross-file
    validation, depth-aware GT info, and exact model diameters,
  - a testable `zed_2i_capture_summary.v1` adapter,
  - web path containment, strict booleans, scoped pipeline paths, and anchored
    job storage,
  - installable web assets/entry point, Ruff, and legacy/dependency cleanup.

- Implemented `docs/ARUCO_GRID_CALIBRATION_PLAN.md`:
  - exact ArUcoGridGen 1.0 import into `calibration_target.v1`, including board,
    dictionary, contiguous-ID, physical-scale validation, source SHA-256, a
    marker-0 top-left grid frame, and optional aligned identity placement,
  - explicit `robot_flange -> template_base` run-config frames, typed fixed
    transforms, and legacy-frame warnings,
  - split native-RGB detection and grid-to-camera PnP/LM pose phases with
    `aruco_detections.json` and enriched pose provenance,
  - `intrinsic_calibration.v1` factory/calibrated color profiles with 15-view,
    6/9-coverage, 3 px per-view, and 1.5 px RMS default gates plus alpha=0
    rectified projections and unchanged SDK depth provenance,
  - `calibration.v2` profiles with native/rectified projections, explicit
    camera-to-flange or camera-to-template-base endpoints, and v1 loading,
  - unknown-target, known-target, and comparison extrinsic modes, static-mode
    observability rejection, cross-method gates/override recording, and
    fixed-edge camera-to-TCP derivation,
  - repeatable per-sensor profile selection with exactly-one promotion,
  - transactional RGB/nearest-depth rectification under
    `processed/rectified/`, with metadata/pose preservation and strict profile
    identity matching,
  - rectified BlenderProc/BOP consumption and projection provenance in
    `scene_camera.json`,
  - queued target, detection, intrinsic, pose, solver/comparison, selection,
    and rectification stages plus `aruco_grid_full_calibration` and
    `calibrated_capture_to_bop_dataset_dry_run` sequences,
  - synthetic recovery, rejection-gate, selection, legacy-loading,
    rectification, pipeline, and BOP integration tests.

- Removed downstream-only packages and scripts:
  - `posetestbot/estimation`
  - `posetestbot/evaluation`
  - root `evaluation/`
  - estimator stage/wrapper scripts
  - BOP result export/evaluation scripts
  - synthetic BOP result script
  - metric export and legacy evaluator scripts
- Trimmed artifact constants to active acquisition/BOP dataset contracts.
- Reduced runtime status to acquisition dependencies:
  - BlenderProc executable,
  - Stereolabs ZED SDK Python module.
- Updated preflight so optional missing runtimes warn unless a selected
  non-dry-run acquisition stage requires them.
- Removed downstream stage IDs:
  - `foundationpose`
  - `megapose`
  - `sam6d`
  - `bop_result_export`
  - `synthetic_bop_results`
  - `bop_evaluation`
  - `metric_report_export`
- Added/kept acquisition sequences:
  - `capture_to_bop_dataset_dry_run`
  - `fake_capture_to_bop_dataset_dry_run`
  - calibration and sync-to-BOP sequences
  - real full capture validation
- Replaced rewrite gates with acquisition-only gates:
  - `rewrite_fake_acquisition_to_bop.v1`
  - `rewrite_full_capture.v1`
  - `rewrite_calibration_validation.v1`
  - `rewrite_bop_export_readiness.v1`
- Reworked recommendations to suggest acquisition steps only.
- Reworked artifact browser to list/preview acquisition artifacts and inspect
  BOP scene/frame data, GT, masks, and provenance.
- Removed Flask metric dashboard and BOP result CSV endpoints.
- Added per-RealSense inverted-mount capture support that rotates saved RGB-D
  frames 180 degrees, corrects intrinsics, and carries orientation metadata
  through run configs, capture plans, smoke reports, manifests, and the web UI.
- Moved the Flask UI behind `posetestbot.web`, with `web_interface.py` kept as
  a compatibility shim, and added a sidebar workflow overview backed by
  run artifacts, recommendations, jobs, and sequence plans.
- Added sidebar manual IIWA Start/Stop controls that queue non-blocking
  real-robot command jobs with an operator-set robot IP/port target.
- Added an auto-starting sidebar monitor for the UGREEN USB webcam
  (`0c45:2283`) below the IIWA controls. It resolves the current V4L2 node by
  USB identity and captures through a resource-declared background job.
- Changed sensor status to detection-first by default, while preserving
  explicit expected-count checks for CLI/preflight use.
- Added ignored lab-local sensor aliases in `working_data/sensor_aliases.json`
  and queued live RGB preview streams under `working_data/sensor_previews/`.
- Changed live RGB preview controls to card-local stream slots with per-sensor
  toggles, terminal error retention, selected-node/frame metadata, and current
  RealSense inverted orientation restart behavior.
- Added Playwright browser coverage for the sensor preview DOM workflow while
  keeping browser binary installation opt-in.
- Polished the transition web UI empty-run overview state and sidebar branding.
- Reconciled RealSense SDK serials with USB/V4L2 node metadata so three
  connected D435-class cameras appear as three devices, not duplicated SDK and
  USB entries.
- Completed a supervised real-iiwa hot capture with all three lab RealSense
  cameras and passed `rewrite_full_capture.v1` (10/10 checks). The run preserved
  7,499 robot poses and balanced RGB/depth/metadata counts for every sensor.
- Added a disabled-by-default Sunrise calibration-variance proposal with a
  3x3 Cartesian coverage raster, depth sweep, and three-axis orientation
  dither, plus commissioning and image-coverage acceptance guidance based on
  the latest real-capture diagnostics.
- Made RealSense `SIGTERM` shutdown graceful so the capture supervisor cannot
  interrupt an RGB-D tuple between image and metadata commits; control-flow
  exceptions now also roll back partially committed frame files.
- Fixed sync-quality report discovery for relative run roots, which previously
  prepended the run folder twice after successfully discovering reports.
- Added a conservative installation script and `INSTALL.md` covering uv setup,
  optional BlenderProc, vendor SDK caveats, and acquisition readiness checks.
- Updated root agent notes, README, and system overview for the acquisition
  boundary.
- Rewrote stale downstream tests into acquisition-only coverage.

## Current Gates

### `rewrite_fake_acquisition_to_bop.v1`

Requires:

- valid `run_config.json`,
- acceptable `run_preflight_report.json`,
- succeeded fake `capture_execution_report.json` with raw poses,
- succeeded `synthetic_rgbd_report.json`,
- acceptable `sync_quality_report.json`,
- structural BOP export with scene RGB/depth, `scene_camera.json`,
  `scene_gt.json`, an explicit target file (empty is acceptable for structural
  no-GT smoke data), and model metadata.

### `rewrite_full_capture.v1`

Requires:

- real robot `run_config.json`,
- acceptable run preflight,
- run-scoped hardware status selecting the real robot profile,
- capture plan and preflight,
- full capture execution plan with camera and robot receiver roles,
- supervised full capture report,
- raw RGB-D sensor folders with metadata.

### `rewrite_calibration_validation.v1`

Requires:

- `calibration_validation_report.json` with `overall_status=ok`,
- explicit promotion requested and completed,
- promoted `calibration_profiles.json`,
- valid profiles with inlier counts and residual quality fields.

### `rewrite_bop_export_readiness.v1`

Requires:

- validated `bop_export_manifest.v2` using `bop/<split>/<scene_id>`,
- each scene folder containing exact RGB/depth sets, `scene_camera.json`,
  `scene_gt.json`, and `scene_gt_info.json`,
- root `dataset_info.json` and `posetestbot_bop_frame_map.json`,
- `bop/test_targets_bop19.json` with at least one target row,
- `bop/models/models_info.json` with positive exact diameters,
- valid calibration provenance and correct target/model/scene references.

## Validation Commands

Targeted acquisition tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_runtime_status.py \
  tests/test_hardware_status.py \
  tests/test_manifest.py \
  tests/test_pipeline_stages.py \
  tests/test_pipeline_sequences.py \
  tests/test_preflight.py \
  tests/test_rewrite_gate.py \
  tests/test_artifact_browser.py \
  tests/test_pipeline_recommendations.py \
  tests/test_web_interface.py \
  tests/test_web_preview_playwright.py
```

Full validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
git diff --check
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_fake_bop_smoke --overwrite
uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_fake_bop_smoke \
  --gate rewrite_fake_acquisition_to_bop.v1 --write
```

## Remaining Work

- Promote robust calibration profiles from real observations.
- Run BOP export readiness gates on real captured/calibrated datasets.
- Keep improving live capture telemetry and operator ergonomics in the
  transition web UI.
