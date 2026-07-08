# Rewrite Progress

Last updated: 2026-07-08

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

- `bop/bop_export_manifest.json` with exported scenes,
- each scene folder containing RGB/depth frames, `scene_camera.json`, and
  `scene_gt.json`,
- `bop/test_targets_bop19.json` with at least one target row,
- `bop/models/models_info.json`.

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
  tests/test_web_interface.py
```

Full validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git diff --check
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_fake_bop_smoke --overwrite
uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_fake_bop_smoke \
  --gate rewrite_fake_acquisition_to_bop.v1 --write
```

## Remaining Work

- Validate full camera capture in the lab with real robot mode intentionally
  selected.
- Promote robust calibration profiles from real observations.
- Run BOP export readiness gates on real captured/calibrated datasets.
- Keep improving live capture telemetry and operator ergonomics in the
  transition web UI.
