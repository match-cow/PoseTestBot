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
- This host keeps GitHub CLI credentials in the user keyring. A failed
  `gh auth status` inside the sandbox can be a sandbox/keyring visibility false
  negative. Before reporting that GitHub authentication is invalid, rerun the
  same read-only authentication check outside the sandbox; do not ask the
  operator to log in again based only on the sandboxed result.
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
- During repeated calibration, never send the iiwa UDP `STOP` command. It
  cannot interrupt active motion and exits the waiting calibration program,
  requiring a manual Sunrise application restart.
- Do not add blocking request handlers for long-running or hardware-touching
  work. Queue them through `posetestbot.jobs.runner.LocalJobRunner` and declare
  resources.
- Preserve raw capture data. Synchronization/export work should create derived
  artifacts, usually under `processed/`, rather than renaming or deleting the
  only copy of frames.
- Keep completed status current in `docs/REWRITE_PROGRESS.md` and unfinished
  work in `docs/REWRITE_REMAINING_WORK.md`.
- A name containing `legacy` does not by itself make code removable. Keep the
  compatibility readers and entry points named in the remaining-work plan until
  that plan records a migration and sunset decision.
- Before deleting or renaming a tracked file, search production code, tests,
  docs, packaging manifests, and installer checks for references. Rebuild the
  checked-in frontend with Vite; never hand-edit or selectively retain hashed
  files below `posetestbot/web/static/ui/assets/`.

## Web Interface Design Policy

- The operator console is a desktop-first, information-dense interface for
  supervised lab work. Design and review the primary composition at
  1920 x 1080 and 100% browser zoom; use 1440 x 900 as the minimum normal
  desktop check. The persistent application sidebar, workflow step rail, and
  side-by-side configuration, preview, and evidence panes are the canonical
  experience.
- Prioritize desktop clarity and useful information density. Do not reduce,
  hide, or aggressively stack technical evidence, comparisons, provenance,
  validation results, or required controls merely to make every view resemble
  a phone layout. Do not add phone-specific navigation or touch-first
  interaction unless the operator explicitly requests it.
- Widths below the normal desktop target are best-effort fallbacks, not a
  mobile-support commitment. Navigation, dialogs, safety acknowledgements, and
  primary actions must remain reachable and must not overlap; inherently wide
  tables, matrices, timelines, canvases, and steppers may use explicit local
  scrolling. Prefer local overflow over accidental document-wide overflow, and
  never hide safety state or required actions to accommodate a narrow viewport.
- Prioritize Playwright coverage at desktop viewports. Use narrower viewports
  only for a named reachability, overflow, browser-zoom, safety-control, or
  specifically reported regression contract; mobile visual polish and feature
  parity are not release gates.
- Hover explanations may take advantage of mouse-oriented desktop use, but
  they must also be available by keyboard focus or click. Required and
  safety-critical information must never exist only inside a tooltip.

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
- `posetestbot.pose_templates.catalog` as the JSON-backed Workpiece Catalogue
  persistence, identity, lifecycle, and metadata portability contract.
- The remaining `posetestbot.pose_templates.*` exact slicing, immutable bundle,
  run-selection, and object-instance preparation contracts.
- `scripts/run_bop_export_stage.py` and `posetestbot.bop.writer`.
- Flask operator APIs for jobs, capture status, hardware/sensor/runtime
  status, run config, preflight, calibration, the `/workpieces` catalogue,
  sync quality, and pipeline sequence submission.

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
- Run/config-bound physical depth-sync qualification:
  `hardware_sync_qualification.json`, with copied evidence below
  `hardware_sync_qualification_evidence/`.
- Capture artifacts: `capture_plan.json`,
  `capture_plan_preflight_report.json`, `capture_execution_plan.json`,
  `capture_execution_status.json`, `capture_execution_report.json`,
  and `capture_execution_logs/`.
- Derived sync report: `sync_report.json`.
- Run-level sync quality report: `sync_quality_report.json`.
- Authoritative complete mixed-mount hardware-sync groups:
  `processed/synchronized/multiview_frame_groups.json`.
- Calibration artifacts: `calibration_preflight_report.json`,
  `calibration_target.json`, `intrinsic_calibration_profiles.json`,
  attempt-level `intrinsic_comparison.json`,
  per-sensor `aruco_detections.json`, `camera_rectification_report.json`,
  `calibration_observations.json`, `calibration_candidates.json`,
  `calibration_profiles_from_observations.json`,
  `calibration_solver_report.json`, `calibration_profiles_solved.json`,
  `calibration_validation_report.json`, and promoted
  `calibration_profiles.json` (`calibration.v2`; v1 remains loadable).
- Run-owned reusable-calibration selection is recorded in
  `calibration_profile_selection.json`. Exact copied
  `calibration_profiles.json` and `intrinsic_calibration_profiles.json`
  snapshots live below `processed/calibration_inputs/<bundle_sha256>/`; the
  selection manifest binds their hashes and per-sensor profile mapping so a
  later source-run change cannot alter the dataset run.
- Intent-level calibration attempts live under
  `processed/calibration/<attempt_id>/` and retain `request.json`,
  `progress.json`, `intrinsic_comparison.json`, `time_offset_search.json`,
  `pnp_candidates.json`, `extrinsic_candidates.json`, `ranking.json`,
  `checks.json`, `candidate_profiles.json`, the selected target bundle, and
  explicit promotion evidence.
- BlenderProc render plan artifact: `blenderproc_render_plan.json`.
- Workpiece Catalogue artifacts: global
  `object_catalog/object_catalog.json`, retained UUID-addressed assets below
  `object_catalog/objects/<uuid>/`, canonical geometry revisions and derived
  `pose_template_orientation_analysis.json` and bounded
  `pose_template_orientation_thumbnail.json` caches below each object's
  `derived/` directory, numbered manifest snapshots below
  `object_catalog/revisions/`, and deletion tombstones in the catalog JSON.
- Pose-template artifacts: global immutable
  `pose_templates/<uuid>/pose_template_bundle.json`, exact
  `pose_template_preview.json`, bounded `pose_template_thumbnail.json`,
  run-owned `pose_template_selection.json`, its hidden durable
  `.pose_template_selection.transaction.json` journal while replacement is in
  progress, and `object_instances.json`.
- BOP export artifacts: `bop/bop_export_manifest.json`,
  `bop/posetestbot_bop_frame_map.json`, `bop/posetestbot_frame_sets.json`,
  `bop/test_targets_bop19.json`,
  `bop/models/models_info.json`, pose-template
  `bop/posetestbot_pose_template.json` and `bop/posetestbot_instance_map.json`, optional
  `bop/posetestbot_multiview_targets.json`, and optional
  `bop/posetestbot_coco_annotations.json`.

## Workpiece Catalogue Contracts

The persistent catalogue root is normally `working_data/object_catalog/`.
`object_catalog.v1` retains stable UUID and BOP `obj_id` identity while adding
editable `name`, `alias`, `description`, `tags`, `groups`, and `attributes`
metadata. Source CAD, canonical PLY, and optional PNG texture assets live in
each workpiece's UUID directory and are referenced by catalog-relative path,
size, and SHA-256.

Serialize every catalogue mutation across threads and processes, write an
atomic numbered revision before replacing the current manifest, and never
reuse a UUID or BOP `obj_id`. Archive is reversible. Permanent deletion is
allowed only for an archived workpiece after explicit confirmation and only
when no pose-template bundle references it. Fail closed if any published
bundle cannot be validated, serialize bundle publication with catalogue
deletion, commit the tombstone before removing assets, and retain the
tombstone. Record asset-cleanup status and errors in that tombstone; a repeated
confirmed delete of the retired UUID must safely retry pending cleanup.

Workpiece JSON export/import is metadata-only. The JSON does not embed CAD,
canonical PLY, or texture bytes, and import updates matching local UUIDs while
reporting records whose managed assets are absent as skipped. Preserve or move
the complete managed asset tree separately when binary portability is needed.
Queue CAD inspection/conversion through `LocalJobRunner`; it is CPU/disk work
and must not open cameras or command the robot. The legacy
`/pose-templates/catalog` APIs remain supported for compatibility, while new
operator work belongs under `/workpieces`.

Treat metre/millimetre correction as a new canonical geometry revision. It
requires an archived workpiece, explicit confirmation/operator provenance, and
an expected revision/hash compare-and-swap. Regenerate from the retained source
at the cumulative source-to-mm scale, preserve every earlier canonical version,
and never rewrite existing pose-template or run snapshots. Stable-orientation
analysis is a reproducible cache bound to the canonical hash and implementation
revision; its compact thumbnail is a separately bounded card-read cache with
the same provenance binding. Do not record either mutable cache as an immutable
catalogue asset.

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

`run_config.v3` owns the explicit `capture.synchronization` contract.
`timestamp_aligned` remains the general default. The only supported
`hardware_trigger` implementation on the current lab inventory is
`realsense_inter_cam_sync` with `scope=depth_exposure`, across exact-ID D435
cameras that include at least one `static` and one `eye_in_hand` view. Exactly
one is the master and the others are subordinates. This does not certify D435
RGB exposure synchronization. The USB OAK-D Pro and USB ZED 2i cannot join that
trigger group; reject such configurations instead of silently falling back.
Hardware-trigger capture and synchronization require a current
`hardware_sync_qualification.json` produced from operator-confirmed external
exposure-timing evidence. The recorder must never open cameras or contact the
robot. Changing the resolution, FPS, trigger policy, camera membership, mount,
orientation, or role invalidates qualification. Publish or replace
qualification only before acquisition starts; once capture status/report/logs,
raw camera data, or raw robot-pose evidence exists, it is immutable and a
different qualification requires a new run.
During supervised capture, require append-only camera metadata progress using
the independent default of 12 planned frame periods clamped to 2–5 seconds,
regardless of the robot UDP timeout. Preserve partial raw evidence on failure.
A successful hardware-sync capture report must bind the exact configuration
and qualification hashes after immediate pre-receiver revalidation. Carry that
binding through authoritative groups and BOP frame sets, and require the BOP
rewrite gate to compare it with the capture report and current qualification.
Preserve early and incomplete raw frames, and treat only the complete groups in
`processed/synchronized/multiview_frame_groups.json` as authoritative combined
views.

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
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium  # only if browser binaries are missing
```

The default pytest selection excludes the explicitly marked Playwright modules
so a normal `uv sync --all-groups` checkout does not require optional Chromium.
Keep each test tied to a distinct production contract, public boundary, or
failure mode; consolidate cases whose setup and assertions are strictly
subsumed by stronger coverage. A browser screenshot is regression evidence only
when it has a golden/pixel comparison or meaningful UI assertions—successfully
writing an image and checking its dimensions is not sufficient.
