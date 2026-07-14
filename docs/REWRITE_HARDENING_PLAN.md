# Comprehensive Rewrite Hardening Plan

> Historical design record. This plan is complete; references to the retired
> fake-acquisition validation path describe the repository at that time and are
> not current operator instructions.

## Summary

The review covers the 183-file rewrite diff, static analysis, packaging, the
full test suite, installer checks, Playwright, and the fake
acquisition-to-BOP smoke/gate. The current workflow broadly works, but the
audit found data-integrity risks in synchronization/export, orphan-process
risk during cancellation, incomplete ZED coverage, non-standard BOP layout,
unsafe legacy scripts, weak validation, and a broken installed-wheel web UI.

Implement the fixes below while preserving the acquisition-only boundary and
fake-first behavior.

## Implementation Status

Completed on 2026-07-10. All four implementation sections below are in the
working tree. Hardware-free validation passed with 269 tests, Ruff, diff
checks, installer check-only validation, Playwright, isolated wheel/UI/asset
smoke, and the fake acquisition-to-BOP smoke plus
`rewrite_fake_acquisition_to_bop.v1` at 11/11 ready.

The intentionally unexecuted items are the physical-lab gates: real iiwa/full
camera capture, promotion of calibration derived from real observations, and
BOP readiness on a real calibrated dataset. No real robot command or physical
camera capture was run during this hardening pass.

## Implementation Changes

### 1. Acquisition, Synchronization, and Job Safety

- Add same-directory atomic JSON writing using temporary files, flush/fsync,
  and `os.replace`; use it for manifests, reports, job records, run
  configuration, calibration promotion, BOP metadata, and final raw-pose
  output.
- Harden frame writing:
  - Validate RGB/depth dimensions, channels, dtypes, and numeric frame stems.
  - Refuse to overwrite existing frames or sidecars.
  - Write image pairs through temporary files and remove newly created partial
    pairs if metadata append fails.
- Make capture preflight fail when a planned raw sensor folder is non-empty or
  `raw_robot_ee_poses.json` already exists. Do not add a raw overwrite option;
  operators must use a new run root.
- Make capture supervision handle SIGINT/SIGTERM, terminate every spawned
  process group, close logs, and persist `canceled` or `failed` status/report
  state before exiting.
- Add job status `canceling`; retain resource locks until the process group has
  actually exited. Add runner shutdown/join support and atomic persistence.
- Make resources hierarchical: `camera` conflicts with all `camera:*`, while
  `camera:<sensor_type>:<device_id>` conflicts only with its ancestors or the
  same resource. Preview and snapshot jobs use device-specific locks; full
  capture remains globally locked.
- Emit `sync_report.v2` and `sync_quality_report.v2`:
  - Build each synchronized sensor folder in staging and promote only after
    validation, eliminating stale frames on rerun.
  - Generate a synchronized `frame_metadata.jsonl` whose IDs and paths match
    renamed frames while retaining source-frame provenance.
  - Record requested and actual timestamp sources, source counts, and
    fallbacks; a required-source fallback becomes a quality error.
  - Validate metadata paths remain under the raw sensor folder, frame pairs
    exist, robot poses are non-empty, and timestamps are usable.
  - Match against ordered contiguous robot-motion intervals so repeated motion
    labels cannot bridge unrelated time ranges.
  - Continue reading v1 reports for existing-run browsing, but mark their
    timestamp provenance unaudited.

### 2. Calibration, BlenderProc, and BOP Correctness

- Strengthen `calibration.v1` validation for finite numbers, positive depth
  scale, normalized quaternion, valid intrinsic matrix, nonnegative quality
  metrics, inlier bounds, and collection uniqueness. Zero legacy intrinsics
  remain allowed only for explicitly migrated `needs_validation` profiles.
- Require `valid` profiles whenever calibrated BlenderProc or BOP export is
  requested; reject missing, ambiguous, deprecated, failed, or unvalidated
  matches.
- Make promotion merge with an existing profile collection, replacing only the
  same sensor/mount/rig slot and preserving unrelated profiles. Validate and
  atomically promote the merged collection.
- Replace the `sys.argv`-driven BlenderProc preparation script with importable
  functions and ordinary exceptions. Stage preparation and render output,
  validate expected camera/GT/mask frame counts, and promote all sensor outputs
  only after every job succeeds.
- Validate render scripts and safe single-component subdirectory names during
  dry runs; failed renders leave the prior derived output intact.
- Emit standard BOP-scenewise structure at `bop/<split>/<scene_id>`, assigning
  one unique scene per sensor. This replaces
  `bop/<sensor>/<split>/<scene_id>` and follows the
  [official BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).
- Emit `bop_export_manifest.v2` with portable paths, split/scene-to-sensor
  provenance, object-ID mapping, and validation results. Write:
  - Root `dataset_info.json` with PoseTestBot schema, dataset name, format,
    split, sensors, scene count, and generation time.
  - Root `posetestbot_bop_frame_map.json` containing scene/image source
    provenance.
  - Standard scene RGB, uint16 depth, camera, GT, GT-info, and optional mask
    directories.
- Make the entire BOP dataset export transactional. `--overwrite` replaces the
  complete validated dataset, never individual scenes, and a failed overwrite
  preserves the previous export.
- Require exact RGB/depth name sets, matching dimensions, valid
  intrinsics/depth scale, aligned scene JSON keys, known object IDs, valid
  target references, and correctly named masks.
- Recompute `scene_gt_info` from exported masks and actual depth;
  `px_count_valid` counts object-mask pixels with nonzero depth.
- Compute true BOP model diameter from convex-hull vertex pairs in bounded
  chunks, cache it by model SHA-256, and remove the inaccurate AABB-diagonal
  fallback.
- Strengthen rewrite gates: the fake gate may retain empty structural GT, while
  BOP readiness requires the v2 standard layout, validated scenes, non-empty
  targets/models, correct cross-file references, and valid calibration
  provenance.

### 3. Sensors and Web API

- Add a testable `posetestbot.sensors.zed_2i` adapter with injected SDK support,
  argument validation, resolved camera serial, consistent sidecars/timestamps,
  error handling, cleanup, and `zed_2i_capture_summary.v1`. Keep the script as
  a thin CLI wrapper.
- Isolate RealSense discovery tests from actual USB/V4L devices so the suite is
  deterministic on the lab host.
- Retain the requested unauthenticated `0.0.0.0` web default and manual
  real-robot controls, but constrain filesystem access:
  - Web run roots default to `<repo>/working_data`; additional roots use the
    `POSETESTBOT_WEB_RUN_ROOTS` path-list environment variable.
  - Read-only external inputs default to `object_models` and
    `scripts/default_data`; additions use `POSETESTBOT_WEB_INPUT_ROOTS`.
  - Resolve symlinks and reject traversal. Output path parameters must remain
    below the selected run root.
  - CLI tools retain arbitrary explicit paths.
- Annotate pipeline path parameters with input/output/repository scope so stage
  and sequence submissions receive the same validation.
- Replace Python truthiness conversions with strict boolean parsing for JSON
  booleans and recognized true/false strings.
- Anchor job storage to the repository root, remove the duplicate sensor-status
  route, and remove the superseded `realsense_multi` web command.

### 4. Cleanup, Packaging, and Documentation

- Remove the selected legacy implementations: `realsense_multi.py`, destructive
  capture/sync wrappers, the shell-based BlenderProc wrapper, superseded
  transform scripts, the migrated BlenderProc prep script, and
  `scripts/ROI_generation/`.
- Remove the empty tracked `test` file and drop now-unused `requests` and
  `pandas` dependencies through `uv`; update the lock file through the tool.
- Add Ruff as a dev dependency, fix current lint findings and whitespace errors
  without applying a repository-wide formatting rewrite.
- Fix wheel packaging by excluding `tests`, including templates/static assets,
  moving the logo into packaged static data, adding a meaningful project
  description, and exposing a `posetestbot-web` entry point. Keep
  `web_interface.py` as the source-checkout compatibility shim.
- Update README, installation guidance, system overview, and
  `docs/REWRITE_PROGRESS.md` with the v2 artifact contracts, BOP migration, web
  path configuration and LAN trust warning, removed-script replacements,
  completed audit work, and remaining physical-lab gates.

## Public Interfaces and Compatibility

- New writer schemas are `sync_report.v2`, `sync_quality_report.v2`,
  `bop_export_manifest.v2`, root `posetestbot_bop_frame_map.v2`, and
  `zed_2i_capture_summary.v1`.
- Readers continue to support v1 sync/BOP artifacts for browsing, but readiness
  gates require newly generated v2 evidence.
- `frame_metadata.v1`, `calibration.v1`, dataset-manifest schemas, and rewrite
  gate IDs remain stable; validation becomes stricter and synchronized metadata
  gains additive provenance fields.
- Job APIs gain the non-terminal `canceling` status and hierarchical resource
  names.
- `--overwrite` for BOP export now means atomic replacement of the complete BOP
  dataset. No legacy sensor-nested layout or compatibility copy is produced.
- Removed scripts have no stubs. Supported replacements are documented in the
  README and rewrite progress file.

## Test Plan

- Add regressions for synchronized reruns with fewer frames, metadata/path
  provenance, timestamp fallbacks, repeated motion labels, malformed inputs,
  raw overwrite refusal, and atomic-failure preservation.
- Test cancellation with a detached fake child process, `canceling` lock
  retention, hierarchical camera conflicts, clean shutdown, and persisted-job
  recovery.
- Test standard multi-sensor BOP paths, root frame map, exact diameter,
  depth-aware GT info, invalid profile rejection, cross-file validation, and
  failed-overwrite rollback.
- Test calibration NaN/quaternion/quality/duplicate rejection, legacy migration
  allowance, and merge-preserving promotion.
- Test BlenderProc staging/rollback and frame-count validation without
  requiring BlenderProc execution.
- Test ZED capture with a fake SDK, deterministic RealSense discovery, strict
  web booleans/path traversal, scoped stage paths, and installed-wheel UI
  rendering.
- Run:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`
  - Existing Playwright tests without installing new browser binaries.
  - `git diff --check`
  - Installer syntax/help/check-only validation.
  - `uv build` plus isolated wheel install and web asset smoke.
  - Fake acquisition-to-BOP smoke and
    `rewrite_fake_acquisition_to_bop.v1`.

## Assumptions

- The BOP path change is intentionally breaking; no legacy directory copy or
  compatibility writer will be added.
- Unsafe legacy scripts are deleted rather than retained as stubs.
- The web server remains LAN-accessible and unauthenticated by explicit choice;
  path containment reduces filesystem exposure but the UI must still be
  treated as trusted-lab-only.
- Object models are supplied in millimetres, matching BOP requirements.
- No physical camera capture or real-robot command will be run during
  implementation. Read-only status checks are allowed; real full-capture,
  calibration, and BOP-readiness gates remain documented operator tasks.
