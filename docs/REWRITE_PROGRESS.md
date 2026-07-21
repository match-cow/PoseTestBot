# Rewrite Progress

Last updated: 2026-07-21

PoseTestBot is acquisition-first. Its repository boundary is real capture,
calibration, non-destructive synchronization, optional GT/mask generation,
pose-template provenance, and BOP dataset export. Downstream estimator and
evaluation work is excluded.

## Current State

The code rewrite is implemented across:

- real-only capture planning, preflight, supervised execution, and raw-evidence
  protection;
- RealSense, OAK-D Pro, and ZED 2i sensor adapters and status contracts;
- transactional synchronization and sync-quality reporting;
- PoseGridGen targets and attempt-scoped intrinsic/extrinsic calibration with
  explicit promotion;
- PoseTemplateCreator catalog, immutable printable templates, run placement,
  per-instance GT, and Cell provenance;
- BlenderProc 2.8.0 preparation/render identity checks and transactional BOP
  v3 export; and
- the packaged React operator console, managed jobs/services, and scoped Flask
  APIs.

The active gates are:

- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

Historical run `working_data/hot_full_capture_fixed_20260710_1351` passes the
full-capture gate at 10/10 for three RealSense cameras. The current five-sensor
profile, real calibration promotion, and real BOP v3 export still require
operator-run acceptance.

## 2026-07-21 Audit and Cleanup

- Confirmed that no estimator, evaluator, BOP-result conversion, or metric
  implementation remains in tracked production code.
- Removed obsolete duplicate launch/ArUco/Sunrise files, definition-only
  helpers/constants, stale completed plans, generated build debris, and the
  misleading downstream-compatibility test name.
- Fixed calibration-workflow lint and run-switch state leakage.
- Required both execution gates for manual IIWA start requests while leaving
  Stop available without motion-start gates.
- Made live-preview capability explicit and disabled unsupported ZED preview
  controls before a doomed background job can be queued.
- Verified the pinned PoseGridGen and PoseTemplateCreator checkouts and
  packaged pose-template/backend/UI contents.

Validation completed without camera or robot access:

- 438 non-browser pytest tests, 14 operator-console Playwright tests, and 11
  preview Playwright tests passed (463 total);
- Ruff, frontend type checking, frontend lint, the production frontend build,
  and `git diff --check` passed; and
- the installer check-only path, wheel/sdist build, packaged-asset audit, and
  installed-wheel Flask smoke passed. The production build's approximately
  932 kB lazy Cell chunk remains the optional P5 performance item.

## Remaining Work

All unfinished tasks, dependencies, safety constraints, and exit criteria now
live in [REWRITE_REMAINING_WORK.md](REWRITE_REMAINING_WORK.md). Keep that file
and this short status snapshot current; completed plan documents are available
through Git history.
