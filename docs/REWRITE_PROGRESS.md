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
- RealSense, OAK-D Pro, and ZED 2i sensor adapters and status contracts, with
  run-scoped enable/disable selection that preserves disabled camera metadata;
- transactional synchronization and sync-quality reporting;
- PoseGridGen targets and attempt-scoped factory-vs-OpenCV intrinsic evidence,
  whole-board PnP support gates, global-sensor-time RealSense synchronization,
  common-bundle multi-camera extrinsic ranking, and explicit promotion;
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
profile, three-RealSense service/full-capture acceptance, and real BOP v3
export still require operator-run acceptance. The reduced two-camera
calibration described below is promoted, but it does not satisfy those broader
milestones. That historical run used an older 10 × 7 / 70-marker board and
insufficient hand-eye motion diversity; it is a preserved negative baseline,
not `calib00` calibration evidence.

The current `calib00` campaign completed physical acquisition from two enabled
RealSense cameras while retaining the temporarily unavailable third camera as
disabled run configuration. That reduced camera set is valid for this
calibration attempt but does not satisfy the separate three-RealSense or
five-sensor service/full-capture milestones. Immutable attempt
`3c4a0b7b765f44bd9cc37fffc48fb321` promoted the complete common
`IPPE + Horaud` bundle for the two enabled cameras. It retained factory color
intrinsics and the bounded manual OpenCV comparisons, produced 652/652 and
655/656 extrinsic inliers, held-out means of 3.129 mm / 0.491 degrees and
3.332 mm / 0.427 degrees, and 3.612 mm / 0.277 degrees pairwise companion
closure. Both exact camera-to-flange and grid-to-template-base estimates are
available in the promoted profiles and Cell scene. Factory SDK depth scale and
depth-to-color alignment were not recalibrated; a range-dependent metric-depth
anomaly on `923322072633` remains a separate validation item.
The run-level rewrite status is 2/3 gates and 12/17 checks ready. Its only
blocked gate is the expected BOP-export gate because this calibration run did
not produce a BOP dataset; full capture is 9/9 ready and calibration validation
is 3/3 ready.

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
- Reduced the iiwa calibration capture profile to 60% of requested Cartesian
  speed (8–45 mm/s), lowered repositioning and central orientation speeds,
  applied 3% acceleration/jerk limits to every motion, and added a 1.5-second
  vibration dwell after every leg.
- Verified the pinned PoseGridGen and PoseTemplateCreator checkouts and
  packaged pose-template/backend/UI contents.
- Retired the static object registry, bundled sample models, legacy run-setup
  selector, Cell registry preview/assets, and BlenderProc/BOP fallback paths;
  object-bearing runs now flow only through immutable pose-template bundles.
- Hardened every production IIWA START entry point with fresh robot-and-camera
  acknowledgements, made the UDP pose receiver refuse prior raw evidence before
  network I/O, added finite first-packet/inter-packet timeouts with terminal
  manifest states and unique partial evidence, and kept reusable sequence and
  capture plans free of execution authorization.
- Hardened eye-in-hand attempts so RealSense SDK `global_time` sensor exposure
  timestamps pair with robot host-wall timestamps at zero manual offset, with
  no fallback and a 20 ms maximum nearest-pose delta. Added spatial/campaign
  target support, rotation-axis rank checks before and after pruning,
  per-motion balanced fitting, and full-input validation.
- Made `inverse_brown_conrady` forward-OpenCV-compatible only for finite,
  exact-zero coefficients. Compatible factory projection remains selected;
  factory/manual comparison and rejection evidence remains immutable, and a
  gated manual profile is required when factory projection is unusable.
- Added deterministic multi-camera ranking over complete same-PnP/same-
  extrinsic bundles. Every individual candidate must pass, pairwise stationary
  companion closure must remain within 10 mm / 5°, and bundles within 0.01 of
  the best normalized mean individual score are ordered by normalized closure.
  Six-decimal normalized comparison suppresses physically meaningless solver
  dust before canonical method tie-breaking. Ranking/promotion fail closed when
  no complete common bundle passes.
- Added a Run Setup camera enable control. Disabled cameras retain identity,
  mounting/orientation metadata, and profile selection but are excluded from
  capture/preflight, calibration, Cell, and rewrite-gate expectations.
- Distinguished physically detected sensors from SDK-addressable,
  capture-ready sensors. USB-descriptor-only RealSense records remain visible
  as diagnostic evidence but no longer satisfy expected counts, preflight, or
  hardware-snapshot selection. SDK-enumerated RealSense devices with a known
  USB major version below 3 are likewise blocked before capture; status reports
  the affected serial and transport descriptor. Optional SDK-recommended
  firmware metadata is retained as warning-only troubleshooting evidence and
  never drives an automatic firmware change.
- During a pre-fix web-route diagnostic, a string-valued gate request
  accidentally queued possible IIWA START job `0a4ec1902719`. Its local
  workload returned code 1 and retained no send confirmation, so delivery is
  unverified and more likely did not occur. It produced no camera frames or raw
  robot-pose artifact, and no `STOP` was sent. The route now rejects non-boolean
  execution gates before normalization.

Software validation completed. The two-camera physical `calib00` capture now
has an explicitly promoted common bundle and passing calibration-validation
evidence:

- 564 non-browser pytest tests, 15 operator-console Playwright tests, and 12
  preview Playwright tests passed (591 total);
- Ruff, frontend type checking, frontend lint, the production frontend build,
  and `git diff --check` passed; and
- the installer check-only path, wheel/sdist build, packaged-asset audit, and
  installed-wheel Flask smoke passed. The production build's approximately
  937 kB lazy Cell chunk remains the optional P5 performance item.

## Remaining Work

All unfinished tasks, dependencies, safety constraints, and exit criteria now
live in [REWRITE_REMAINING_WORK.md](REWRITE_REMAINING_WORK.md). Keep that file
and this short status snapshot current; completed plan documents are available
through Git history.
