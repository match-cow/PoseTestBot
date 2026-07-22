# Rewrite Progress

Last updated: 2026-07-22

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
- PoseGridGen targets, including immutable-library card previews rendered from
  stored marker geometry, and attempt-scoped factory-vs-OpenCV intrinsic
  evidence, whole-board PnP support gates, global-sensor-time RealSense
  synchronization, common-bundle multi-camera extrinsic ranking, and explicit
  promotion;
- the dedicated Workpiece Catalogue page and `/workpieces` API, backed by the
  existing JSON/UUID asset store with editable classification, previews,
  guarded lifecycle, revisioned metre/mm correction, metadata portability, and
  pose-template integration;
- the updated PoseTemplateCreator stable-orientation workflow, with bounded
  isometric previews, exact planar layout/fit validation, immutable printable
  templates sourced from active workpieces, preview-rich run selection,
  per-instance GT, and Cell provenance;
- BlenderProc 2.8.0 preparation/render identity checks and transactional BOP
  v3 export; and
- the packaged React operator console, managed jobs/services, and scoped Flask
  APIs.

The active gates are:

- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

The Cell view now composes the run's actual context surface: exact compensated
pose-template footprint contours for object-bearing runs, the selected or
latest run-local calibration-attempt board for calibration runs, and the
packaged HRI sheet only as a fallback. Promoted board placement is recovered
from calibration-profile companion transforms; boards without promoted
placement remain visibly marked as reference overlays. Optional robot-base and
TCP frames are reported as not configured instead of unresolved, while cameras
still fail closed when this run has no matching promoted profile.

Historical run `working_data/hot_full_capture_fixed_20260710_1351` passes the
full-capture gate at 10/10 for three RealSense cameras. The current five-sensor
profile, combined camera-service lifecycle acceptance, and real BOP v3 export
still require operator-run acceptance. That historical run used an older 10 ×
7 / 70-marker board and insufficient hand-eye motion diversity; it is a
preserved negative baseline, not `calib00` calibration evidence.

The current `calib00` confirmation campaign completed physical acquisition
with all three configured RealSense cameras. Immutable attempt
`12e6a40eff444b889870597b787bf016` promoted the complete common
`IPPE + Shah` bundle. It retained compatible factory color intrinsics and the
bounded manual OpenCV comparisons, produced 605/606, 608/608, and 610/610
extrinsic inliers, held-out means of 3.052 mm / 0.628 degrees, 3.241 mm / 0.473
degrees, and 3.226 mm / 0.425 degrees, and 7.104 mm / 0.421 degrees maximum
pairwise companion closure. Both exact camera-to-flange and
grid-to-template-base estimates are available in the promoted profiles and
Cell scene. The repeat agrees with the first three-camera campaign within
5.400 mm / 0.365 degrees at worst. The retained validation record is
[EYE_IN_HAND_CALIBRATION_VALIDATION_20260722.md](EYE_IN_HAND_CALIBRATION_VALIDATION_20260722.md).
Factory SDK depth scale and depth-to-color alignment were not recalibrated; a
range-dependent metric-depth anomaly on `923322072633` remains a separate
validation item. The run-level rewrite status is 2/3 gates and 13/18 checks
ready. Its only blocked gate is the expected BOP-export gate because this
calibration-only run did not produce a BOP dataset; full capture is 10/10 ready
and calibration validation is 3/3 ready.

## 2026-07-22 Workpiece Catalogue

Implementation of the dedicated **Workpiece Catalogue** feature is complete:

- added the navigation entry and `/workpieces` page below Calibration Targets
  and above Pose Templates, with upload, detail/edit, search, tag/group/state
  filtering, compact isometric and interactive previews, usage evidence,
  archive/restore/delete, revisioned metre/mm correction, and JSON
  import/export;
- retained PLY/STL/OBJ source CAD, canonical PLY, optional PNG texture, hashes,
  and editable `name`, `alias`, `description`, `tags`, `groups`, and
  `attributes` in `working_data/object_catalog/` without adding a database;
- added a selected-object interactive client-side mesh view plus bounded static
  isometric card previews without multiplying WebGL contexts. Catalogue cards
  use a separate at-most-256-KiB, canonical-hash-bound orientation thumbnail;
  only the selected editor path reads ranked orientations and exact contours;
- serialized Flask/worker mutations with cross-process locking and atomic
  numbered revisions, and made permanent deletion require archive, explicit
  confirmation, zero pose-template references, and a fully valid published
  template library while retaining never-reused UUID/BOP-ID tombstones;
- serialized immutable template publication against catalogue deletion, made
  deletion commit its tombstone before removing assets, contained queued-worker
  cleanup to managed request UUID directories, capped streamed multipart and
  JSON requests even when Content-Length is absent, and bounded both persisted
  job logs and per-line API tails. Upload and unit-correction workers clean
  request folders on failure as well as success, and submissions prune stale
  folders older than 24 hours without touching active jobs. Tombstones retain
  retryable asset-cleanup status/error evidence if post-commit removal fails;
- kept JSON portability intentionally metadata-only: it never embeds CAD or
  texture bytes, remains exportable for metadata recovery after asset damage,
  and reports locally absent or corrupt UUID assets as skipped;
- moved new pose-template selection to active workpieces from the same
  catalogue while preserving legacy catalogue APIs and immutable bundle/run
  snapshots. The template editor now filters catalogue metadata, presents
  ranked stable grounded orientations beside exact base contours, supports
  direct planar drag/rotation, and enables generation only for an exact current
  server preview. Library and run selection use hash-verified bounded footprint
  cards (with explicit simplification evidence), while the selected version
  loads its full immutable interactive 3D scene. Pre-thumbnail bundles derive
  the bounded card in memory without mutation. Physical-placement confirmation
  clears after any template or transform change; preview submission retries
  transient resource conflicts and discards stale configuration results. New
  manifests omit duplicate raw contours while the hash-verified exact preview
  retains them. Metadata/card reads are bounded, and preview/PDF/individual-
  asset requests hash only their requested declared artifact; strict whole-
  bundle validation remains mandatory for run selection and catalogue delete;
- updated the PoseTemplateCreator pin from `450747b` to `97ddb9b`, retained old
  bundle/six-DoF draft readability, and records canonical geometry revision,
  unit scale, stable-orientation provenance, and the composed
  `Txy * Rz * source_to_placed` transform in new immutable bundles;
- made unit correction archive/confirmation/operator/CAS-gated, regenerate
  from retained source at the cumulative scale, preserve all canonical
  revisions, tolerate optional stable-orientation cache failure, and leave
  every existing template/run snapshot untouched; and
- made run selection a locked, staged transaction across the copied bundle,
  selection record, and run config, with strict record-to-bundle validation,
  live validation, complete rollback on ordinary promotion failure, and a
  durable journal that rolls back or finishes cleanup after process loss.
  Every production run-config writer shares the same per-run cross-process
  lock, recovery rejects symlinked/non-directory ancestors, and exact orphaned
  selection staging names are pruned without touching unrelated hidden files.
  Selection snapshotting is serialized against archive, all published bundle
  trees reject undeclared files and symlinks, and expensive template analysis,
  slicing, PDF rendering, and asset copying occur outside the short catalogue
  publication lock before exact geometry identities are rechecked.

Repository-wide software validation of this addition completed on 2026-07-22:

- 657 non-browser pytest tests, 17 operator-console Playwright tests, and 12
  preview Playwright tests passed (686 total);
- the Workpiece Catalogue browser coverage exercised one orbitable bounded 3D
  detail canvas plus static isometric cards, metadata validation, filters,
  queued upload, import, archive/restore, confirmed deletion, and reference
  conflicts without fetching the full canonical PLY;
- an additional focused Playwright case forced a transient preview-resource
  409 and passed after the automatic retry path;
- Ruff, frontend type checking, frontend lint, the production frontend build,
  wheel/source packaging, `git diff --check`, shell syntax validation, and the
  installer check-only path passed.

No camera, robot, lab service, or physical capture was accessed during this
feature validation. Network access was limited to reading/fetching the named
GitHub upstream. BlenderProc and `pyzed.sl` remain optional unavailable
runtimes on this host, as reported by the successful installer check-only path.

## 2026-07-22 Housekeeping and Evidence Reconciliation

- Reconciled the README, iiwa commissioning documents, current-state summary,
  and remaining-work plan with the retained three-camera `calib00` repeat. The
  earlier two-camera promotion remains historical evidence instead of being
  presented as the current result.
- Advanced the checked-in PoseTemplateCreator gitlink from `450747b` to the
  already-required `97ddb9b`; code, installer, documentation, and submodule now
  agree on one usable revision.
- Removed dead helpers and constants left by retired downstream/fake-mode
  paths, the unused `tqdm` dependency, an unreachable frontend separator and
  its Radix dependency, the replaced cow image, a byte-identical duplicate HRI
  SVG, and the orphaned fixed sync-offset sample. Explicitly retained the
  compatibility readers and entry points whose sunset remains undecided.
- Replaced the last operator-facing static-object-registry recommendation with
  the immutable pose-template/objectless contract and removed misleading
  estimator wording from the frame-writer regression name.
- Moved license-file declarations to the current project metadata contract,
  required a compatible setuptools build backend, and kept the installer import
  smoke aligned with the actual direct dependencies.
- Marked localhost Playwright coverage explicitly. The default pytest run now
  works without optional Chromium; browser validation remains a separate,
  documented `-m playwright` command.
- Added agent guidance for compatibility-aware deletion, reference searches,
  and Vite-owned hashed assets.
- Audited the complete test suite by production contract. Removed nine
  collected cases: one exact robot-profile subset, two catalogue lifecycle/v1
  cases superseded by stronger revision/tombstone coverage, a tautological USB
  forwarding case, three duplicate Flask route/shell checks, and a screenshot
  case that asserted only image dimensions. Preserved the unique assertions in
  the stronger tests, and retained all hardware-safety, compatibility-reader,
  transaction/race, and acquisition-boundary coverage.

Validation after this pass completed on 2026-07-22:

- 650 default pytest tests and all 28 explicitly marked Playwright tests passed
  (678 total);
- the focused maintenance/redundancy sets, Ruff, frontend type checking and lint,
  the production frontend build, shell syntax, installer check-only with both
  pinned submodules, and `git diff --check` passed; and
- an isolated wheel/sdist build completed without the prior setuptools
  deprecation warning and contained the current Python, frontend, static, and
  license/notice assets.

No camera, robot, lab service, or physical capture was accessed during this
housekeeping pass. BlenderProc and `pyzed.sl` remain optional unavailable
runtimes on this host.

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

For the 2026-07-21 baseline, software validation completed. The two-camera
physical `calib00` capture has an explicitly promoted common bundle and passing
calibration-validation evidence:

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
