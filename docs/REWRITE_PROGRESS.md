# Rewrite Progress

Last updated: 2026-07-23

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
- transactional synchronization and sync-quality reporting, including strict
  hash-bound reuse of each selected calibration profile's saved timing policy;
- explicit `run_config.v3` mixed-mount D435 depth-exposure triggering, complete
  cross-camera frame grouping, and BOP frame-set provenance;
- PoseGridGen targets, including immutable-library card previews rendered from
  stored marker geometry, and attempt-scoped factory-vs-OpenCV intrinsic
  evidence, whole-board PnP support gates, global-sensor-time RealSense
  synchronization, calibration-attempt-only constant effective-latency search,
  common-bundle multi-camera extrinsic ranking, and explicit promotion;
- the dedicated Workpiece Catalogue page and `/workpieces` API, backed by the
  existing JSON/UUID asset store with editable classification, previews,
  guarded lifecycle, revisioned metre/mm correction, metadata portability, and
  pose-template integration;
- the updated PoseTemplateCreator stable-orientation workflow, with bounded
  isometric previews, exact planar layout/fit validation, immutable printable
  templates sourced from active workpieces, preview-rich run selection,
  per-instance GT, and Cell provenance;
- BlenderProc 2.8.0 preparation/render identity checks and transactional BOP
  v4 export; and
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
still fail closed when this run has no matching promoted profile. Calibration
target scenes retain the canonical top-left, +X-right, +Y-down, +Z-into-board
frame and use a presentation-only right-handed target alignment, so cameras on
the printed/front negative-Z side appear above the grid without changing any
stored pose or promoted calibration transform.

Historical run `working_data/hot_full_capture_fixed_20260710_1351` passes the
full-capture gate at 10/10 for three RealSense cameras. The current five-sensor
profile, combined camera-service lifecycle acceptance, and real BOP v4 export
still require operator-run acceptance. That historical run used an older 10 ×
7 / 70-marker board and insufficient hand-eye motion diversity; it is a
preserved negative baseline, not `calib00` calibration evidence.

The `calib00` guided campaign completed three independent physical acquisition
and calibration journeys with all three eye-in-hand RealSense cameras.
Historical attempts `d909d13cc5944a068e8a2ec13eeedd32`,
`3106a2b80b87444db0ac26de89bc01b3`, and
`f1e990d3424a48ed95b266f7bf134838` each produced one complete common bundle.
Maximum within-run stationary-companion closure was 7.847 mm / 1.115°; the
8.642 mm / 1.099° maximum cross-run difference remains a method-confounded
diagnostic requiring a controlled repeat. The validation record is
[EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md](EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md).
At campaign completion each run passed `rewrite_full_capture.v1` at 10/10 and
`rewrite_calibration_validation.v1` at 3/3. The later top-level reusable
profile collections were retired because they predate required time-alignment
provenance; raw captures and immutable attempt evidence remain.

Calibration attempts now default to per-camera Auto time alignment and retain
the complete bounded search and fail-closed decision in
`time_offset_search.json`. A retrospective replay selected +70 to +75 ms,
+80 to +85 ms, and +45 to +55 ms for the three cameras and reduced
cross-validated stationary-target translation residual by 14–38% versus 0 ms.
This is effective-latency tuning from robot motion, not hardware-clock proof,
and it was not promoted back into the historical attempts.

Object-dataset synchronization now applies the exact per-camera timing from the
selected run-owned calibration snapshot and rejects overrides. Sync quality,
rectification, and BOP export recheck camera coverage, values, profile identity,
bundle hash, and timestamp provenance. The guided page exposes the policy and
blocks readiness when it cannot be verified.

The campaign also closed three defects in the guided page: a brand-new run no
longer stalls on a missing config, recorded calibration cameras refresh after
capture, and the physical action now submits the canonical
`real_full_capture_validation` sequence rather than bypassing its hardware
snapshot and preflight artifacts. The calculation card documents the observed
10–20 minute duration and persisted background-job behavior. The
full-capture gate accepts the immutable pre-START preflight embedded in an
execution plan when the standalone report is absent, and rejects mismatched
embedded status.

## 2026-07-23 Combined Mixed-Mount Hardware Synchronization

- Added the explicit `run_config.v3` `capture.synchronization` contract.
  `timestamp_aligned` remains the general default. `hardware_trigger` accepts
  only `realsense_inter_cam_sync` with `scope=depth_exposure`, a safe group ID,
  a selected exact master key, and a bounded earliest-to-latest depth-timestamp
  span across the complete camera group.
  Legacy v1/v2 configs remain loadable as timestamp-aligned evidence and cannot
  claim hardware synchronization.
- Hardware-trigger validation fails closed unless every enabled device is an
  exact-ID D435 and the run includes at least one `static` and one
  `eye_in_hand` camera. Exactly one is the master and all others are
  subordinates. The current USB OAK-D Pro and USB ZED 2i are rejected from this
  mode; an invalid group never silently falls back.
- Capture planning starts the internal D435 master before its subordinates.
  RealSense acquisition configures and reads back each SDK
  `inter_cam_sync_mode`, requires global depth timestamps, and records the
  group, role, scope, depth frame number, and depth exposure timestamp with each
  raw frame. D435 RGB is retained as the associated device frameset but is
  explicitly not certified as a simultaneous cross-camera exposure.
- The capture supervisor now requires append-only frame-metadata progress from
  every camera throughout sequential startup, all-camera overlap, and robot
  receiver execution. An alive process whose metadata stalls, is truncated, is
  rewritten, or becomes non-monotonic aborts local capture while preserving raw
  evidence. Its independent FPS-derived freshness limit spans 12 planned frames
  and is clamped to 2–5 seconds; an explicit execution override cannot exceed
  five seconds, and receiver completion performs the same final check. The
  exact hardware contract and qualification are also rebound
  immediately before receiver startup, so a changed config or replacement
  qualification refuses robot `START`. A successful full-capture report
  persists the exact configuration SHA-256, qualification-artifact SHA-256,
  and immediate-revalidation result.
- Added a fail-closed `hardware_sync_qualification.v1` artifact and evidence
  recorder. It copies operator-supplied pulsed-light or equivalent
  exposure-timing evidence without accessing hardware, verifies file hashes,
  and binds the pass to the exact run, resolution, FPS, trigger policy, camera
  identities, mounts, and roles. Capture preflight, synchronization, and BOP
  consumption reject missing, tampered, or stale qualifications. Publication
  and replacement are blocked once capture status/report/logs, raw
  RGB-D/metadata, or raw robot-pose evidence exists; changing the qualification
  after acquisition requires a new run.
- Non-destructive synchronization preserves every early, incomplete, and
  unmatched raw frame. It publishes only complete cross-camera sets within the
  configured full earliest-to-latest depth-timestamp span to
  `processed/synchronized/multiview_frame_groups.json`, with mounting, source,
  timestamp, skew, and robot-pose references. The artifact also hashes the
  current hardware contract, metadata, matched poses, and every referenced raw
  and synchronized RGB-D file so later mutation invalidates it. Durable writes,
  loads, and downstream consumption rebuild the canonical grouping from those
  inputs, require exact payload equality, and require the exact current physical
  qualification, so self-consistent edits to the derived group JSON are also
  rejected. Authoritative grouping additionally requires a succeeded
  full-capture report with both execution gates and carries its exact
  configuration/qualification binding.
- Transactional BOP v4 export maps those authoritative complete groups across
  the per-camera scenes in `bop/posetestbot_frame_sets.json` and propagates the
  capture-report binding. Its claims state that depth exposures are
  hardware-synchronized, RGB exposures are not certified, and synthetic robot
  occlusion is not modeled. `rewrite_bop_export_readiness.v1` compares the
  binding across the capture report, authoritative groups, and BOP frame sets,
  and also requires the current qualification and exported mapping/bytes to
  agree.
- The dataset setup UI exposes the timing choice, exact master selection, and
  capability/claim blockers while preserving an existing policy when unrelated
  fields are saved.

This work implemented and tested the software contract only. No camera, robot,
lab service, sync harness, or physical capture was accessed. Operator-run
electrical and pulsed-LED/exposure qualification remains required before the
feature becomes real research evidence; the implemented recorder and gates
make that operator step durable and mandatory. Complete real depth observations
share the synchronized exposure instant and depth-visible occlusion; their
associated RGB images are not certified to share a moving-robot or illumination
instant.
Current BlenderProc GT/masks do not render the articulated iiwa.

## 2026-07-23 Workpiece Recognition Preview Fidelity

- The selected Workpiece Catalogue detail now loads the hash-versioned exact
  canonical PLY in its authored orientation. Vertex colours and source normals
  are retained when present, missing normals are generated, and open CAD is
  rendered double-sided. Selected-object recognition no longer depends on
  stable-pose analysis or its compact convex proxy.
- Catalogue cards now use a separate recognition-focused LOD: indexed geometry
  is welded and, when it does not fit, deterministic quadric and spatial
  candidates are compared by component/Euler topology before the legacy convex
  safety proxy. The chosen strategy, counts, topology signatures, and fallback
  reason are retained with the cache. The card remains bounded to 4,096
  vertices, 8,192 faces, and 256 KiB, while immutable pose-template previews
  keep their smaller interaction-oriented tier.
- Dense card meshes render through one Canvas2D surface instead of thousands of
  SVG nodes and are loaded only near the viewport. A keyboard-accessible
  `LOD`/`Approx`/`Proxy` badge reports source and displayed face counts and any
  topology loss, and the catalogue exposes an explicit queued **Refresh card
  preview** action for stale caches.
- The current DGS-108 example improved from a 118-vertex / 232-face convex card
  proxy to a 2,194-vertex / 4,636-face recognition LOD (125,838-byte thumbnail)
  that retains the welded source's Euler value of -124; its selected detail
  uses the full 245,508-vertex / 81,836-face canonical model.
- Added hollow/perforated-part, relative-quantization, cache-bound, and
  deterministic-spatial regressions so topology loss is detected and labelled,
  plus desktop browser coverage for exact selected-model loading, hash
  cache-busting, analysis-free rendering, dense Canvas2D cards, accessible LOD
  evidence, and queued card refresh.

Validation completed on 2026-07-23:

- 720 non-browser pytest tests and all 40 explicitly marked Playwright tests
  passed;
- Ruff, frontend type checking and lint, the production frontend build, shell
  syntax, installer check-only, and `git diff --check` passed; and
- no camera, robot, lab service, or physical capture was accessed.

## 2026-07-22 Guided Operator Workflows

Implementation of the outcome-oriented operator workflow architecture is
complete:

- replaced the generic seven-phase primary navigation with two guided journeys:
  a five-step required camera-calibration spine and a six-step required
  object-dataset spine, with persistent artifact-backed status and visual
  dependencies;
- kept optional target/template authoring, advanced calibration evidence, and
  BlenderProc GT/mask work visibly outside the required spine, while retaining
  individual stage forms only under **Advanced tools** for diagnostics and
  recovery;
- collapsed operator preflight into one visible readiness facade per journey,
  with human-readable missing/stale/failed/invalid states, while preserving the
  separate two-acknowledgement capture dialog and fresh startup checks at the
  physical execution boundary;
- made a prior promoted calibration a required object-dataset input, with
  per-camera compatibility checks, a hash-bound
  `calibration_profile_selection.json`, and exact run-owned profile snapshots
  below `processed/calibration_inputs/<bundle_sha256>/`; snapshot pairs are
  re-hashed at readiness and immediately before rectification, BlenderProc
  preparation, or BOP export, while selection replacement is confirmation/CAS
  gated and blocked after capture or derived dataset material exists;
- added keyboard-accessible contextual help and explicit explanations for
  camera mounting modes, template placement, synchronization, BOP output, and
  Factory SDK versus OpenCV intrinsics. Compatible factory projection remains
  selected by policy; OpenCV activates only as the fully gated fallback when
  factory projection is unusable;
- bound calibration analysis to the step-2 run-owned grid and the step-1 camera
  mounting identities, split mixed static/robot-mounted selections into
  separate attempts, and reject contradictory mode or target submissions at
  the API boundary;
- made guided progress depend on schema/status-validated evidence rather than
  file existence, use the run-level sync-quality report as the aggregate over
  per-camera sync reports, and refresh progress while queued work completes;
  and
- added the versioned `operator_workflows.v1` description endpoint while
  retaining old workflow URLs as redirects into the corresponding guided step.

This was a software-only workflow and documentation change. It did not open a
camera, contact the robot, authorize motion, or complete any outstanding
operator-run acceptance item in
[REWRITE_REMAINING_WORK.md](REWRITE_REMAINING_WORK.md).

Repository-wide software validation of this redesign completed on 2026-07-22:

- 677 non-browser pytest tests and all 33 explicitly marked Playwright tests
  passed (710 total);
- Ruff, frontend type checking and lint, the production Vite build, and
  `git diff --check` passed; and
- the browser suite covered both numbered journeys, responsive navigation,
  one visible readiness action, calibration selection/replacement CAS, fresh
  capture gates, Factory/OpenCV guidance, automatic progress refresh, and the
  consolidated dataset-processing action.

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
  timestamps pair with robot host-wall timestamps, with no fallback and a
  20 ms maximum nearest-pose delta. The original fixed-zero-only behavior is
  retained as an explicit baseline; new guided attempts can apply the
  evidence-gated per-camera auto offset described in Current State. Added
  spatial/campaign target support, rotation-axis rank checks before and after
  pruning, per-motion balanced fitting, and full-input validation.
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
physical `calib00` capture produced an explicitly promoted common bundle and
passing historical calibration-validation evidence:

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
