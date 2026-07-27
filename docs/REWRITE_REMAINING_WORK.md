# Acquisition Rewrite Remaining Work

Last reviewed: 2026-07-26

This is the only repository-owned planning document for unfinished rewrite
work. Completed design plans are retained in Git history, not as live plans.
Operator contracts remain in the focused calibration-target,
[Workpiece Catalogue](WORKPIECE_CATALOGUE.md), pose-template, iiwa teaching,
installation, and README documents.

## Boundary

PoseTestBot ends at a validated BOP dataset. Capture, calibration,
synchronization, optional BlenderProc GT/mask generation, pose-template
provenance, and BOP export are in scope. Estimator execution and BOP result CSV
conversion belong in a consumer repository. The sole evaluation exception is
the Inspect-only, run-scoped official BOP19 validation path: it consumes a
completed annotation-bearing export and an already compatible standard result
CSV, or generates a deterministic test-only slight GT perturbation, and writes
derived evidence only below `processed/bop_evaluation/`. It is not a pipeline
stage.

The lab iiwa is the sole robot profile. Raw capture evidence must never be
overwritten. Every physical action requires explicit operator authorization;
robot-and-camera capture requires both execution gates.

## Verified Baseline

The rewrite already provides:

- real-only capture planning, preflight, supervised execution, cancellation,
  and rewrite gates;
- RealSense, OAK-D Pro, and ZED 2i capture/status contracts, plus supported
  RealSense/OAK live previews, the managed UGREEN WebRTC monitor, and
  run-scoped camera enable/disable selection that retains disabled camera
  identity and metadata;
- transactional non-destructive synchronization and sync-quality reports,
  including the implemented `run_config.v3` mixed-mount D435 depth-exposure
  trigger contract, authoritative complete frame groups, and BOP frame sets;
- PoseGridGen target bundles, attempt-scoped two-mode calibration, exact
  RealSense timebase/intrinsic compatibility gates, evidence-gated per-camera
  constant-offset search with search-corrected leave-one-motion-out timing
  consistency, deterministic common-bundle multi-camera ranking, explicit
  validation/promotion, and derived rectification;
- a dedicated JSON-backed Workpiece Catalogue page and `/workpieces` API with
  queued CAD upload, editable labels/tags/groups/attributes, client-side
  previews, tag/group/state filtering, revisioned/locked mutations, guarded
  archive/delete with retryable cleanup evidence, audited metre/mm geometry
  correction, metadata import/export, stale request cleanup, and stable
  UUID/BOP identity;
- managed PoseTemplateCreator stable-orientation bundles sourced from filtered
  active catalogue workpieces, exact slice/layout validation, bounded catalogue
  and template thumbnails, target-specific artifact verification, interactive
  selected-template 3D previews, strict bundle/selection validation, durable
  selection recovery, run-owned placement and instance snapshots, BlenderProc
  2.8.0 identity validation, explicit pose-only versus pose-plus-mask GT
  generation, and clean BOP v5 export/provenance contracts;
- Inspect-only immutable BOP19 result registration, deterministic GT-derived
  test fixtures, selectable method/result history, and pinned official BOP19
  metric jobs below `processed/bop_evaluation/`;
- a packaged React operator console and scoped Flask APIs; and
- the three acquisition-only gates: `rewrite_full_capture.v1`,
  `rewrite_calibration_validation.v1`, and
  `rewrite_bop_export_readiness.v1`.

The console now makes **Workflow** the canonical operator path and explicitly
hands supporting library, configuration, inspection, and job pages back to the
relevant guided step. Status labels distinguish configuration from hardware
readiness, long job histories have client-side filtering and progressive
disclosure, and required process or safety instructions remain visible rather
than existing only in tooltips. These software improvements do not replace the
operator-run camera, controller, capture, calibration, or BOP acceptance below.

Historical real evidence at
`working_data/hot_full_capture_fixed_20260710_1351` passes
`rewrite_full_capture.v1` at 10/10 for the three RealSense cameras. It remains
valid evidence for that configuration, but it is not acceptance of the current
five-sensor default profile, which also includes OAK-D Pro and ZED 2i. It is
also not calibration evidence for the current `calib00` campaign: the captured
board is the older 10 × 7 / 70-marker geometry, and its motion labels and
rotation diversity do not satisfy the current hand-eye gates. Preserve it as a
negative baseline; never relabel or retrofit it as a `calib00` run.

The 2026-07-21 repository audit also verified both pinned source checkouts, a
wheel and sdist containing the complete pose-template/UI surface, and an
installed-wheel Flask smoke. Optional BlenderProc and `pyzed.sl` were not
available on the audit host; that is non-blocking for ordinary development but
must be resolved for the relevant real-data milestones below.

The three-RealSense `calib00` guided campaign retained three independent runs.
At campaign completion each passed the full-capture gate at 10/10 and
calibration-validation gate at 3/3; historical evidence is summarized in
[EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md](EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md).
Their obsolete reusable profiles are now retired, so a fresh Auto time-aligned
calculation and promotion has now completed for all three cameras as immutable
attempt `268c897e1baf49e7bd78a434a4569b99`. Its common `IPPE + Shah` bundle is
published and the calibration-validation rewrite gate is ready at 3/3. The
camera-service lifecycle acceptance in P1, the five-sensor capture in P2,
controller commissioning, and real BOP acceptance in P4 remain open.

## P0 — Safety and Capture-Contract Hardening

The completed receiver-hardening items below precede every deliberate robot
capture. The still-open ordinary full-capture application items are P2
blockers; the nine-frame calibration deployment and commissioning blockers are
tracked separately under P3.

### Harden the low-level pose receiver

- [x] Make direct `scripts/pose_receiver_udp_json.py` execution require fresh
  `allow_real_robot` and `allow_cameras` acknowledgements at the execution
  boundary. Do not bake approvals into reusable plan artifacts.
- [x] Refuse an existing `raw_robot_ee_poses.json` before binding the socket or
  sending the start command, closing the preflight/execution race and
  protecting direct invocation.
- [x] Add a configurable receive-start/idle timeout and record a terminal
  failed/canceled manifest state on timeout, malformed packets, bind failure,
  or interruption. Preserve any partial evidence separately; never replace a
  prior raw artifact.
- [x] Cover direct invocation, supervised invocation, overwrite refusal,
  timeout, malformed packet, and cancellation behavior without contacting the
  robot.

The hardened receiver uses a conservative 120-second first-packet timeout and
60-second inter-packet timeout by default. Supervised capture injects both
fresh acknowledgements and timeout values only into the runtime receiver
command, records that command in the execution report, and leaves reusable
capture/sequence plans authorization-free. The supervisor default is 720
seconds so the lowest selectable 0.01 m/s A1 sweep can finish. Every selected
camera must publish at least three valid, committed
`frame_metadata.jsonl` records within the 15-second readiness window before
receiver bind or `START`. Direct IIWA start uses the same two fresh
acknowledgements.

### Establish one authoritative Sunrise capture application

- [ ] Confirm which Sunrise application is deployed for ordinary full capture.
  `iiwa/PoseTestBot_Test.java` is the remaining likely candidate, but its name
  and deployed status are not authoritative evidence.
- [x] Align the repository candidate's command semantics with
  `cartesian_velocity_m_s`. It now converts the requested tangential flange
  speed through the measured A1 orbit radius and a conservative published A1
  speed bound before calling `setJointVelocityRel`. Run-owned acquisition
  retains a 0.03 m/s legacy/calibration cap; object-dataset requests above it
  use `robot_command.v1` and may pass through up to 1.00 m/s. The candidate
  accepts that Cartesian request and independently caps A1 at 3°/s. The
  separately acknowledged manual Dashboard/Devices motion test sends a legacy
  0.1 request; reconcile the deployed application before treating that value
  as Cartesian m/s rather than 10% relative joint speed.
- [x] Align documented receiver fallback/address behavior with the lab receiver
  `172.31.1.169`, while retaining the command-supplied receiver target and an
  explicit wildcard-to-command-sender mode.
- [x] Make packet/parse/transmit failures observable instead of silently
  swallowing them, document that UDP stop is not a safety stop and cannot
  interrupt active motion, and retain the repository candidate's v1 packet
  sequence/run/frame evidence in the backward-compatible Python receiver.
- [ ] Create, teach, and commission the distinct persistent ordinary-capture
  frames `/PoseTestBot/PoseTemplateBase`, `/PoseTestBot/CaptureStart`, and
  `/PoseTestBot/CaptureEnd`; record the measured relationship between the
  ordinary reference frame and the calibration application's
  `/PoseTestBot/TemplateBase`; commission the complete PTP/A1/PTP path; and
  verify that pose-template placement plus every selected camera profile is
  expressed in the ordinary run's dataset reference. Follow
  [IIWA_FULL_CAPTURE_APPLICATION.md](IIWA_FULL_CAPTURE_APPLICATION.md).
- [ ] Compile and simulate the exact controller project offline, then rename or
  replace the source only after the deployed application is identified. Keep
  the nine-frame calibration application separate from the ordinary
  full-capture application and reconcile its enabled repository/deployed state
  independently.

## P1 — Combined Camera-Service Lifecycle Acceptance

This is operator-run hardware work. It must not send a robot command or start
the acquisition pipeline.

- [ ] On the operator-ready lab host, run UGREEN WebRTC, all three RealSense
  live previews, and OAK-D Pro preview concurrently for at least 30 seconds.
- [ ] Exercise the UI through both `10.145.8.132` and the current Tailscale
  address, with two simultaneous UGREEN peers. Require every media/frame
  counter to keep advancing.
- [ ] Capture and validate RGB/depth snapshots from the three RealSense cameras
  and OAK-D Pro while the service matrix is exercised.
- [ ] Repeat across graceful web-server `SIGTERM`, forced web-server `SIGKILL`,
  restart, and an individual monitor-worker crash. Require owned PIDs and
  device handles to clear within five seconds, then require every supported
  stream to restart.
- [ ] Retain a timestamped JSON report and logs below
  `working_data/web_camera_acceptance/`, including device IDs, endpoints,
  counters, process identities, release timing, and failures.
- [ ] Re-run the synthetic Playwright suites and frontend validation after any
  hardware-discovered fix.

ZED live preview is not implemented and the console now exposes that
capability as unavailable. ZED capture acceptance belongs to P2; implementing a
ZED preview is optional unless an operator workflow requires it.

## P2 — Current Five-Sensor Full-Capture Acceptance

Depends on P0 and on an operator-ready robot/camera cell.

- [ ] Verify all three configured RealSense serials, the OAK-D Pro, and the ZED
  2i are visible and correctly aliased/mounted. Install and verify `pyzed.sl` on
  the lab host first.
- [ ] Create a new run root configured at the reviewed capture velocity (the
  prior plan called for `0.05 m/s`). Never reuse a root containing raw frames or
  raw robot poses.
- [ ] Generate and inspect `real_full_capture_validation` with `--plan-only`,
  including receiver routing, sensor IDs, resolution/FPS, resources, startup
  order, output folders, and overwrite blockers.
- [ ] With explicit operator authorization and both execution gates, run a
  short supervised trial, then the deliberate full capture.
- [ ] Require balanced RGB/depth/metadata tuples for every selected sensor,
  nonempty robot poses, clean process/device release, acceptable sync quality,
  and a 10/10 `rewrite_full_capture.v1` report.
- [ ] Preserve the plan, preflight, execution status/report/logs, hardware
  snapshot, raw folders, sync reports, and gate report as acceptance evidence.

### Combined mixed-mount D435 hardware-sync acceptance

This is a separate D435-only research acceptance, not an extension of the
five-sensor run above. The software contract is complete, but no sync harness,
camera, robot, or physical capture was accessed during its implementation.

- [ ] Assign exact D435 identities and physically fixed mounts with at least one
  `static` and one `eye_in_hand` view. Select exactly one internal master and
  treat every other group member as a subordinate.
- [ ] Create a fresh `run_config.v3` dataset run with
  `capture.synchronization` set to `hardware_trigger`,
  `realsense_inter_cam_sync`, and `depth_exposure`. Never reuse a run root that
  contains capture status/report/logs, raw camera data, or raw robot poses.
- [ ] Build and inspect the sync harness against Intel's D400 multi-camera
  guidance. Retain continuity/electrical-limit evidence and verify the cable
  routing remains safe through the complete iiwa motion envelope.
- [ ] Without robot motion, verify master/subordinate
  `inter_cam_sync_mode` configuration and read-back, global depth timestamps,
  expected frame-number cadence, and a shared-view pulsed LED visible in the
  depth/IR stream, or an equivalent exposure-timing test, at the intended frame
  rate.
- [ ] Record the passed external evidence with
  `scripts/record_hardware_sync_qualification.py --confirm-passed`; retain its
  run/config-bound `hardware_sync_qualification.json` and copied evidence.
  Confirm the recorder refuses publication/replacement after any acquisition
  evidence exists and directs the operator to a new run. Plan and preflight the
  qualified run, then perform a short supervised robot trial only with explicit
  authorization and both execution gates.
- [ ] Verify that raw early/incomplete frames remain preserved while
  `processed/synchronized/multiview_frame_groups.json` contains only complete
  mixed-mount sets whose full earliest-to-latest timestamp span is within the
  configured threshold, with valid robot-pose and source references. Exercise
  the live camera-progress watchdog during the supervised trial. Verify its
  default 12-frame-period deadline is clamped to 2–5 seconds and remains
  independent of the robot UDP timeout; retain evidence that an intentionally
  stopped test stream aborts local capture without deleting partial raw data.
- [ ] Verify the succeeded full-capture report records the exact configuration
  SHA-256, qualification-artifact SHA-256, immediate pre-receiver
  revalidation, full mode, and both execution gates. Require authoritative
  grouping to fail when that report or binding is changed.
- [ ] Confirm representative complete depth sets consistently observe real
  robot occlusion at the synchronized depth exposure. Retain the explicit
  limitation that associated D435 RGB images are not hardware-certified and
  cannot establish a shared moving-robot or changing-illumination instant.
- [ ] Export BOP v5 and verify `bop/posetestbot_frame_sets.json` maps every
  complete set to all expected scene/image views and carries the exact
  capture-report configuration/qualification binding. Require
  `rewrite_bop_export_readiness.v1` to fail if the current qualification,
  capture report, authoritative groups, BOP frame sets, frame map, or exported
  bytes disagree. Do not treat the full rendered object mask as
  robot-occluder truth; the articulated iiwa is not rendered. The visible mask
  may reflect robot occlusion only where captured depth validly observed it.
- [ ] If the research later requires OAK or ZED views in the same hardware
  exposure group, qualify trigger-capable replacement interfaces, a
  level-compatible isolated distribution design, and new adapter contracts.
  The current USB OAK-D Pro and USB ZED 2i must remain rejected rather than
  silently timestamp-aligned inside a claimed hardware group.

## P3 — Nine-Frame Commissioning and Real Calibration

### Sunrise.Workbench and physical commissioning

- [ ] Import and compile
  `iiwa/PoseTestBot_CalibrationVarianceProposal.java` in the exact Workbench
  project. Create and resolve exactly nine persistent children below
  `/PoseTestBot/TemplateBase`; there are no persistent depth or orientation
  variant frames in this revision.
- [ ] Reconcile the exact deployed application revision with the enabled
  repository source and retain the offline endpoint/swept-path evidence in
  `docs/IIWA_CALIBRATION_TEACHING_CHECKLIST.md`. The enable boolean alone is not
  commissioning evidence.
- [ ] Obtain separate authorization for reduced-speed T1 single-stepping and
  record joint branch, singularity, collision/clearance, payload, cable, and
  visibility evidence for the raster and all nine relative orientation legs.
- [ ] Only after offline and T1 acceptance, run a supervised calibration
  capture with both execution gates.

### Calibration evidence and promotion

Three retained guided runs live at
`working_data/calib00_guided_real_20260723_run01`,
`working_data/calib00_guided_real_20260723_run02`, and
`working_data/calib00_guided_real_20260723_run03`. Their raw captures and
immutable attempts remain historical evidence. Their reusable top-level
profiles were retired because the 0 ms attempts predate required per-camera
Auto time-alignment provenance. The 8.642 mm maximum cross-run difference is
also method-confounded and requires a controlled repeat; see
[EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md](EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md).
Retained run `working_data/calib00_test20260724` now has the explicitly
promoted v2 replacement from attempt
`268c897e1baf49e7bd78a434a4569b99`. That satisfies the reusable-profile
dependency for those recordings, but it does not replace the still-open
post-commissioning supervised capture below.

Status and capture-plan preflight continue to reject an enabled
SDK-enumerated RealSense whose known `usb_type_descriptor` has a major version
below 3, preventing a future USB2 fallback from satisfying readiness. Disabled
cameras remain visible configuration/diagnostic evidence but are excluded from
capture/preflight, calibration, Cell, and rewrite-gate expectations.

- [ ] Run and explicitly publish a fresh supervised three-camera `calib00`
  calibration through the current workflow. Require passing per-camera Auto
  time alignment, one complete common solver bundle, current intrinsic and
  target evidence, and passing rewrite gates before using it for a dataset.
  The detailed completed software contracts live in
  [REWRITE_PROGRESS.md](REWRITE_PROGRESS.md).
- [ ] Revalidate metric depth on RealSense `923322072633` after the cable and
  firmware maintenance opportunity. Saved aligned-depth checks showed a
  range-dependent scale anomaly; factory depth scale/alignment remains
  explicitly not recalibrated.

## P4 — Real Pose-Template, Optional BlenderProc, and BOP v5 Acceptance

A reusable three-camera v2 calibration is now published from retained run
`calib00_test20260724`. A future dataset may select it only when its exact
sensor, mount, resolution, orientation, target, and timing compatibility gates
pass. The real annotation-free v4 output from
`working_data/test20260725_04` proved the capture/synchronization content but
failed the later official BOP Toolkit model-loader audit. Future exports use
the clean `bop_export_manifest.v5` contract. The retained
`working_data/test20260726_BOPv5` run has now passed BlenderProc 2.8.0
pose-plus-mask generation, official BOP19 target/metric validation, and the
11/11 rewrite gate; regenerating the older v4 run and the broader
physical-template review below remain outstanding.

- [ ] Import/inspect and classify the real CAD and texture assets through
  **Workpiece Catalogue**, exercise name/tag/group filters, verify the compact
  and exact interactive identification previews and millimetre dimensions,
  confirm that representative real holes, ports, handles, and recesses remain
  recognizable against the CAD/physical workpiece, choose reviewed stable
  orientations, generate an immutable printable template from active
  workpieces, print/measure it, and confirm the full
  `template_base_from_pose_template` placement for a new pose-template run.
- [ ] Include at least one duplicate physical instance if duplicate-category
  behavior is part of the intended dataset. Verify exact slicing, immutable
  hashes, stable `obj_id` reuse, and unique instance UUIDs.
- [x] Run the new guided **Pose + masks** product on the retained real v5
  dataset with BlenderProc 2.8.0. Require camera, calibration,
  selection, geometry, BlenderProc/toolkit identity, GT-index, and instance
  identity evidence to agree for every sensor/frame. The official mask pass
  compares rendered object depth with captured depth, so unmodelled robot
  occlusion is reflected when the sensor measured it; still inspect
  robot-intersected and missing-depth views explicitly. The retained product
  contains 1,621 pose rows, full masks, and visible masks across its two
  scenes; immutable read-back and all 1,621 visibility-filtered targets agree.
- [x] Audit the retained real timestamp-aligned run. Confirm that its two
  811-frame BOP scenes contain only synchronized capture-sweep frames and that
  pre/post-motion raw evidence remains outside the export. Reproduce the v4
  canonical-model failure with the official BOP Toolkit PLY loader and replace
  future byte-for-byte model copying with normalized BOP ASCII PLY output.
- [ ] Regenerate `working_data/test20260725_04/bop/` through the v5 exporter
  when the operator wants the retained derived output replaced. Confirm the
  official loader accepts both model copies, `models_eval/` is present for the
  metric scripts, absolute run paths and the unused third-camera profile are
  absent, annotation placeholders are absent, and
  `rewrite_bop_export_readiness.v1` passes. Do not alter its raw camera or
  robot evidence.
- [ ] A future hardware-trigger run must additionally require
  `posetestbot_frame_sets.json` to cover every authoritative complete
  mixed-mount group.
- [x] After optional rendered annotations exist, inspect representative
  RGB/depth/GT/masks and any repeated-object rows, then rerun
  `rewrite_bop_export_readiness.v1` on that annotation-bearing real dataset.
  Confirm every target `inst_count` exactly counts GT rows with
  `visib_fract >= 0.1`; do not treat a legacy target mismatch warning as
  leaderboard-comparable acceptance.
  Use **Inspect → BOP Evaluation** to run a deterministic slight-offset GT
  fixture through the pinned official BOP19 metrics and retain its dataset,
  result, toolkit, renderer, and score provenance below
  `processed/bop_evaluation/`. The real v5 run passed the 11/11 rewrite gate;
  it has one instance per image, so duplicate-category row acceptance remains
  with the separate unchecked physical-duplicate task above;
  its seed-42, 1 mm / 0.25° validation fixture produced official BOP19 AR
  0.9628 (VSD 0.8885, MSSD 1.0000, MSPD 1.0000).
- [ ] When external pose-estimator output becomes available, convert it in the
  consumer project, import at least two canonical BOP19 CSV result runs, and
  verify that their method/result selections and retained reports remain
  independent. Do not add an estimator wrapper or result converter here.

## P5 — Non-Blocking Maintainability Work

These tasks are useful but do not replace the real-data gates.

- [ ] Finish decomposing `posetestbot/web/legacy.py`: move its active route
  groups into focused blueprints, move shared runner/configuration ownership to
  an explicitly named module, retain endpoint compatibility tests, and delete
  the legacy module only when no active import remains.
- [ ] Decide and document a compatibility sunset for `web_interface.py` and
  the direct single-sensor `scripts/sync_non_destructive.py` CLI. Do not remove
  either while external use is unknown.
- [ ] Decide when legacy `run_config.v1`/`run_config.v2`, calibration v1, and
  historical BOP/sync readers can be retired. Until a data migration policy
  exists, they remain supported readers rather than dead code. The
  object-registry path was retired on 2026-07-21; object-bearing runs now
  require pose-template bundles.
- [ ] Reduce the shared lazy Three.js/OrbitControls production chunk (about
  909 kB minified in the 2026-07-22 build) if operator load time or deployment
  limits justify it. Preserve the WebGL-free fallback and add a bundle-size
  assertion before treating this as a release gate.
- [ ] If operators need one-file cross-host catalogue portability, design a
  verified binary bundle format that contains the manifest plus CAD, canonical
  PLY, and texture bytes. Current JSON export/import is intentionally
  metadata-only and skips workpieces whose UUID-addressed assets are absent on
  the importing host; normal filesystem backup of the managed asset tree is
  the current binary-preservation path.
- [ ] Bound persisted job-history loading and the `/jobs` response with an
  explicit retention or server-side pagination contract. The console now
  filters and progressively reveals the returned history, but
  `LocalJobRunner` still loads every retained `job.json` and the API still
  serializes the complete operator-visible history on each request. Preserve
  active jobs, failed-job diagnostics, logs, and resource ownership when
  introducing the bound.

## Repository Exit Criteria

Before declaring the rewrite complete, record a clean run of:

```bash
bash -n scripts/install.sh
bash scripts/install.sh --check-only \
  --with-posegridgen --with-posetemplatecreator --with-bop-toolkit
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
cd frontend && bun run typecheck && bun run lint && bun run build
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
git diff --check
```

The default pytest selection excludes the marked Playwright modules. Run the
explicit browser command outside the sandbox only when localhost socket
restrictions require the standing authorization. Build a wheel and sdist to a
temporary directory, install the wheel without dependencies into a temporary
environment, and verify the Flask app, `/workpieces` APIs, retained legacy
pose-template catalogue routes, pose-template generation/selection routes, the
Workpiece Catalogue page, and the exact bundled asset set.

Rewrite completion additionally requires acceptance evidence from P1–P4 and
all three rewrite gates passing on the intended real dataset. Ordinary
automated tests must never open cameras, contact the robot, or execute physical
capture.

## Explicitly Deferred or Out of Scope

- Articulated iiwa rendering remains deferred until datasets record joint
  states and approved robot geometry/transforms exist.
- ZED live preview is optional; ZED status, snapshot/capture, calibration, and
  export contracts remain in scope.
- Estimator runtimes and result conversion remain permanently outside this
  repository.
- General evaluator bridges, evaluation pipeline stages, and legacy
  metric-report exports remain out of scope. The only supported metric path is
  the run-scoped Inspect-only official BOP19 validation exception documented
  in the Boundary section.
- Upstream `third_party/PoseGridGen/plan.md` and
  `third_party/PoseTemplateCreator/PLAN.md` belong to pinned submodules and must
  not be edited or deleted here.
