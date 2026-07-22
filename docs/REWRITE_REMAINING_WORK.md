# Acquisition Rewrite Remaining Work

Last reviewed: 2026-07-22

This is the only repository-owned planning document for unfinished rewrite
work. Completed design plans are retained in Git history, not as live plans.
Operator contracts remain in the focused calibration-target,
[Workpiece Catalogue](WORKPIECE_CATALOGUE.md), pose-template, iiwa teaching,
installation, and README documents.

## Boundary

PoseTestBot ends at a validated BOP dataset. Capture, calibration,
synchronization, optional BlenderProc GT/mask generation, pose-template
provenance, and BOP export are in scope. Estimator execution, BOP result CSV
conversion, evaluation, and metric reporting belong in a consumer repository.

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
- transactional non-destructive synchronization and sync-quality reports;
- PoseGridGen target bundles, attempt-scoped two-mode calibration, exact
  RealSense timebase/intrinsic compatibility gates, deterministic common-
  bundle multi-camera ranking, explicit validation/promotion, and derived
  rectification;
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
  2.8.0 identity validation, and BOP v3 provenance sidecars;
- a packaged React operator console and scoped Flask APIs; and
- the three acquisition-only gates: `rewrite_full_capture.v1`,
  `rewrite_calibration_validation.v1`, and
  `rewrite_bop_export_readiness.v1`.

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

The current three-RealSense `calib00` confirmation run passes the full-capture
gate at 10/10 and calibration-validation gate at 3/3. Its complete retained
evidence is summarized in
[EYE_IN_HAND_CALIBRATION_VALIDATION_20260722.md](EYE_IN_HAND_CALIBRATION_VALIDATION_20260722.md).
This closes the three-camera calibration dependency, but not the combined
camera-service lifecycle acceptance in P1, the five-sensor capture in P2, the
controller commissioning evidence below, or the real BOP acceptance in P4.

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
capture/sequence plans authorization-free. The supervisor default is 300
seconds. Every selected camera must publish at least three valid, committed
`frame_metadata.jsonl` records within the 15-second readiness window before
receiver bind or `START`. Direct IIWA start uses the same two fresh
acknowledgements.

### Establish one authoritative Sunrise capture application

- [ ] Confirm which Sunrise application is deployed for ordinary full capture.
  `iiwa/PoseTestBot_Test.java` is the remaining likely candidate, but its name
  and deployed status are not authoritative evidence.
- [ ] Align its command semantics with `cartesian_velocity_m_s`. It currently
  passes that value to `setJointVelocityRel`, so the Python unit/name and robot
  interpretation disagree.
- [ ] Align documented receiver fallback/address behavior with the lab receiver
  `172.31.1.169`, while retaining the command-supplied receiver target.
- [ ] Make packet/parse/transmit failures observable instead of silently
  swallowing them, and document that a UDP stop is not a safety stop and cannot
  interrupt an active motion in the current program.
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

The initial 2026-07-21 live preflight for
`working_data/eye_in_hand_calib00_20260721_163559` stopped before `START` after
RealSense `033422071805` lost its SuperSpeed/UVC interfaces. The subsequent
two-camera promotion remains an independent baseline, but it was superseded by
two preserved three-camera runs on 2026-07-22. The confirmation run
`working_data/eye_in_hand_calib00_3cam_repeat_20260722_1202` promoted immutable
attempt `12e6a40eff444b889870597b787bf016` with a complete common
`IPPE + Shah` bundle. It produced 605/606, 608/608, and 610/610 inliers,
held-out means of 3.052 mm / 0.628 degrees, 3.241 mm / 0.473 degrees, and
3.226 mm / 0.425 degrees, and 7.104 mm / 0.421 degrees maximum
stationary-companion closure. Exact transforms and promotion provenance are
exposed in Cell. `rewrite_full_capture.v1` is 10/10 ready and
`rewrite_calibration_validation.v1` is 3/3 ready for all three cameras.

Status and capture-plan preflight continue to reject an enabled
SDK-enumerated RealSense whose known `usb_type_descriptor` has a major version
below 3, preventing a future USB2 fallback from satisfying readiness. Disabled
cameras remain visible configuration/diagnostic evidence but are excluded from
capture/preflight, calibration, Cell, and rewrite-gate expectations.

- [x] For the current eye-in-hand campaign, use the actual printed and measured
  `calib00` bundle (`15b49f67-7cf5-4c00-9e7f-914aa6ed5da0`):
  `DICT_5X5_100`, 7 × 5, 35 markers, geometry SHA-256
  `3da681424ff77e55dc51c8c1c9bb58e0a425f7fa039b63d29c798aa2ad02b256`.
  Keep placement `unknown` so eye-in-hand solving estimates the stationary
  target companion transform. Require target/hash/placement agreement through
  detections, observations, candidates, ranking, and promotion.
- [x] Retain `intrinsic_comparison.json` for every enabled camera, including
  both factory and manual evidence and the selection reason. Treat RealSense
  `inverse_brown_conrady` as forward-OpenCV-compatible only when every
  coefficient is finite and exactly zero; retain that factory pinhole
  projection and keep the manual result comparison-only. Never pass nonzero
  inverse coefficients as forward distortion. When factory projection is
  unusable, activate manual intrinsics only after at least 15 training views,
  6/9 image-centroid cells, five held-out views, parameter-plausibility,
  3 px/view, and 1.5 px RMS gates pass.
- [x] Require each target pose to have at least 12 common corner inliers, at
  least 50% whole-board support, and no more than 3 px whole-board mean
  reprojection error, with four three-corner-supported markers spanning two
  rows/columns per view and 50% marker plus 60% row/column campaign coverage.
- [x] For every required camera, require at least 15 accepted extrinsic views,
  6/9 image-centroid cells, four distinct motion poses, at least 20 mm
  translation span, at least 5° rotation span, six hand-eye inliers, no more
  than 10 mm / 5° mean held-out residual, no more than 25% motion-balanced
  outliers, no more than 25% outliers within any repeated motion, and
  rotation-axis singular ratio 0.15 from at least 2° samples before/after
  pruning. Treat raw outlier density as evidence rather than a promotion gate.
  Balance fitting to five frames per motion and validate every accepted frame.
  For RealSense require SDK `global_time` color sensor timestamps paired with
  robot host-wall timestamps, zero manual offset, no fallback, and at most
  20 ms nearest-pose delta.
- [x] Confirm the historical high reprojection error on RealSense
  `825412070181` did not recur: factory/manual held-out RMS is 1.230/0.964 px
  and promoted-candidate mean reprojection is 1.040 px. Trajectory variation
  was not treated as a correction.
- [x] Review every PnP/extrinsic result. For the enabled cameras require a
  complete common bundle using the same PnP and extrinsic method, with every
  individual candidate passing and maximum pairwise stationary-companion
  closure no greater than 10 mm / 5°. Establish the best normalized mean
  individual score, treat bundles within 0.01 as quality-equivalent, and rank
  that band by normalized companion closure rounded to six decimals, followed
  by canonical method tie-breaks. Keep clearly worse individual solutions
  outside the band even if their closure is smaller. If no common bundle
  passes, fail closed and forbid partial or mixed-method promotion. Explicitly
  promote the selected complete bundle, preserve unrelated profiles, expose
  its exact transforms in Cell, and pass
  `rewrite_calibration_validation.v1` with promotion evidence.
- [ ] Revalidate metric depth on RealSense `923322072633` after the cable and
  firmware maintenance opportunity. Saved aligned-depth checks showed a
  range-dependent scale anomaly; the promoted RGB eye-in-hand transform is
  valid, but factory depth scale/alignment remains explicitly not recalibrated.

## P4 — Real Pose-Template, BlenderProc, and BOP v3 Acceptance

The promoted-calibration dependency is satisfied by the repeated three-camera
`calib00` attempt. Real BlenderProc 2.8.0 and BOP acceptance remain outstanding
for an appropriate dataset run.

- [ ] Import/inspect and classify the real CAD and texture assets through
  **Workpiece Catalogue**, exercise name/tag/group filters, verify the compact
  and interactive identification previews and millimetre dimensions, choose
  reviewed stable orientations, generate an immutable printable template from
  active workpieces, print/measure it, and confirm the full
  `template_base_from_pose_template` placement for a new pose-template run.
- [ ] Include at least one duplicate physical instance if duplicate-category
  behavior is part of the intended dataset. Verify exact slicing, immutable
  hashes, stable `obj_id` reuse, and unique instance UUIDs.
- [ ] Prepare and render real GT/masks with BlenderProc 2.8.0. Require camera,
  calibration, selection, geometry, renderer-version, GT-index, and instance
  identity evidence to agree for every sensor/frame.
- [ ] Export `bop_export_manifest.v3`, standard BOP scenes/models/targets, the
  frame map, `posetestbot_pose_template.json`, and
  `posetestbot_instance_map.json` transactionally.
- [ ] Inspect representative RGB/depth/GT/masks and repeated-object rows, then
  pass `rewrite_bop_export_readiness.v1` on the real dataset.

## P5 — Non-Blocking Maintainability Work

These tasks are useful but do not replace the real-data gates.

- [ ] Finish decomposing `posetestbot/web/legacy.py`: move its active route
  groups into focused blueprints, move shared runner/configuration ownership to
  an explicitly named module, retain endpoint compatibility tests, and delete
  the legacy module only when no active import remains.
- [ ] Decide and document a compatibility sunset for `web_interface.py` and
  the direct single-sensor `scripts/sync_non_destructive.py` CLI. Do not remove
  either while external use is unknown.
- [ ] Decide when legacy `run_config.v1`, calibration v1, and historical
  BOP/sync readers can be retired. Until a data migration policy exists, they
  remain supported readers rather than dead code. The object-registry path was
  retired on 2026-07-21; object-bearing runs now require pose-template bundles.
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

## Repository Exit Criteria

Before declaring the rewrite complete, record a clean run of:

```bash
bash -n scripts/install.sh
bash scripts/install.sh --check-only \
  --with-posegridgen --with-posetemplatecreator
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
- Estimator runtimes, result conversion, BOP evaluation, and metric reporting
  remain permanently outside this repository.
- Upstream `third_party/PoseGridGen/plan.md` and
  `third_party/PoseTemplateCreator/PLAN.md` belong to pinned submodules and must
  not be edited or deleted here.
