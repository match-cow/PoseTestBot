# Acquisition Rewrite Remaining Work

Last reviewed: 2026-07-21

This is the only repository-owned planning document for unfinished rewrite
work. Completed design plans are retained in Git history, not as live plans.
Operator contracts remain in the focused calibration-target, pose-template,
iiwa teaching, installation, and README documents.

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
  RealSense/OAK live previews and the managed UGREEN WebRTC monitor;
- transactional non-destructive synchronization and sync-quality reports;
- PoseGridGen target bundles, attempt-scoped two-mode calibration, explicit
  validation/promotion, and derived rectification;
- managed PoseTemplateCreator catalog/bundles, run-owned placement and
  instance evidence, BlenderProc 2.8.0 identity validation, and BOP v3
  provenance sidecars;
- a packaged React operator console and scoped Flask APIs; and
- the three acquisition-only gates: `rewrite_full_capture.v1`,
  `rewrite_calibration_validation.v1`, and
  `rewrite_bop_export_readiness.v1`.

Historical real evidence at
`working_data/hot_full_capture_fixed_20260710_1351` passes
`rewrite_full_capture.v1` at 10/10 for the three RealSense cameras. It remains
valid evidence for that configuration, but it is not acceptance of the current
five-sensor default profile, which also includes OAK-D Pro and ZED 2i.

The 2026-07-21 repository audit also verified both pinned source checkouts, a
wheel and sdist containing the complete pose-template/UI surface, and an
installed-wheel Flask smoke. Optional BlenderProc and `pyzed.sl` were not
available on the audit host; that is non-blocking for ordinary development but
must be resolved for the relevant real-data milestones below.

## P0 — Safety and Capture-Contract Hardening

These items precede another deliberate robot capture.

### Harden the low-level pose receiver

- [ ] Make direct `scripts/pose_receiver_udp_json.py` execution require fresh
  `allow_real_robot` and `allow_cameras` acknowledgements at the execution
  boundary. Do not bake approvals into reusable plan artifacts.
- [ ] Refuse an existing `raw_robot_ee_poses.json` before binding the socket or
  sending the start command, closing the preflight/execution race and
  protecting direct invocation.
- [ ] Add a configurable receive-start/idle timeout and record a terminal
  failed/canceled manifest state on timeout, malformed packets, bind failure,
  or interruption. Preserve any partial evidence separately; never replace a
  prior raw artifact.
- [ ] Cover direct invocation, supervised invocation, overwrite refusal,
  timeout, malformed packet, and cancellation behavior without contacting the
  robot.

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
  the nine-frame calibration proposal as a separate, disabled commissioning
  program.

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
- [ ] Complete the offline endpoint and swept-path simulation in
  `docs/IIWA_CALIBRATION_TEACHING_CHECKLIST.md`. Keep
  `ENABLE_AFTER_OFFLINE_VALIDATION=false` until that review passes.
- [ ] Obtain separate authorization for reduced-speed T1 single-stepping and
  record joint branch, singularity, collision/clearance, payload, cable, and
  visibility evidence for the raster and all nine relative orientation legs.
- [ ] Only after offline and T1 acceptance, run a supervised calibration
  capture with both execution gates.

### Calibration evidence and promotion

- [ ] Use the actual printed, measured target bundle and require target/hash/
  placement agreement through detections, observations, candidates, ranking,
  and promotion.
- [ ] For every required camera, meet at least 15 accepted views, 6/9 coverage
  cells, no more than 3 px per-view error, no more than 1.5 px intrinsic RMS,
  sufficient translation/rotation diversity, and passing sync quality.
- [ ] Investigate RealSense `825412070181` separately if its historical high
  reprojection error recurs; trajectory variation alone is not a correction.
- [ ] Review every PnP/extrinsic result, explicitly accept only passing camera
  profiles, preserve unrelated profiles, and pass
  `rewrite_calibration_validation.v1` with promotion evidence.

## P4 — Real Pose-Template, BlenderProc, and BOP v3 Acceptance

Depends on promoted real calibration and BlenderProc 2.8.0.

- [ ] Import/inspect the real CAD and texture assets, generate an immutable
  printable template, print/measure it, and confirm the full
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
- [ ] Decide when legacy `run_config.v1`, calibration v1, object-registry, and
  historical BOP/sync readers can be retired. Until a data migration policy
  exists, they remain supported readers rather than dead code.
- [ ] Reduce the lazy Cell production chunk (about 932 kB minified in the
  2026-07-21 build) if operator load time or deployment limits justify it.
  Preserve the WebGL-free fallback and add a bundle-size assertion before
  treating this as a release gate.

## Repository Exit Criteria

Before declaring the rewrite complete, record a clean run of:

```bash
bash -n scripts/install.sh
bash scripts/install.sh --check-only \
  --with-posegridgen --with-posetemplatecreator
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
cd frontend && bun run typecheck && bun run lint && bun run build
git diff --check
```

Run both Playwright modules outside the sandbox only when localhost socket
restrictions require the standing authorization. Build a wheel and sdist to a
temporary directory, install the wheel without dependencies into a temporary
environment, and verify the Flask app, pose-template routes, and exact bundled
asset set.

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
