# Rewrite Progress

Last updated: 2026-07-17

PoseTestBot has been refocused as an acquisition, calibration,
synchronization, and BOP dataset export repository. The BOP dataset export is
the finish line for this repo.

## Current Scope

In scope:

- fixed real lab robot profile handling,
- sensor registry/status/capture contracts,
- capture planning, preflight, and supervised execution,
- non-destructive synchronization,
- sync quality reports,
- calibration profile preflight, observations, candidates, solver, validation,
  and explicit promotion,
- ArUco/target detection support for calibration,
- BlenderProc preparation/render planning for optional GT/masks,
- BOP dataset export,
- React/shadcn operator console, Flask API, and local job runner.

Out of scope in this repository:

- pose-estimator runtime orchestration,
- estimator output conversion to BOP result CSVs,
- BOP Toolkit evaluation,
- metric report/export dashboards.

## Completed In This Acquisition-Only Pass

- Replaced the generic calibration-stage card grid with a purpose-built,
  acquisition-only two-mode workflow:
  - exactly two authoritative modes cover robot-mounted cameras observing a
    stationary target and static cameras observing a flange-mounted target,
  - the façade exposes setup, immutable history, attempt detail, and explicit
    promotion endpoints while queueing one CPU/disk parent job and never
    initiating capture,
  - attempt preparation synchronizes only selected captured sensor folders and
    reuses compatible intrinsics or captured factory sidecars under
    `processed/calibration/<attempt_id>/`, preserving raw and prior evidence,
  - reusable synchronization, observation, and solver stage internals now
    accept explicit subsets and alternate derived output roots while their
    existing run-wide CLI/API defaults remain unchanged,
  - planar IPPE/ITERATIVE/SQPNP share a robust point mask, use LM refinement,
    retain IPPE alternatives, and reject non-finite or non-cheiral hypotheses,
  - both geometries recover camera and companion target transforms across
    Tsai, Park, Horaud, Andreff, Daniilidis, Shah, and Li, with deterministic
    robust-closure outlier rejection, leave-one-pose-out validation, and stable
    score/tie ranking,
  - results expose transforms, matrices, quaternions, translations, counts,
    reprojection and held-out residuals, every candidate/failure, passing
    overrides, and partial multi-camera acceptance,
  - promotion transactionally mirrors canonical evidence, merges one valid
    profile per accepted camera, preserves unrelated profiles, records full
    solver/operator/attempt provenance, and updates selected-camera mounting
    metadata,
  - saved target browsing and selection remain available when PoseGridGen
    generation is unavailable; stage-level APIs and CLIs remain advanced
    diagnostics.

- Integrated the pinned PoseGridGen calibration-target workflow:
  - upgraded the project contract and lock to Python `>=3.12,<3.13`, Pydantic
    2, NumPy 2, SciPy, Pillow, and ReportLab while retaining the single
    `opencv-python` wheel,
  - committed PoseGridGen as an exact, clean submodule at
    `ad152e369e8d2746d0cf66cb1455f2371b0ec0f0` and added lazy source-only
    verification plus an isolated backend namespace with no FastAPI import,
  - added `calibration_target.v2` with authoritative compensated marker
    corners, exact bounds, geometry/configuration hashes, v1 expansion, and
    generic `cv2.aruco.Board` construction,
  - added immutable UUID source/spec/PDF bundles, hash/symlink/containment
    validation, rollback-capable selection, three explicit placements, and
    concrete target-dependent replacement blockers,
  - propagated target identity through detection, intrinsic, pose, coverage,
    observation, candidate, and solver evidence with mismatch rejection,
  - added queued generation/selection jobs, scoped Flask APIs, run/calibration
    preflight checks, exact Cell bounds, and selected-target workflow context,
  - added a native lazy React page for form editing, fit, debounced preview,
    generation, downloads, confirmed library deletion, placement selection,
    run switching, and unavailable source-checkout guidance,
  - updated the installer, operator documentation, committed web build, unit/API
    coverage, and mocked Playwright coverage without accessing lab hardware.

- Replaced the Bootstrap/Jinja/vanilla-JavaScript transition page atomically
  with a Bun-locked React, TypeScript, Vite, Tailwind, and Radix-based shadcn
  operator console:
  - fixed desktop navigation for Dashboard, Devices, phase-based Workflow,
    and Jobs with a global contained run picker,
  - system/light/dark theming and local selected-run/robot-target persistence,
  - `web_bootstrap.v1` and symlink-safe, newest-first `web_run_index.v1` APIs,
  - device cards with aliases, mounting/orientation, card-local previews,
    queued snapshots, selection, raw detail sheets, and separated IIWA controls,
  - metadata-generated stage forms, artifact statuses, plan-only setup defaults,
    preflight blockers, preview shutdown before camera work, and a fresh two-gate
    physical-capture dialog,
  - active-first jobs, live logs, cancellation, and immediate capture-stop
    controls,
  - committed hashed UI assets included in wheels, installer verification, and
    opt-in `scripts/install.sh --with-web-build`,
  - theme-aware MATCH COW branding sourced from ArUcoGridGen, with dedicated
    light/dark logos and the current cow favicon,
  - lazy read-only Cell route with a demand-driven React Three Fiber canvas,
    Z-up millimetre coordinates, view presets, layers, selection provenance,
    exact paged timeline playback, and a WebGL-free component-list fallback,
  - Flask discovery/static/package tests plus mocked Playwright coverage.

- Implemented `docs/INTERACTIVE_3D_CELL_PLAN.md`:
  - added one validated object registry for the viewer, BlenderProc, and BOP,
    with stable full-registry IDs, explicit `run_config.v1.selected_objects`,
    legacy fallback warnings, safe PLY/texture containment, and rigid-transform
    inversion provenance,
  - propagated subset and explicit objectless operation through configuration,
    sequence option injection, transactional preparation, skipped rendering,
    BOP GT/models/masks/targets/COCO output, v2 manifest provenance, and the
    readiness gate,
  - added pytransform3d-composed `cell_scene.v1`, exact non-interpolated paged
    `cell_timeline.v1`, static/eye-in-hand frustums, fixed base/TCP frames,
    targets, objects, bounded trajectory previews, and unresolved entities,
  - packaged the 420 × 297 mm HRI SVG and added allowlisted conditional asset
    routes plus Cell navigation, inspection, playback, and fallback behavior.

- Implemented `docs/REAL_ONLY_ROBOT_ACQUISITION_CLEANUP_PLAN.md`:
  - collapsed robot configuration and `robot_status.v2` to the real lab iiwa,
  - removed robot-mode flags, selectors, environment mode selection, fake
    controller/rehearsal/synthetic capture workflows, and their artifacts,
  - reduced capture planning to enabled sensor commands followed by exactly one
    pose receiver and reduced execution to supervised full capture,
  - retained independent `allow_real_robot` and `allow_cameras` gates,
  - made `real_full_capture_validation` the default non-executing run sequence,
  - kept the RealSense smoke camera-only and independent of robot profile,
  - reduced rewrite status to the three real-data gates.

- Completed the rewrite hardening audit in `docs/REWRITE_HARDENING_PLAN.md`:
  - atomic JSON/report/manifest writes and rollback-capable directory promotion,
  - raw-capture overwrite refusal and validated paired frame writes,
  - process-group capture cancellation plus hierarchical job resource locks,
  - transactional `sync_report.v2` / `sync_quality_report.v2` with timestamp
    provenance and source-fallback gates,
  - stricter finite/unique calibration validation and merge-preserving
    promotion,
  - importable transactional BlenderProc preparation/render orchestration,
  - standard transactional BOP-scenewise export with
    `bop_export_manifest.v2`, root dataset/frame metadata, cross-file
    validation, depth-aware GT info, and exact model diameters,
  - a testable `zed_2i_capture_summary.v1` adapter,
  - web path containment, strict booleans, scoped pipeline paths, and anchored
    job storage,
  - installable web assets/entry point, Ruff, and legacy/dependency cleanup.

- Implemented `docs/ARUCO_GRID_CALIBRATION_PLAN.md`:
  - exact ArUcoGridGen 1.0 import into `calibration_target.v1`, including board,
    dictionary, contiguous-ID, physical-scale validation, source SHA-256, a
    marker-0 top-left grid frame, and optional aligned identity placement,
  - explicit `robot_flange -> template_base` run-config frames, typed fixed
    transforms, and legacy-frame warnings,
  - split native-RGB detection and grid-to-camera PnP/LM pose phases with
    `aruco_detections.json` and enriched pose provenance,
  - `intrinsic_calibration.v1` factory/calibrated color profiles with 15-view,
    6/9-coverage, 3 px per-view, and 1.5 px RMS default gates plus alpha=0
    rectified projections and unchanged SDK depth provenance,
  - `calibration.v2` profiles with native/rectified projections, explicit
    camera-to-flange or camera-to-template-base endpoints, and v1 loading,
  - unknown-target, known-target, and comparison extrinsic modes, static-mode
    observability rejection, cross-method gates/override recording, and
    fixed-edge camera-to-TCP derivation,
  - repeatable per-sensor profile selection with exactly-one promotion,
  - transactional RGB/nearest-depth rectification under
    `processed/rectified/`, with metadata/pose preservation and strict profile
    identity matching,
  - rectified BlenderProc/BOP consumption and projection provenance in
    `scene_camera.json`,
  - queued target, detection, intrinsic, pose, solver/comparison, selection,
    and rectification stages plus `aruco_grid_full_calibration` and
    `calibrated_capture_to_bop_dataset_dry_run` sequences,
  - synthetic recovery, rejection-gate, selection, legacy-loading,
    rectification, pipeline, and BOP integration tests.

- Reworked the iiwa calibration-variance proposal into a teachable Workbench
  program under `/PoseTestBot/TemplateBase`:
  - versioned nine-frame teaching manifest covering the complete 3 × 3 raster,
    with `CalibrationCenter` as the shared phase anchor,
  - Sunrise `ObjectFrame` resolution during initialization with no runtime
    numeric absolute targets, early missing-frame failure, center-anchored
    phases, and the commissioning interlock still disabled,
  - program-owned zero-translation A/B/C dither using nine `LIN_REL` legs
    relative to the taught center; the former six orientation frames, two depth
    frames, Ready frame, and depth phase are removed,
  - reproducible equal-scale Matplotlib SVG/PNG engineering plot with the
    measured 420 × 297 mm template, exact taught poses, derived orientation
    triads/deltas, sequence views, and explicitly non-metric ceiling-cell
    context,
  - printable nine-row Workbench teaching, relative-path commissioning, T1, and capture
    acceptance checklist,
  - manifest/Java/delta consistency, KUKA degree conversion, known orientation,
    and headless plot artifact tests. No robot or camera was accessed.

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
  - calibration and sync-to-BOP sequences
  - real full capture validation
- Replaced rewrite gates with acquisition-only gates:
  - `rewrite_full_capture.v1`
  - `rewrite_calibration_validation.v1`
  - `rewrite_bop_export_readiness.v1`
- Reworked recommendations to suggest acquisition steps only.
- Removed the standalone Artifacts page and its Flask browsing/preview APIs;
  operators use the focused workflow/readiness views while filesystem
  inspection remains available for troubleshooting.
- Removed Flask metric dashboard and BOP result CSV endpoints.
- Added per-RealSense inverted-mount capture support that rotates saved RGB-D
  frames 180 degrees, corrects intrinsics, and carries orientation metadata
  through run configs, capture plans, smoke reports, manifests, and the web UI.
- Moved the Flask UI behind `posetestbot.web`, with `web_interface.py` kept as
  a compatibility shim, and added a sidebar workflow overview backed by
  run artifacts, recommendations, jobs, and sequence plans.
- Added sidebar manual IIWA Start/Stop controls that queue non-blocking
  real-robot command jobs with an operator-set robot IP/port target.
- Added an auto-starting sidebar monitor for the UGREEN USB webcam
  (`0c45:2283`) below the IIWA controls. It resolves the current V4L2 node by
  USB identity and captures through a resource-declared background job.
- Replaced that monitor's 1 Hz JPEG polling with a video-only WebRTC worker:
  MJPEG 640×480/30 capture with a one-frame V4L2 buffer, timestamped PyAV
  frames, unbuffered aiortc relay fan-out, VP8-first negotiation, loopback-only
  aiohttp signaling proxied by Flask, `monitor_webrtc.v1` status, and bounded
  browser restart/renegotiation. RGB-D sensor previews remain JPEG-based, and
  Playwright exercises the real peer connection with a synthetic track rather
  than lab hardware.
- Added a monitor-only WebRTC hardware smoke command with a non-hardware plan
  mode and explicit operator/camera/physical-execution gates. It checks the
  selected UGREEN node, active MJPG 640×480/30 negotiation, connected WebRTC
  media with advancing frames, and clean peer, signaling, process, and camera
  release without importing the global web job runner or contacting the robot.
- Made web shutdown cancel and join all locally owned jobs, persisted process
  identity for safe orphan cleanup after an interrupted restart, and added
  release-and-retry behavior when opening V4L2 preview nodes. This prevents the
  sidebar webcam and sensor previews from retaining camera devices after exit;
  recovery also releases verified legacy orphans whose jobs were already marked
  terminal by an older server instance.
- Made the React dashboard replace one persisted terminal UGREEN monitor job on
  page load, so V4L2 node renumbering after a reboot cannot leave the room
  monitor stuck on a stale failure. Automatic retries remain bounded to one per
  page load.
- Changed sensor status to detection-first by default, while preserving
  explicit expected-count checks for CLI/preflight use.
- Added ignored lab-local sensor aliases in `working_data/sensor_aliases.json`
  and queued live RGB preview streams under `working_data/sensor_previews/`.
- Changed live RGB preview controls to card-local stream slots with per-sensor
  toggles, terminal error retention, selected-node/frame metadata, and current
  RealSense inverted orientation restart behavior.
- Added Playwright browser coverage for the sensor preview DOM workflow while
  keeping browser binary installation opt-in.
- Fixed the React device cards to prefer an active RealSense preview over stale
  terminal history for the same sensor. Playwright now covers the production
  newest-first job ordering and requires the active preview JPEG to render.
- Prevented persisted/failed RealSense preview jobs from displaying their last
  JPEG as a live frame, disabled browser caching for rolling preview images,
  and locked preview images out of card layout flow. Playwright covers all
  three RealSense previews while the lower OAK-D Pro and ZED controls remain
  scrollable and selectable in a 1280×720 operator viewport.
- Contained OAK-D Pro preview images, errors, and long DepthAI source IDs inside
  their card-local slot so toggling that preview cannot resize or obscure the
  remaining Devices page. Playwright covers the full OAK start/render/stop
  lifecycle while keeping all five lab sensor cards reachable.
- Kept OAK-D Pro visible during live preview by reconciling healthy preview job
  sensor specs into the web sensor status while DepthAI exclusively owns the
  camera and omits it from discovery. Flask and Playwright regressions cover
  that claimed-device transition. A real browser acceptance kept the 640-pixel
  live frame and card visible across periodic discovery refreshes, then stopped
  and released the preview cleanly; no acquisition or robot command ran.
- Audited the dashboard, Devices, Setup, Preflight, and Capture controls for
  operator-console semantics: RealSense previews are explicit pressed-state
  toggles with transition locking, repeated snapshot/stop requests are guarded,
  stop-all failures are visible, robot start and stop share validated target
  confirmation, and capture/preflight controls expose loading states.
- Aligned the operator console with the compact MATCH engineering-tool theme:
  exact neutral and lime semantic tokens, denser shared controls and cards,
  quieter selection states, and an accessible persisted light/dark toggle.
- Made run setup snapshot exactly the cameras selected on Devices, reject
  missing/disconnected selections, and preserve each camera's saved static or
  eye-in-hand mounting mode instead of overwriting the lab layout globally.
  Required stage inputs, form labels, evidence refresh, and safe default button
  types now have browser-facing validation and accessibility semantics.
- Polished the transition web UI empty-run overview state and sidebar branding.
- Matched the operator-console sidebar to the selected light/dark theme, removed
  the padded logo backing, and simplified the trusted-network note.
- Added a persisted process supervisor for every local web job. Workloads and
  descendants now have verified process-group identities, Linux parent-death
  cleanup, restart orphan recovery, and a shared five-second TERM/KILL shutdown
  window across every supported web entry point.
- Promoted the UGREEN WebRTC monitor to a hidden managed service with lazy V4L2
  ownership, `monitor_webrtc.v2` heartbeats and frame/peer health, idle release,
  timed peer cleanup, automatic stale-worker replacement, and a configurable
  local STUN binding responder for numeric Chrome ICE candidates.
- Added browser-triggered UGREEN monitor brightness auto-calibration without a
  second camera owner or blocking Flask hardware handler. The managed monitor
  worker reads the signed V4L2 brightness range, runs a bounded central-frame
  luma search while streaming, publishes `monitor_brightness.v1` progress and
  the selected value, and exposes a guarded Auto brightness control with
  synthetic WebRTC Playwright coverage.
- Added OAK-D Pro 640×480/6 fps RGB preview through a non-blocking, one-frame
  DepthAI v3 queue while retaining aligned 720p RGB-D snapshots. RealSense and
  OAK preview reuse now rejects stale heartbeat artifacts.
- Updated the operator console to use the monitor's advertised STUN service,
  retry failed negotiation after 1/3/10 seconds, preserve the concrete final
  error, and reset the bounded retry budget only on manual Retry. Rebuilt the
  checked-in production assets without discarding existing console changes.
- Fixed the room monitor's false-positive connected state: browser ICE
  connectivity now remains `receiving` until a camera frame is decoded and
  rendered. A five-second first-frame watchdog reports packet/receive/decode
  counters and exercises the bounded reconnect path instead of exposing the
  empty grid indefinitely.
- Fixed real UGREEN streams that delivered RTP packets but no complete browser
  frames over a 1280-byte Tailscale path. The dedicated worker now caps aiortc
  VP8 payloads at 1100 bytes, publishes that limit in monitor status, and tests
  browser playback with textured multi-packet frames instead of a flat-color
  fixture that could not expose MTU fragmentation.
- Completed an operator-authorized real UGREEN acceptance after the MTU fix.
  Chromium rendered the live stream at 640×480 with `readyState=4` and advancing
  playback time; a second receiver decoded 35/35 frames; V4L2 reported MJPEG
  640×480 at 30 fps; and the validation peer detached while the operator peer
  and media counters remained healthy. No robot or acquisition-pipeline command
  was executed.
- Reconciled RealSense SDK serials with USB/V4L2 node metadata so three
  connected D435-class cameras appear as three devices, not duplicated SDK and
  USB entries.
- Completed a supervised real-iiwa hot capture with all three lab RealSense
  cameras and passed `rewrite_full_capture.v1` (10/10 checks). The run preserved
  7,499 robot poses and balanced RGB/depth/metadata counts for every sensor.
- Added a disabled-by-default Sunrise calibration-variance proposal with a
  3x3 Cartesian coverage raster, depth sweep, and three-axis orientation
  dither, plus commissioning and image-coverage acceptance guidance based on
  the latest real-capture diagnostics.
- Made RealSense `SIGTERM` shutdown graceful so the capture supervisor cannot
  interrupt an RGB-D tuple between image and metadata commits; control-flow
  exceptions now also roll back partially committed frame files.
- Fixed sync-quality report discovery for relative run roots, which previously
  prepended the run folder twice after successfully discovering reports.
- Added a conservative installation script and `INSTALL.md` covering uv setup,
  optional BlenderProc, vendor SDK caveats, and acquisition readiness checks.
- Updated root agent notes, README, and system overview for the acquisition
  boundary.
- Rewrote stale downstream tests into acquisition-only coverage.

## Current Gates

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

- validated `bop_export_manifest.v2` using `bop/<split>/<scene_id>`,
- each scene folder containing exact RGB/depth sets, `scene_camera.json`,
  `scene_gt.json`, and `scene_gt_info.json`,
- root `dataset_info.json` and `posetestbot_bop_frame_map.json`,
- `bop/test_targets_bop19.json` with at least one target row,
- `bop/models/models_info.json` with positive exact diameters,
- valid calibration provenance and correct target/model/scene references.

Explicit objectless exports instead require empty targets and per-frame GT,
with no model or mask trees. Valid calibration, RGB-D integrity, scene metadata,
and frame-map provenance remain required.

## Validation Commands

Targeted acquisition tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_iiwa_teaching_plan.py \
  tests/test_iiwa_teaching_plot.py \
  tests/test_runtime_status.py \
  tests/test_hardware_status.py \
  tests/test_manifest.py \
  tests/test_pipeline_stages.py \
  tests/test_pipeline_sequences.py \
  tests/test_preflight.py \
  tests/test_rewrite_gate.py \
  tests/test_pipeline_recommendations.py \
  tests/test_web_interface.py \
  tests/test_web_preview_playwright.py
```

Full validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
MPLCONFIGDIR=/tmp/posetestbot-mpl UV_CACHE_DIR=/tmp/uv-cache \
  uv run python scripts/plot_iiwa_calibration_teaching_plan.py
git diff --check
uv run python scripts/run_pipeline_sequence.py working_data/new_real_run \
  --sequence real_full_capture_validation --plan-only
```

## Remaining Work

- Run the safety-gated camera lifecycle acceptance from
  `docs/CAMERA_SERVICE_LIFECYCLE_PLAN.md` on the operator-ready lab host through
  both LAN and Tailscale. The standalone UGREEN/browser leg now passes; the
  remaining combined acceptance must exercise it concurrently with all three
  RealSense devices and OAK-D Pro through the full restart matrix.
- On an operator-ready lab host with all configured cameras visible, create a
  fresh `0.05 m/s` run, inspect `real_full_capture_validation` with
  `--plan-only`, deliberately execute it, and require
  `rewrite_full_capture.v1` to pass.
- Import/compile `PoseTestBot_CalibrationVarianceProposal.java` in the real
  Sunrise.Workbench project, create and teach all 18 persistent Application
  Data children, resolve every frame, and complete offline path simulation with
  the printable checklist. Then perform separately authorized T1 validation;
  neither step is automated by repository tests.
- Promote robust calibration profiles from real observations.
- Run BOP export readiness gates on real captured/calibrated datasets.
- Keep improving live capture telemetry from real operator feedback.
