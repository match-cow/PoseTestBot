# ArUco-Grid Intrinsic and Robot/Camera Extrinsic Calibration

## Summary

Build a non-destructive calibration workflow around the
[ArUcoGridGen JSON export](https://github.com/match-cow/ArUcoGridGen/blob/main/app.py),
supporting:

- Optional RealSense color-camera intrinsic calibration.
- Unknown-grid hand-eye calibration for wrist-mounted cameras.
- Known-grid calibration for wrist-mounted and static cameras.
- Side-by-side method comparison with explicit operator selection.
- Rectified derived RGB/aligned-depth data for BlenderProc and BOP.
- Explicit robot frames: the current iiwa stream is
  `robot_flange -> template_base`, not generic TCP/base.

## Contracts and Artifacts

- Add `calibration_target.json` (`calibration_target.v1`):
  - Import ArUcoGridGen `version: 1.0` JSON and require it for real
    calibration.
  - Validate ArUco grid type, dictionary, rows/columns, contiguous row-major
    IDs, positive marker dimensions, and exactly 100% horizontal/vertical
    scale.
  - Normalize to OpenCV
    `GridBoard(cols, rows, marker_length, separation)`.
  - Define the grid frame at marker 0's outer top-left corner: +X right, +Y
    down, +Z into the board.
  - Preserve the generator source and checksum, but ignore its optional
    page-coordinate transformation.
  - For aligned placement, record identity
    `aruco_grid -> template_base`.

- Extend `run_config.json` with:
  - `frames.robot_pose = {from: "robot_flange", to: "template_base", convention: "kuka_abc_radians"}`.
  - `frames.dataset_reference_frame = "template_base"`.
  - Optional typed fixed transforms such as flange-to-TCP or
    template-base-to-physical-robot-base.
  - Existing configurations without this section remain readable with a
    legacy-frame warning.

- Add `intrinsic_calibration_profiles.json` (`intrinsic_calibration.v1`)
  containing, per serial/resolution/orientation:
  - Native row-major K, image size, Brown-Conrady coefficients ordered
    `[k1, k2, p1, p2, k3]`, source, and quality metrics.
  - Alpha=0 rectified K, zero distortion, valid ROI, and unchanged output
    resolution.
  - Factory depth scale and SDK depth-alignment provenance; ArUco calibration
    does not recalibrate the depth imager or depth scale.

- Write final `calibration.v2` profiles with native and rectified intrinsics,
  explicit transform endpoints, method/provenance, and quality. Loaders
  continue accepting `calibration.v1` profiles and normalize them internally.

- Add derived artifacts:
  - Per-sensor `aruco_detections.json` with IDs and pixel corners.
  - Enriched `aruco_pose_estimation.json` with PnP inliers, reprojection error,
    target/profile provenance, and explicit frame convention.
  - `camera_rectification_report.json`.
  - Rectified data under `processed/rectified/<sensor>/`, never modifying raw
    or synchronized inputs.

## Implementation Changes

### ArUco Detection and Intrinsics

- Refactor ArUco processing into detection and pose phases:
  - Detect once from synchronized native RGB using the imported target.
  - For `--intrinsics-mode factory`, wrap captured SDK intrinsics in the typed
    profile.
  - For `--intrinsics-mode calibrate`, use `cv2.calibrateCameraExtended` over
    matched GridBoard corners, seeded by factory K.
  - Require at least 15 accepted views, at least 6/9 image-centroid coverage
    cells, per-view reprojection error at or below 3 px, and final RMS at or
    below 1.5 px. Report every rejected view and keep thresholds configurable.
  - Estimate `grid -> camera` with `solvePnPRansac` plus LM refinement using
    the selected native K and distortion.

### Extrinsic Solving

- Extend the extrinsic solver with explicit modes:
  - `hand_eye_unknown_target`: for wrist cameras, solve
    `camera -> robot_flange` without using grid placement; also estimate
    grid-to-template-base consistency.
  - `known_target`: use the aligned identity edge to solve camera-to-flange for
    wrist cameras or camera-to-template-base for static cameras.
  - `compare`: run both observable wrist-camera methods, emit separate
    candidates, and compare translation/rotation disagreement.
  - Reject unknown-target robot-relative calibration for static cameras as
    unobservable.
  - Preserve current defaults of at least six inliers, mean residuals at or
    below 10 mm/5 degrees, outlier ratio at or below 25%, and use 10 mm/5
    degrees as the default cross-method agreement gate.

### Validation and Promotion

- Make promotion unambiguous:
  - Add repeatable CLI selection
    `--select-profile SENSOR=PROFILE_ID` and an equivalent API mapping.
  - When two wrist candidates exist, validation refuses promotion without one
    explicit selection.
  - Cross-method disagreement beyond the configured gate blocks promotion
    unless the operator explicitly changes or disables that gate; record the
    override.
  - Promote exactly one valid profile per sensor/mount/rig slot.

### Rectification and Dataset Export

- Add transactional camera rectification:
  - Apply the same alpha=0 map to synchronized RGB and already color-aligned
    depth.
  - Use linear interpolation for RGB, nearest-neighbor for depth, and zero for
    invalid depth pixels.
  - Preserve frame names, matched robot poses, timestamps, and source
    provenance; write rectified K/zero-distortion sidecars.
  - Refuse profile resolution/orientation/serial mismatches.
  - Update BlenderProc preparation and BOP export to consume
    `processed/rectified`, resolve camera poses through the explicit frame
    graph, and record profile/projection provenance in `scene_camera.json`.

### Pipeline Integration

- Register target import, detection, intrinsic solving, pose solving,
  rectification, comparison, and selection options in the queued pipeline/job
  APIs.
- Add a full calibration sequence and calibrated capture-to-BOP sequence while
  keeping `sync_quality` immediately after `sync_run`.
- Retain the existing ArUco command as a factory-intrinsics compatibility
  wrapper.

## Test Plan

- Import representative ArUcoGridGen JSON and reject the wrong board type,
  unsupported dictionary/version, inconsistent IDs, or scaled geometry.
- Use synthetic projected grids to verify intrinsic recovery, coverage/error
  gates, serial-resolution-orientation matching, and factory fallback.
- Recover known synthetic camera-to-flange and camera-to-template transforms
  through both extrinsic methods; test static/unknown unobservability.
- Verify comparison metrics, mandatory per-sensor selection, disagreement
  blocking, fixed flange-to-TCP composition, and legacy profile loading.
- Verify rectification leaves source trees unchanged, preserves RGB/depth
  pairing and metadata, remaps depth without interpolation, and writes correct
  rectified sidecars.
- Exercise BlenderProc/BOP integration using rectified inputs and confirm
  ground-truth/profile frame provenance.
- Extend CLI, pipeline, Flask, and Playwright coverage as applicable.
- Finish with targeted calibration tests, full `uv run pytest`,
  `git diff --check`, fake acquisition/BOP smoke, and rewrite gates.

## Assumptions and Defaults

- "Grid aligned to robot base" means the selected grid frame is identical to
  `template_base`, because that is the parent frame currently streamed by
  Sunrise. Physical robot-base or TCP results require supplied fixed transform
  edges.
- The primary wrist result is camera-to-robot-flange; camera-to-TCP is
  additionally derived only when a flange-to-TCP edge is configured.
- Intrinsic calibration covers the RealSense color projection used by aligned
  RGB-D output. SDK depth scale and depth-to-color alignment remain
  authoritative.
- Real calibration requires the exact generator JSON and a physically printed
  board at 100% scale.
- Existing uncommitted repository work must be preserved and implementation
  merged into the current calibration rewrite rather than replacing it.
