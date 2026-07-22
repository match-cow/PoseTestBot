# IIWA Nine-Frame Teaching and Commissioning Checklist

Print this checklist for Sunrise.Workbench teaching and physical commissioning
of `iiwa/PoseTestBot_CalibrationVarianceProposal.java`. The complete frame and
relative-motion contract is in
[`iiwa/calibration_teaching_plan.v2.json`](../iiwa/calibration_teaching_plan.v2.json).

> Teaching aid only—not reachability, redundancy, singularity, collision, or
> cable-clearance validation. Physical work requires the normal lab risk
> assessment, controller safety functions, an authorized operator, and a
> reviewer. The repository source is currently enabled for lab validation, but
> the exact deployed controller application and revision are not yet recorded.
> That boolean is not commissioning evidence: verify the deployed identity and
> retain the checks below whenever the application or cell changes.

> **Current program lifecycle:** do not send the UDP `STOP` command during a
> repeated calibration session. While the application is waiting, `STOP` exits
> the application and requires a manual application restart. It cannot
> interrupt active motion and is not a safety control.

## Record Before Creating Frames

| Record | Value |
| --- | --- |
| Controller | |
| Sunrise.OS version | |
| Workbench project revision | |
| PoseTestBot repository revision | |
| Enabled camera set / retained disabled cameras | |
| Operator | |
| Reviewer | |
| Date | |

- [ ] Back up and synchronize the Workbench project and Application Data.
- [ ] Verify `/PoseTestBot/TemplateBase` origin, axes, persistence, and physical
  relationship to the 420 × 297 mm template and ceiling-mounted robot.
- [ ] Do not copy old HRC-relative seeds unless base equivalence is proved or a
  measured transform is supplied.
- [ ] Confirm the actual camera rig, tool/load, payload and center of gravity,
  brackets, fasteners, cables, calibration target, and every required camera.
- [ ] Record the run-scoped enabled camera set. A temporarily unavailable
  camera may remain configured as disabled so its identity, mounting,
  orientation, and calibration-profile selection are preserved; do not count
  it as capture, calibration, Cell, or rewrite-gate evidence.
- [ ] Select the robot flange as the teaching and motion point.
- [ ] Confirm the repository and deployed application agree on
  `ENABLE_AFTER_OFFLINE_VALIDATION`, and record the exact deployed revision.
- [ ] Confirm this revision intentionally has no `CalibrationReady`, depth, or
  orientation-variant Workbench frames.
- [ ] Review the loss of the separate high-clearance Ready transit and approve
  `CalibrationCenter` as the program's start/end anchor.

## Create, Seed, and Touch Up

- [ ] Create exactly nine frames as direct children of
  `/PoseTestBot/TemplateBase`, using the exact spelling below.
- [ ] Synchronize/reload the Workbench project and verify all nine persist.
- [ ] Enter numeric values only as uncommissioned initial seeds. Never command
  an unvalidated seed automatically.
- [ ] Manually jog in T1 and teach `CalibrationCenter` first, then raster
  neighbors progressively outward from center.

At every taught frame:

- [ ] Record all seven joint values and available redundancy information.
- [ ] Check joint margin, redundancy, and singularity margin.
- [ ] Check robot-arm, camera-rig, fixture, and cable clearance.
- [ ] Confirm target detection in every required camera and useful image-space
  change.
- [ ] Confirm “upper,” “lower,” “left,” and “right” from camera images; do not
  assume TemplateBase signs map directly to pixels.
- [ ] Save and read back XYZABC, record a camera snapshot, and obtain
  reviewer/date sign-off.
- [ ] If a frame is retouched, invalidate and re-commission every connected
  absolute or relative path.

## Nine-Frame Sign-Off

| Frame path | Created | Seed entered | Touched | XYZABC read-back | 7 joints/redundancy recorded | Reach/joint/singularity OK | arm/rig/cable clearance OK | required cameras detect target | reviewer/date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/PoseTestBot/TemplateBase/CalibrationCoverageUpperLeft` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageUpperCenter` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageUpperRight` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageMiddleRight` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCenter` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageMiddleLeft` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageLowerLeft` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageLowerCenter` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| `/PoseTestBot/TemplateBase/CalibrationCoverageLowerRight` | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |

## Offline Path Commissioning

Use these abbreviations only in this section: Center = `CalibrationCenter`,
UL/UC/UR = upper-left/upper-center/upper-right, MR/ML = middle-right/middle-left,
and LL/LC/LR = lower-left/lower-center/lower-right.

- [ ] Compile the exact controller project's Sunrise.OS API level and confirm
  `Transformation.ofDeg(...)` plus `linRel(offset, calibrationCenter)` resolve.
- [ ] Confirm the controller API accepts the 3% relative joint-acceleration and
  joint-jerk limits on all motion types and the 4% relative joint-velocity
  limit on every orientation `LIN_REL` motion.
- [ ] Resolve `/PoseTestBot/TemplateBase` and all nine child frames.
- [ ] Validate every raster endpoint and swept path:
  `Center → UL → UC → UR → MR → Center → ML → LL → LC → LR → Center`.
- [ ] Confirm the center transits are PTP and the eight raster legs are LIN.
- [ ] Validate these zero-translation relative orientation legs in order:

  1. ΔA=-15° → A−15°
  2. ΔA=+30° → A+15°
  3. ΔA=-15° → Center
  4. ΔB=-12° → B−12°
  5. ΔB=+24° → B+12°
  6. ΔB=-12° → Center
  7. ΔC=-15° → C−15°
  8. ΔC=+30° → C+15°
  9. ΔC=-15° → Center

- [ ] Confirm all nine orientation motions are `LIN_REL` relative to the taught
  center and leave the flange XYZ fixed as intended.
- [ ] Verify the actual result orientation after every leg; do not rely only on
  accumulated Euler arithmetic.
- [ ] Confirm a fresh capture command first PTP-anchors at the absolute taught
  center, including after an interrupted or failed prior run.
- [ ] Verify every result's seven-joint solution, redundancy branch, joint and
  singularity margin, arm/rig/fixture clearance, cable clearance, and
  required-camera target visibility.
- [ ] Run offline path simulation for coverage, relative orientation, and the
  combined sequence.

| Offline evidence | Result / location | Reviewer / date |
| --- | --- | --- |
| Frame resolution and compile | | |
| Coverage endpoint/path simulation | | |
| Nine-leg relative orientation simulation | | |
| Combined sequence simulation | | |

## Physical T1 Commissioning — Operator Run Only

- [ ] Obtain explicit authorization for physical T1 commissioning.
- [ ] Confirm the actual tool/load, payload/CoG, cameras, brackets, target,
  fixtures, and cables match the reviewed Workbench cell.
- [ ] Manually position the robot at or near the taught center before the first
  start command. This is an operator requirement, not an enforced safety check.
- [ ] Single-step the initial PTP anchor to center at reduced override.
- [ ] Single-step Center→UL and LR→Center PTP transits before raster LIN legs.
- [ ] Single-step all nine relative orientation legs individually, checking the
  actual joint branch, fixed-origin behavior, camera visibility, and cables.
- [ ] Confirm every motion accelerates and decelerates with the expected 3%
  acceleration/jerk limits,
  each leg gets a 1.5-second vibration dwell, the orientation dither uses 4%
  relative joint velocity, and the raster uses the expected 60%-scaled
  Cartesian velocity (8–45 mm/s), without treating those limits as safety
  functions.
- [ ] Confirm each dwell is followed by `_settled` robot-pose samples and that
  synchronized camera frames include sharp, stationary calibration views.
- [ ] For unexpected joint branches, position drift, cable tension, clearance
  loss, target loss, vibration, or other unreviewed behavior, use the
  controller's approved safety response—not the UDP `STOP` message.
- [ ] Confirm UDP stop messages cannot interrupt active motion and are not
  safety controls. Use only controller safety functions for safety response.

| T1 evidence | Result / location | Operator / reviewer / date |
| --- | --- | --- |
| Initial center anchor | | |
| Coverage phase | | |
| Relative orientation phase | | |
| Combined sequence | | |

## Capture Acceptance — Operator Run Only

The current `calib00` campaign completed physical acquisition with two enabled
RealSense cameras while retaining the temporarily unavailable camera as
disabled configuration. Attempt `3c4a0b7b765f44bd9cc37fffc48fb321`
subsequently promoted the complete `IPPE + Horaud` bundle for serials
`825412070181` and `923322072633`. The pairwise stationary-target closure is
3.612 mm / 0.277 degrees. This is calibration evidence for the reduced
two-camera run only; it is not three-camera service/full-capture acceptance or
Sunrise commissioning evidence.

- [ ] Obtain explicit operator authorization.
- [ ] Pass both PoseTestBot execution gates: `--allow-real-robot` and
  `--allow-cameras`.
- [ ] Run capture preflight and inspect the execution plan before deliberately
  starting physical execution.
- [ ] Confirm Run Setup shows exactly the intended cameras enabled. After any
  enable/disable change, regenerate the capture plan and preflight rather than
  relying on stale evidence; at least one camera must remain enabled.
- [ ] Run a short supervised trial before a full calibration capture.
- [ ] Confirm the selected immutable target bundle is the physical printed and
  measured board. For the current campaign record `calib00`, target ID
  `15b49f67-7cf5-4c00-9e7f-914aa6ed5da0`, geometry SHA-256
  `3da681424ff77e55dc51c8c1c9bb58e0a425f7fa039b63d29c798aa2ad02b256`,
  and placement `unknown`; do not reuse detections from a different grid.
- [ ] For every required camera, verify:
  - [ ] at least 15 accepted views;
  - [ ] at least 6/9 image-centroid coverage cells;
  - [ ] strong target detections at all raster extremes;
  - [ ] at least 12 common PnP corner inliers and 50% whole-board support;
  - [ ] at least four supported markers with three corners each spanning two
    target rows and columns per view, and campaign coverage of at least 50% of
    markers and 60% of rows/columns (`calib00`: 18 markers, 3 rows, 5 columns);
  - [ ] whole-board and intrinsic per-view reprojection error no greater than
    3 px;
  - [ ] retain compatible factory intrinsics and the manual comparison
    evidence. Treat RealSense `inverse_brown_conrady` as compatible with a
    forward OpenCV projection only when every coefficient is finite and exactly
    zero; never pass nonzero inverse coefficients as forward distortion.
    Activate the manual fit only when factory projection is unusable and its
    training, five-view held-out, plausibility, 3 px/view, and 1.5 px RMS gates
    pass;
  - [ ] `intrinsic_comparison.json` retains the factory/manual candidates,
    deltas, selection reason, and any manual-calibration rejection;
  - [ ] at least four distinct motion poses, 20 mm translation span, and 5°
    rotation span;
  - [ ] rotation-axis samples of at least 2° and second/first singular ratio of
    at least 0.15 before and after pruning; fit at most five evenly spaced
    frames per motion and validate every accepted frame;
  - [ ] at least six hand-eye inliers, mean held-out residual no greater than
    10 mm / 5°, no more than 25% motion-balanced outliers, and no more than 25%
    outliers within any repeated motion; retain raw outlier density as evidence,
    not a promotion gate;
  - [ ] RealSense color `sensor_timestamp_ns` in SDK `global_time`, paired to
    robot `host_wall_timestamp_ns`, with zero manual offset, no timestamp
    fallback, and at most 20 ms nearest-pose delta.
- [ ] For two or more enabled cameras estimating the same stationary companion
  frame, require one complete bundle with the same PnP and extrinsic methods
  for every camera. Require every candidate to pass and maximum pairwise
  companion closure no greater than 10 mm / 5°. Among bundles within 0.01 of
  the best normalized mean individual score, use normalized companion closure
  for recommendation; do not let a clearly worse individual solution win on
  closure alone. Compare normalized ranking values at six decimal places and
  use canonical method order below that numerical precision. If no common
  bundle passes, do not promote a partial or mixed-method selection.
- [ ] After explicit promotion, confirm the Cell view lists the exact profile
  ID and camera-to-parent matrix/quaternion/translation with matching target,
  intrinsic, synchronization, solver, quality, and promotion provenance.
- [x] Confirm the prior high-error RealSense `825412070181` result did not
  recur: current factory/manual held-out RMS is 1.194/0.967 px and promoted
  mean reprojection is 1.035 px. Trajectory variance was not treated as a
  correction.
- [ ] Revalidate metric depth on RealSense `923322072633` after cable/firmware
  maintenance. The promoted RGB extrinsic remains valid, but saved depth-plane
  checks showed a range-dependent scale anomaly and factory depth
  scale/alignment is explicitly not recalibrated.

The hard calibration timing gates pass for both enabled cameras. The overall
sync report's warning is only the sub-0.8 whole-capture match ratio
(652/936 and 656/951 frames); frames outside the robot-motion intervals are not
calibration observations.

| Camera / serial | Run state | Accepted views | Coverage cells | Factory held-out max per-view RMS (px) | Manual-fit RMS / selection | Extremes detected | Sync quality | Reviewer / date |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| RealSense `033422071805` | disabled, retained | — | — | — | — | excluded | excluded | hardware follow-up pending |
| RealSense `825412070181` | enabled, promoted | 652 | 7/9 | 1.432 factory held-out | 0.973 train / 0.967 held-out; factory retained | yes | sensor global-time to host-wall, max 15.442 ms, no fallback | automated artifact validation / 2026-07-21 |
| RealSense `923322072633` | enabled, promoted | 656 | 6/9 | 1.910 factory held-out | 0.988 train / 1.013 held-out; factory retained | yes | sensor global-time to host-wall, max 17.094 ms, no fallback | automated artifact validation / 2026-07-21 |
| | | | | | | | | |
| | | | | | | | | |

## Final Disposition

| Decision | Selection / notes |
| --- | --- |
| Repository/deployed enable-state agreement recorded | ☐ |
| Offline commissioning approved | ☐ |
| T1 commissioning approved | ☐ |
| Supervised trial accepted | ☐ |
| Two-camera calibration promotion | `3c4a0b7b765f44bd9cc37fffc48fb321`, `IPPE + Horaud`, 2 valid profiles |
| Reduced-run rewrite gates | full capture 9/9; calibration validation 3/3 |
| Metric-depth disposition | `923322072633` depth-specific validation pending |
| Frames requiring retouch | |
| Paths invalidated by retouch | |
| Operator / date | |
| Reviewer / date | |
