# IIWA Nine-Frame Teaching and Commissioning Checklist

Print this checklist for Sunrise.Workbench teaching and physical commissioning
of `iiwa/PoseTestBot_CalibrationVarianceProposal.java`. The complete frame and
relative-motion contract is in
[`iiwa/calibration_teaching_plan.v2.json`](../iiwa/calibration_teaching_plan.v2.json).

> Teaching aid only—not reachability, redundancy, singularity, collision, or
> cable-clearance validation. Physical work requires the normal lab risk
> assessment, controller safety functions, an authorized operator, and a
> reviewer. Keep `ENABLE_AFTER_OFFLINE_VALIDATION=false` until the entire
> offline commissioning section passes.

## Record Before Creating Frames

| Record | Value |
| --- | --- |
| Controller | |
| Sunrise.OS version | |
| Workbench project revision | |
| PoseTestBot repository revision | |
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
- [ ] Select the robot flange as the teaching and motion point.
- [ ] Confirm `ENABLE_AFTER_OFFLINE_VALIDATION=false`.
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
- [ ] Confirm all starts/stops use the expected 3% acceleration/jerk limits,
  each leg gets a 1.5-second vibration dwell, the orientation dither uses 4%
  relative joint velocity, and the raster uses the expected 60%-scaled
  Cartesian velocity (8–45 mm/s), without treating those limits as safety
  functions.
- [ ] Confirm each dwell is followed by `_settled` robot-pose samples and that
  synchronized camera frames include sharp, stationary calibration views.
- [ ] Stop for unexpected joint branches, position drift, cable tension,
  clearance loss, target loss, vibration, or other unreviewed behavior.
- [ ] Confirm UDP stop messages cannot interrupt active motion and are not
  safety controls. Use only controller safety functions for safety response.

| T1 evidence | Result / location | Operator / reviewer / date |
| --- | --- | --- |
| Initial center anchor | | |
| Coverage phase | | |
| Relative orientation phase | | |
| Combined sequence | | |

## Capture Acceptance — Operator Run Only

- [ ] Obtain explicit operator authorization.
- [ ] Pass both PoseTestBot execution gates: `--allow-real-robot` and
  `--allow-cameras`.
- [ ] Run capture preflight and inspect the execution plan before deliberately
  starting physical execution.
- [ ] Run a short supervised trial before a full calibration capture.
- [ ] For every required camera, verify:
  - [ ] at least 15 accepted views;
  - [ ] at least 6/9 image-centroid coverage cells;
  - [ ] strong target detections at all raster extremes;
  - [ ] per-view reprojection error no greater than 3 px;
  - [ ] intrinsic RMS no greater than 1.5 px;
  - [ ] sufficient translation and rotation diversity;
  - [ ] passing synchronization quality.
- [ ] Keep the prior high-error RealSense `825412070181` investigation
  separate. Trajectory variance alone does not correct its reprojection
  discrepancy.

| Camera / serial | Accepted views | Coverage cells | Max per-view px | RMS px | Extremes detected | Sync quality | Reviewer / date |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

## Final Disposition

| Decision | Selection / notes |
| --- | --- |
| Keep `ENABLE_AFTER_OFFLINE_VALIDATION=false` | ☐ |
| Offline commissioning approved | ☐ |
| T1 commissioning approved | ☐ |
| Supervised trial accepted | ☐ |
| Frames requiring retouch | |
| Paths invalidated by retouch | |
| Operator / date | |
| Reviewer / date | |
