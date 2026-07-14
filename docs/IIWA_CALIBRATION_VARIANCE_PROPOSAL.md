# IIWA Calibration-Variance Program Proposal

> Historical design record. The robot-mode environment selector shown below
> was retired by the real-only acquisition cleanup and is not a current command.

## Outcome

`iiwa/PoseTestBot_CalibrationVarianceProposal.java` is a separate Sunrise
application proposal for collecting ArUco calibration images with materially
more image-space, distance, and orientation variance than
`iiwa/PoseTestBot_Test.java`.

It is deliberately disabled in source. None of its proposed frames or paths
has been validated on the physical cell.

## Motivation

`PoseTestBot_Test` changes only A1 from -169 to +169 degrees while preserving
the other six joints. That produces many robot poses, but it does not guarantee
useful camera observations. In the diagnostic run
`working_data/hot_full_capture_fixed_20260710_1351`, all 1,490 synchronized
frames per RealSense detected the board, yet every board centroid stayed in
image coverage cell 4, the center of the 3x3 grid. The intrinsic-calibration
gate requires at least 6 of 9 cells.

At approximately 910 px focal length, moving a target by about 280 px
horizontally corresponds to roughly 17 degrees of optical-axis change. At an
estimated 500 mm working distance, a 160 mm translation without perfectly
re-aiming the camera produces a similar shift. This proposal therefore uses
translation as the primary coverage mechanism and smaller rotations for
additional calibration observability.

## Proposed Sequence

All coordinates are millimetres and KUKA A/B/C degrees relative to
`/HRC_Hub/Template_Base`.

| Phase | Proposed motion | Intended variation |
| --- | --- | --- |
| Coverage raster | Nine waypoints at X = -160/0/+160 and Z = 355/445/535 | Cross image left/center/right and upper/middle/lower thirds |
| Depth sweep | `(0, -360, 600)` to `(0, -230, 350)` and back | Board scale, depth, and oblique-view variation |
| Orientation dither | Fixed XYZ with A/C +/-15 degrees and B +/-12 degrees | Rotation-axis and perspective diversity for intrinsics and hand-eye solving |

The raster is traversed continuously in a snake pattern. Each linear leg and
each orientation motion has a distinct `motion` label in the UDP pose stream,
so observations can later be grouped or rejected by phase.

The positions are deliberately inside the broad envelope of the legacy
`HRC_Hub_Cap.java` Center, CenterClose, LeftClose, RightClose, Top, and Bottom
frames. This is only a starting point; it is not evidence that the new frames
or their connecting paths are safe or reachable.

## Velocity Contract

The PoseTestBot receiver sends `cartesian_velocity_m_s`. The proposed program
converts that value to the millimetres-per-second unit expected by Sunrise
Cartesian motion and clamps it to 20--80 mm/s. A first image trial should use
40 mm/s:

```bash
POSETESTBOT_ROBOT_MODE=real uv run python scripts/pose_receiver_udp_json.py \
  working_data/<new_run> --capture_vel 0.04
```

That command is an operator action for an intentionally selected real-robot
trial; it has not been run as part of this proposal.

## Required Commissioning

Before physical execution:

1. Import the class into the controller's Sunrise.Workbench project without
   replacing `PoseTestBot_Test`.
2. Confirm the active tool, load data, flange-to-camera mounts, and
   `/HRC_Hub/Template_Base` definition match the real cell.
3. Validate every waypoint and every PTP/LIN connection in the offline cell,
   including joint limits, singularities, self-collision, fixtures, target,
   camera bodies, and cable routing.
4. Leave `ENABLE_AFTER_OFFLINE_VALIDATION = false` until that review passes.
5. For T1 commissioning, enable only one of `RUN_COVERAGE_RASTER`,
   `RUN_DEPTH_SWEEP`, and `RUN_ORIENTATION_DITHER` at a time, use reduced
   override, and single-step with an operator at the enabling device.
6. Confirm all mounted cameras keep enough of the board in view at each
   extreme. Reduce the corresponding X/Z or A/B/C offset if any camera loses
   the board.
7. Only after all three phases pass separately, enable the combined sequence.

The UDP stop message is only read while the application is waiting for a new
start command. It does not interrupt an active Sunrise motion, and it must not
be treated as a safety stop. Normal controller safety functions remain the
only safety controls for commissioning.

## Capture Acceptance

A short trial should be analysed before a full calibration capture. For every
camera, require:

- at least 15 accepted views;
- at least 6 of the 9 image-centroid coverage cells;
- strong detections at every raster extreme, with the full board preferred;
- per-view reprojection error no greater than 3 px;
- final intrinsic RMS no greater than 1.5 px;
- sufficiently varied robot translations and rotations for the selected
  extrinsic mode;
- a passing synchronization-quality report.

The previous diagnostic also found approximately 4.06 px mean board PnP
reprojection error for RealSense `825412070181`, versus about 0.67 px and
0.60 px for the other two sensors using factory intrinsics. More trajectory
variance does not by itself fix that discrepancy; inspect that camera's
intrinsics, image orientation, mount rigidity, and synchronization separately.

The exact ArUcoGridGen JSON export is still required before producing physical
intrinsic or extrinsic calibration results. Detection-only coverage checks can
be used during motion commissioning without treating them as calibration.
