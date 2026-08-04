# IIWA Single-Frame Static-Camera Calibration Application

## Outcome and Status

`iiwa/PoseTestBotSingleFrameStaticCameraCalibrationApplication.java` is the
repository alternative for calibrating a fixed camera with a calibration
target rigidly mounted on the robot. It requires one additional taught motion
frame and generates the remaining poses as bounded relative motions.

This source is not deployment or physical-commissioning evidence. It must be
added to the exact Sunrise.Workbench project, compiled against the installed
Sunrise.OS API, simulated, commissioned in T1, and selected on the controller
by an authorized operator. Repository validation never commands the robot.

## One-Frame Contract

The existing persistent `/PoseTestBot/PoseTemplateBase` remains the application
base, pose-stream reference, and destination frame of the reusable static
`camera -> PoseTemplateBase` result. Create and teach only this additional
motion frame:

```text
/PoseTestBot/PoseTemplateBase/CalibrationStatiCenter
```

`CalibrationStatiCenter` intentionally preserves the exact operator-specified
spelling. Teach it with the robot-carried target centered and fully visible in
the intended static camera, with useful border margin for translation and
rotation. The program does not construct numeric absolute frames at runtime.

All translations are expressed in the taught center's axes. “Upper”, “lower”,
“left”, “depth plus”, and “depth minus” are route labels, not promises about
camera-image direction or camera distance. Verify their actual meaning in
Workbench and in every required camera.

## Relative Motion Plan

The planar grid uses 65 mm half-spans. A corner is
`sqrt(65^2 + 65^2) = 91.924 mm` from center, leaving margin below the 100 mm
limit.

| Generated point | X | Y | Z | Distance from center |
| --- | ---: | ---: | ---: | ---: |
| Upper left | -65 mm | +65 mm | 0 | 91.9 mm |
| Upper center | 0 | +65 mm | 0 | 65 mm |
| Upper right | +65 mm | +65 mm | 0 | 91.9 mm |
| Middle right | +65 mm | 0 | 0 | 65 mm |
| Lower right | +65 mm | -65 mm | 0 | 91.9 mm |
| Lower center | 0 | -65 mm | 0 | 65 mm |
| Lower left | -65 mm | -65 mm | 0 | 91.9 mm |
| Middle left | -65 mm | 0 | 0 | 65 mm |
| Depth plus | 0 | 0 | +50 mm | 50 mm |
| Depth minus | 0 | 0 | -50 mm | 50 mm |

Each generated translation is a blocking `LIN_REL` from center and is followed
by its exact inverse `LIN_REL` back to center. The program runtime-checks each
relative translation against the 100 mm envelope before commanding it. It then
visits independent -10° and +10° results about A, B, and C, returning to center
after each orientation.

The 100 mm bound applies to the flange origin's translation. It is not a bound
on every point of the robot, tool, mounted target, or cables. Orientation can
sweep target corners outside that sphere, so the complete robot-and-target
swept volume still requires collision and clearance validation.

## Runtime Sequence

Launching the application causes no robot motion. It resolves the shared
`PoseTestBotPoseStreamTask`, binds UDP port 30300, and waits for an accepted
positive-velocity START command. After START it:

1. queries the current flange pose without motion and rejects the run unless it
   is already within 25 mm of `CalibrationStatiCenter`;
2. configures pose streaming in `/PoseTestBot/PoseTemplateBase`;
3. moves PTP to the taught `CalibrationStatiCenter` anchor;
4. visits and returns from all eight planar grid points;
5. visits and returns from the two depth points;
6. visits and returns from the six orientation results;
7. performs a final blocking PTP to the taught center; and
8. sends the successful terminal marker only after that return completes.

Relative motions use 60% of the requested Cartesian speed, clamped to
8–30 mm/s, plus 3% relative joint velocity, acceleration, and jerk limits.
Center PTP motions use 8% relative joint velocity and the same 3%
acceleration/jerk limits. Every leg has a 1.5-second dwell followed by three
settled pose samples. These are motion-quality settings, not safety-rated
limits.

The UDP socket is read only while idle. `STOP` cannot interrupt active motion,
is not a safety stop, and exits the waiting application. Never send it between
repeated calibration captures; use a new START after the prior capture has
completed.

## Workbench and T1 Commissioning

- Back up Application Data and record the exact controller, Sunrise.OS,
  Workbench project, repository revision, tool/load, target, operator,
  reviewer, and date.
- Rename/import the repository classes consistently in Workbench:
  `PoseTestBotFullCaptureApplication`,
  `PoseTestBotNineFrameCalibrationApplication`,
  `PoseTestBotSingleFrameStaticCameraCalibrationApplication`, and
  `PoseTestBotPoseStreamTask`. A repository rename does not update Workbench
  application or automatic-background-task metadata.
- Register `PoseTestBotPoseStreamTask` as the one automatic cyclic provider of
  `PoseTestBotPoseStreamFunction`, then compile all five Java sources.
- Read back the exact parent and center paths and center XYZABC/joint branch.
  Confirm the target is rigidly mounted and its attachment does not change
  during the recording.
- Before START, manually position the flange within 25 mm of the taught center.
  The program rejects a farther start without moving, but that proximity check
  does not prove the initial PTP joint-space path is collision-free.
- Simulate every PTP and `LIN_REL` endpoint and swept path. Check reachability,
  joint/redundancy branch, singularity margin, fixtures, the complete mounted
  target, arm, and cable clearance—not only the flange-origin envelope.
- Verify all grid/depth/orientation results keep the complete target detectable
  with useful image coverage in every selected static camera.
- With explicit operator authorization, single-step the entire sequence in T1
  at reduced pendant override before a supervised capture. Do not infer
  readiness from a successful status request or repository test.
- Retain v1 pose-stream identity/cadence evidence and the normal calibration
  solver, validation, and promotion artifacts. The resulting primary profile
  remains `camera -> PoseTemplateBase`; the estimated
  `aruco_grid -> robot_flange` attachment is supporting evidence, not a runtime
  hand-tracking product.

The installed Sunrise.OS Javadoc and exact Workbench project are authoritative
for the available `linRel(Transformation, ObjectFrame)` overload and motion
parameter setters.
