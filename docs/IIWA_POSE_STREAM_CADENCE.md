# IIWA Pose-Stream Cadence

## Decision

The repository iiwa applications now request a **10 ms best-effort period
(nominally 100 Hz)** from a separate read-only Sunrise cyclic background task.
For this KLI/UDP path, **50 Hz measured end to end is the commissioning target**.
Falling below it produces actionable cadence evidence; it does not by itself
fail a calibration attempt. The retained 2026-07-28 calibration run delivered
only 16.38 Hz median during motion and is below that target.

This is source and offline evidence only. No robot command, camera access, or
physical capture was performed while making this change. The new task has not
yet been compiled in the exact lab Workbench project, deployed, or measured on
the controller.

## Why 10 ms, and What Is Actually Feasible

KUKA describes KLI as a non-real-time Ethernet interface for external
communication. The same KUKA manual describes cyclic background tasks as
parallel to the robot application and defines `BestEffort` behavior: when one
invocation exceeds its period, it finishes and the next invocation starts
immediately. It also prohibits background tasks from commanding robot motion
or changing motion-related parameters, while allowing them to query robot
data. See the KUKA-authored [Sunrise.OS 1.7 programming
manual](https://manualzz.com/doc/4516040/kuka-sunrise.os-1.7-robotersteuerung--sunrise.workbench-1...)
and [Sunrise.OS 1.11 system-integrator
manual](https://www.oir.caltech.edu/twiki_oir/pub/Palomar/ZTF/KUKARoboticArmMaterial/KUKA_SunriseOS_111_SI_en.pdf).

Those contracts support a 10 ms software request, not a hard 10 ms delivery
guarantee. The practical bands for this cell are therefore:

| Contract | Period / rate | Meaning |
| --- | ---: | --- |
| Old retained motion stream | 61.04 ms median / 16.38 Hz | Measured and inadequate. |
| Commissioning target | no slower than 20 ms median / at least 50 Hz | Preferred end-to-end rate with comfortable timing margin. |
| New cyclic target | 10 ms / 100 Hz nominal | Feasible software target with margin; must be measured on the lab controller. |
| Hard sub-10 ms or deterministic control | Not a KLI/UDP promise | Requires a separately designed and licensed real-time path such as Sunrise.FRI. |

KUKA identifies Sunrise.FRI as its optional real-time communication path and
describes it as supporting high communication rates. That is a different
controller/client architecture, not a reason to label ordinary Java/KLI UDP
as real-time. See KUKA's [Sunrise.FRI description](https://www.kuka.com/en-us/company/press/news/2022/02/kuka-sunrise%2C-d-%2Cos-med).

Calibration timing revision
`constant_latency_nearest_pose_motion_lomo_warn_fallback.v3` separates the
preferred timing band from the unusable-data boundary. Nearest robot poses up
to 150 ms away may be retained, with a visible warning above 20 ms. The
constant-offset search spans -300 to +300 ms; a supported offset beyond
±150 ms is also a warning, not an automatic rejection. Weak, ambiguous, or
unevaluable auto-offset statistics keep recorded timing at 0 ms and let the
robot-camera solvers continue. Missing/corrupt pose evidence and final
geometric validation failures remain blocking.

At the calibration application's maximum captured translation speed of
30 mm/s, an uncorrected 100 ms timing error corresponds to about 3 mm of
first-order translational displacement; 150 ms corresponds to about 4.5 mm.
That supports an advisory treatment for this deliberately slow motion, but it
does not prove the calibration is good: rotations, latency variation, and
projection geometry can still dominate. The existing held-out
translation/rotation, reprojection, outlier, and multi-camera closure gates
remain the acceptance authority.

The 50 Hz median / 25 ms p95 / 40 ms maximum cadence checks are therefore
commissioning targets, not calibration matching limits. Requesting 10 ms gives
the controller room to miss occasional best-effort cycles while keeping most
matches in the preferred warning-free band.

## Evidence from `test20260728_CalibStatic`

The retained `raw_robot_ee_poses.json` contains legacy packets without sender
sequence or controller monotonic timestamps. End-to-end host receive cadence
can still be measured. Excluding `_settled` samples and never joining intervals
across motion labels gives:

| Evidence | Value |
| --- | ---: |
| In-motion samples | 1,419 |
| Motion segments | 44 |
| In-motion intervals | 1,375 |
| Minimum gap | 60.345 ms |
| Median gap / rate | 61.042 ms / 16.382 Hz |
| Mean gap / rate | 61.487 ms / 16.264 Hz |
| p95 gap | 62.779 ms |
| p99 gap | 67.098 ms |
| Maximum gap | 79.253 ms |

The prior application asked for a 10 ms sleep but polled
`motion.isFinished()` before every pose query. The retained cadence is
consistent with a much slower operation in that hot path. Because the packets
are legacy and no controller profiling was retained, the evidence does not
prove which individual Sunrise call consumed the time. It does prove that the
old loop delivered roughly 16 Hz, not 100 Hz.

## Implemented Source Contract

The shared implementation consists of:

- `iiwa/PoseTestBot_PoseStreamTask.java`, an automatic-compatible
  `RoboticsAPICyclicBackgroundTask` initialized at 10 ms with
  `CycleBehavior.BestEffort`;
- `iiwa/PoseTestBotPoseStreamFunction.java`, the task-function interface used
  by the motion applications;
- `iiwa/PoseTestBot_Test.java` and
  `iiwa/PoseTestBot_CalibrationVarianceProposal.java`, which now execute
  blocking robot motions while independently starting and stopping the cyclic
  sampler.

There is no `moveAsync()`/`isFinished()` sampling loop left in either motion
application. The background task contains no motion command. It only resolves
the selected Application Data reference frame, queries the current flange
pose, constructs a packet, and sends UDP.

Every pose uses `robot_pose.v1` with run/frame identity and increasing
sequence. It also carries:

- `sender_target_period_ms`;
- `sender_previous_pose_delta_ns`; and
- `sender_pose_query_duration_ns`.

The Python receiver validates and retains those optional fields under
`source_packet`. Host receive/wall timestamps remain synchronization authority;
sender timing remains diagnostic because the clocks are not assumed aligned.
The task catches runtime exceptions instead of allowing an unhandled exception
to terminate it silently, stops that sampling segment, and exposes fatal and
send-failure counters to the motion application. A motion that produces zero
poses or a fatal sampler fault does not emit a successful end marker.

The four sources compile against a Sunrise 1.15.1 public API set. That is a
useful syntax/API check, not controller acceptance. The installed Workbench
Javadoc and exact project remain authoritative.

## Workbench and Physical Commissioning

Before enabling either motion application:

1. Add `PoseTestBot_PoseStreamTask` to the exact controller project through
   Workbench's background-task workflow and configure it for automatic start.
   Do not merely copy the Java class and assume the project metadata was
   created.
2. Include the task-function interface and exactly one provider for it.
3. Compile all four shared/application sources against the installed
   Sunrise.OS API. Record the exact project, controller, source revision, and
   compile result.
4. Keep both repository application gates false until the applicable frame,
   endpoint, swept-path, tool/load, cable, and T1 checks pass.
5. With explicit operator authorization and both physical execution gates,
   retain a supervised trial. Do not send UDP `STOP` during repeated
   calibration.
6. Run the cadence report and retain it:

   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv run python \
     scripts/report_robot_pose_cadence.py <run-root> --write
   ```

   The command exits nonzero when a commissioning target fails and writes only the derived
   `processed/robot_pose_cadence_report.json`; it does not modify raw poses.
7. Aim for all defaults: median rate at least 50 Hz, p95 gap no more than
   25 ms, and maximum gap no more than 40 ms. Treat a miss as a controller or
   delivery warning to investigate; do not fail a calibration attempt solely
   on this standalone cadence result.

When v1 sender evidence is present, compare `sender_cadence` with
`host_receive_cadence`. Slow sender cadence or long pose-query durations point
back to controller workload. Good sender cadence with poor host cadence or
sequence loss points to delivery/receiver scheduling. The relaxed 150 ms
matching boundary preserves slow-stream evidence, while the independent 20 ms
warning keeps the underlying cadence problem visible.
