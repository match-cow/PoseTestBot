# IIWA Ordinary Full-Capture Application

## Status

`iiwa/PoseTestBot_Test.java` is the repository candidate for an ordinary
pose-template capture. It is not evidence of the application or revision
currently deployed on the Sunrise controller. The repository candidate is
deliberately disabled with `ENABLE_AFTER_OFFLINE_VALIDATION=false`.

Before enabling it, compile and simulate the source in the exact
Sunrise.Workbench controller project, create and verify the Application Data
frame below, and commission both A1 paths in T1 with the installed tool/load,
camera rig, fixtures, and cables. Repository validation does not perform any
robot motion.

## Frame Contract

The calibration target and the physical pose template do not have to share an
origin or axis convention. They must not be represented by one ambiguously
retaught frame.

| Use | Persistent Sunrise frame | Repository frame role |
| --- | --- | --- |
| Nine-frame calibration program | `/PoseTestBot/TemplateBase` | Calibration run `template_base` |
| Ordinary pose-template sweep | `/PoseTestBot/PoseTemplateBase` | Dataset run `template_base` |
| Calibration board geometry | Target bundle `aruco_grid` | Explicit `aruco_grid → template_base` placement |
| Selected pose-template geometry | Selection `pose_template` | Explicit `pose_template → template_base` placement |

Create `/PoseTestBot/PoseTemplateBase` as a persistent Application Data
`ObjectFrame`; do not construct a numeric replacement at runtime. Teach its
origin and axes against the physical pose-template datum. Record its
relationship to `/PoseTestBot/TemplateBase` as commissioning evidence.

The word `template_base` in run artifacts is a semantic role, not a Sunrise
path. For an ordinary dataset run, the Java stream maps that role to
`/PoseTestBot/PoseTemplateBase` and records the absolute path in every v1 pose
packet. If the selected digital pose template is physically aligned with that
frame, `template_base_from_pose_template` is identity. Otherwise, enter the
measured rigid transform during pose-template selection; never compensate by
silently retouching a calibration frame.

An eye-in-hand `camera → robot_flange` calibration remains independent of the
chosen world/reference frame as long as the camera mounting has not changed.
A static `camera → template_base` calibration does not: a profile expressed
against `/PoseTestBot/TemplateBase` must not be relabelled as if it targeted
`/PoseTestBot/PoseTemplateBase`. Measure and validate the frame transform and
produce a profile expressed in the dataset reference, or recalibrate. The
current BOP path must receive camera, robot-pose, and object transforms in one
consistent dataset `template_base`.

## Command and Motion Contract

Both accepted start shapes interpret their value as Cartesian metres per
second:

```json
{"start": 0.01}
```

```json
{
  "schema_version": "robot_command.v1",
  "command": "start_capture",
  "cartesian_velocity_m_s": 0.01,
  "receiver_ip": "172.31.1.169",
  "receiver_port": 8080,
  "run_id": "example-run"
}
```

The A1-only sweep is a circular flange path, while Sunrise PTP accepts a
relative joint-velocity setting. At the commissioned A1-minimum pose the
application therefore computes

```text
requested_A1_rad_s = requested_cartesian_mm_s / flange_orbit_radius_mm
joint_velocity_rel = requested_A1_rad_s / 98_deg_s
```

The radius is measured from the robot-root Z/A1 axis. KUKA publishes A1 rated
speeds of 98°/s for the LBR iiwa 7 R800 and 85°/s for the LBR iiwa 14 R820.
Using the larger value as the denominator makes the requested Cartesian value
an upper bound on either model. Before that conversion, the repository host
never transmits a numeric START value above 0.03 and the candidate application
independently caps the Cartesian input at 0.03 m/s. It also limits the computed
A1 angular velocity to 3°/s. The final relative joint velocity is therefore the
lower result of both caps.

The host cap is deliberate while the deployed application remains
unconfirmed: an older application that interprets the legacy numeric value
directly as relative joint velocity receives at most `0.03` (3%), while the
candidate interprets the same value as no more than 0.03 m/s. These ordinary
software limits are defense in depth, not safety-rated limits. Record the
exact installed model and verify the actual speed in Workbench/T1. The product
values are available in KUKA's official
[LBR iiwa 7 R800 data sheet](https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000246832_pl.pdf)
and
[LBR iiwa 14 R820 data sheet](https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000246833_en.pdf).

New run configurations default to 0.01 m/s and the workflow permits
0.01–0.03 m/s. The 720-second supervisor envelope accommodates the slowest
full A1 sweep while the independent first-packet and inter-packet timeouts
still detect a receiver that never starts or stops progressing.

Speed alone cannot guarantee blur-free images. Rolling-shutter skew and motion
blur also depend on exposure/readout time, illumination, optics, object
distance, and the camera's auto-exposure behavior. Verify sharpness in the
supervised trial and shorten exposure or improve lighting when needed. The
blocking 8%-relative PTP motions to the taught
`/PoseTestBot/CaptureStart` frame, the A1 sweep start, and the taught
`/PoseTestBot/CaptureEnd` frame occur outside pose streaming; their camera
frames are raw transition evidence, not authoritative synchronized capture
frames.

No motion occurs before an accepted start command. The application then:

1. moves PTP to the taught `/PoseTestBot/CaptureStart` frame;
2. snapshots that frame's non-A1 joint branch;
3. moves slowly to the commissioned A1 minimum;
4. waits 1.5 seconds;
5. performs the A1 sweep to the commissioned maximum while sending poses;
6. waits 1.5 seconds;
7. moves PTP to the taught `/PoseTestBot/CaptureEnd` frame; and
8. sends the terminal marker only after that final blocking PTP completes.

It remains at `/PoseTestBot/CaptureEnd` after the terminal marker. A later
return through `/PoseTestBot/CaptureStart` to A1 minimum happens only after a
new, independently authorized start command.

## UDP Pose Stream

The command-supplied receiver address and port take precedence. When
`receiver_ip` is omitted, the controller uses the lab fallback
`172.31.1.169`. An explicitly blank or wildcard receiver IP means “use the
start-command sender address.” Ports outside 1–65535 and non-positive or
non-finite velocities are rejected with controller-log errors.

Every new pose packet contains:

- `schema_version=robot_pose.v1`, packet kind, run ID, and increasing sequence;
- controller monotonic and wall-clock diagnostic timestamps;
- `robot_flange → template_base` endpoint semantics;
- `sunrise_reference_frame_path=/PoseTestBot/PoseTemplateBase`; and
- KUKA XYZ in millimetres plus A/B/C in radians.

The hardened Python receiver remains compatible with legacy packets. For v1
packets it validates an unchanging run/frame identity and increasing sequence,
retains sender metadata under `source_packet`, and records sequence gaps as
estimated UDP loss. Host receive/wall timestamps remain the synchronization
authority; controller timestamps are diagnostic because the two clocks are not
assumed synchronized.

Packet parsing, socket creation, pose sending, terminal retries, and thread
interruptions are logged instead of being silently swallowed. The terminal
packet is still sent three times because UDP delivery is not guaranteed.

## Stop and Safety Contract

The application reads the command socket only while idle. A legacy or
structured UDP stop request therefore cannot interrupt the active A1 motion.
It is not an emergency stop or other safety function; while idle it only exits
the application. Use the controller's approved safety response for an unsafe
condition.

## Commissioning Minimum

- Record the exact controller, Sunrise.OS/Workbench project, robot model,
  source revision, tool/load, camera rig, operator, reviewer, and date.
- Back up Application Data; create and teach
  `/PoseTestBot/PoseTemplateBase`, `/PoseTestBot/CaptureStart`, and
  `/PoseTestBot/CaptureEnd`; and read back their XYZABC values and parent-frame
  relationships.
- Confirm the selected run and pose-template placement use the ordinary frame,
  and reject any static calibration profile expressed in the wrong base.
- Compile the exact project and resolve `getRootFrame()`, PTP acceleration/jerk
  setters, JSON-simple, and all three persistent frames.
- Simulate the PTP path to `/PoseTestBot/CaptureStart`, the repositioning path
  to A1 −169°, the captured path to A1 +169°, and the final PTP path to
  `/PoseTestBot/CaptureEnd`, including every joint/redundancy branch, joint and
  singularity margin, collision clearance, and cable motion.
- With explicit operator authorization, single-step the complete
  PTP/A1/PTP sequence in T1 at reduced override before enabling the source.
- Verify the start-frame plus A1-minimum positioning finishes inside the
  receiver's first-packet timeout and the end-frame PTP finishes inside its
  inter-packet timeout.
- During the supervised trial, verify increasing v1 sequences, the recorded
  Sunrise frame path, requested/applied Cartesian and A1 speed logs, exposure
  settings, image sharpness, camera coverage, and a successful receiver
  terminal state.
