# Guided Three-Camera Eye-in-Hand Calibration Validation — 2026-07-23

## Result

Three independent real calibration journeys completed through the guided web
workflow with the three connected eye-in-hand RealSense cameras and immutable
`calib00` target:

- target ID: `15b49f67-7cf5-4c00-9e7f-914aa6ed5da0`
- target geometry SHA-256:
  `3da681424ff77e55dc51c8c1c9bb58e0a425f7fa039b63d29c798aa2ad02b256`
- camera IDs: `033422071805`, `825412070181`, and `923322072633`
- robot profile: real iiwa at `172.31.1.147:30300`, with the receiver at
  `172.31.1.169:8080`

Each journey used a fresh run root, saved all three camera identities as
eye-in-hand, selected the run-owned hash-verified `calib00` bundle, passed the
guided readiness step, required both fresh capture acknowledgements, completed
physical acquisition, compared every supported PnP/hand-eye combination,
reviewed a complete common three-camera recommendation, and explicitly saved
the profiles. No iiwa `STOP` command was sent.

At campaign completion all three runs passed `rewrite_full_capture.v1` at
10/10 and `rewrite_calibration_validation.v1` at 3/3. Their status was 2/3
gates and 13/18 checks because the unrelated BOP-export gate was empty for
these calibration-only runs. Profile retirement later makes the reusable
calibration gate intentionally incomplete without altering this historical
evidence.

## Retained Guided Runs

| Run | Root | Attempt | Recommended bundle |
|---|---|---|---|
| 1 | `working_data/calib00_guided_real_20260723_run01` | `d909d13cc5944a068e8a2ec13eeedd32` | `IPPE + Tsai` |
| 2 | `working_data/calib00_guided_real_20260723_run02` | `3106a2b80b87444db0ac26de89bc01b3` | `IPPE + Park` |
| 3 | `working_data/calib00_guided_real_20260723_run03` | `f1e990d3424a48ed95b266f7bf134838` | `ITERATIVE + Shah` |

All three complete bundles pass the same per-camera and multi-camera gates.
However, the automatic recommendations selected different extrinsic methods,
so the cross-run comparison below combines acquisition variation with a solver
method switch; it is not a clean fixed-method repeatability experiment.

## Physical Acquisition

| Run | Capture time | Robot poses | Frames `033…` | Frames `825…` | Frames `923…` |
|---|---:|---:|---:|---:|---:|
| 1 | 145.959 s | 7,748 | 726 | 732 | 737 |
| 2 | 146.122 s | 7,756 | 776 | 780 | 785 |
| 3 | 147.396 s | 7,763 | 813 | 817 | 811 |

Every camera process returned code 0 in every run, and every robot pose
receiver succeeded. The capture reports record sustained three-frame camera
readiness before robot start and the policy that failure/cancellation cleanup
terminates only local process groups and never sends iiwa `STOP`.

The solver accepted 453/459/452 target-pose frames in run 1,
488/487/490 in run 2, and 513/511/509 in run 3. Each camera covered six or
seven of the nine image regions with no PnP data-preparation error.

## Recommended Bundle Quality

| Run | Camera | Inliers / observations | Mean reprojection | Held-out translation | Held-out rotation |
|---|---|---:|---:|---:|---:|
| 1 | `033422071805` | 453 / 453 | 1.221 px | 3.576 mm | 0.742° |
| 1 | `825412070181` | 459 / 459 | 1.052 px | 3.476 mm | 0.565° |
| 1 | `923322072633` | 448 / 452 | 1.101 px | 3.612 mm | 0.470° |
| 2 | `033422071805` | 487 / 488 | 1.211 px | 3.322 mm | 0.722° |
| 2 | `825412070181` | 487 / 487 | 1.040 px | 3.713 mm | 0.590° |
| 2 | `923322072633` | 490 / 490 | 1.116 px | 3.717 mm | 0.484° |
| 3 | `033422071805` | 513 / 513 | 1.215 px | 3.440 mm | 0.712° |
| 3 | `825412070181` | 511 / 511 | 1.049 px | 3.702 mm | 0.569° |
| 3 | `923322072633` | 509 / 509 | 1.114 px | 3.552 mm | 0.480° |

Maximum three-camera stationary-companion closure was:

| Run | Translation | Rotation | Gate |
|---|---:|---:|---|
| 1 | 4.678 mm | 1.115° | pass |
| 2 | 5.414 mm | 0.477° | pass |
| 3 | 7.847 mm | 0.470° | pass |

All are below the 10 mm / 5° multi-camera thresholds.

## Cross-Run Repeatability

The table records the maximum transform difference seen for each camera over
the three run pairs. These comparisons use the promoted `camera ->
robot_flange` translations and quaternion angular distance.

| Camera | Maximum translation difference | Maximum rotation difference |
|---|---:|---:|
| `033422071805` | 5.096 mm | 1.099° |
| `825412070181` | 3.411 mm | 0.747° |
| `923322072633` | 8.642 mm | 0.519° |

The 8.642 mm value is a historical diagnostic, not a passing cross-run result.
The promoted solver method changed between runs, so acquisition repeatability
and solver variation are confounded. Treat the difference as requiring a
controlled repeat, not as validated accuracy; the 10 mm within-run companion
gate is not a cross-run acceptance threshold.

## Retrospective Calibration Auto-Sync Replay

After the three physical journeys were complete, the new constant-offset
estimator was replayed offline against the retained raw frames, target-pose
observations, and robot-pose records for every camera/run combination. No
camera or robot was opened, no new physical capture occurred, and the original
0 ms attempts and promoted profiles were not rewritten.

The table reports the selected operator-facing offset and the relative
reduction in cross-validated mean stationary-companion translation residual
against the 0 ms baseline:

| Run | `033422071805` | `825412070181` | `923322072633` |
|---|---:|---:|---:|
| 1 | +70 ms / 24.3% | +85 ms / 38.0% | +50 ms / 16.7% |
| 2 | +75 ms / 26.3% | +85 ms / 37.7% | +45 ms / 15.8% |
| 3 | +75 ms / 28.3% | +80 ms / 35.8% | +55 ms / 14.0% |

All nine searches selected an applicable interior offset. Across runs, the
per-camera ranges were +70 to +75 ms for `033422071805`, +80 to +85 ms for
`825412070181`, and +45 to +55 ms for `923322072633`, with 14–38% closure
improvement. Fold-optimum spreads were 5–20 ms. Run 1 camera `923322072633`
retained a non-blocking 30 ms Li-method sensitivity warning.

The sign convention is:

```text
robot_pose_query_time = frame_time + robot_pose_time_offset_ms
sync_delta_ms = -robot_pose_time_offset_ms
```

Positive values therefore select a robot pose recorded later than the camera
frame. The implementation searches -150 to +150 ms on a 5 ms grid and uses
the nearest robot record without interpolation. These retained robot records
have approximately 11 ms median sample cadence, so the 5 ms grid is search
resolution, not demonstrated timing accuracy.

The search uses a fixed full-range observation set, fixed IPPE target poses,
Shah selection with Li sensitivity, and three fixed motion-disjoint folds.
Those folds are cross-validation tuning evidence: their validation residuals
participate in selecting the offset, so there is no untouched offset holdout.
The original retrospective table above used implementation revision
`constant_latency_nearest_pose_motion_cv.v1`, which required material
improvement in every one of the three fixed folds.

Revision `constant_latency_nearest_pose_motion_lomo_cv.v2` keeps the full
three-fold search, aggregate materiality, fold-optimum stability, method
sensitivity, boundary, and rotation gates, but replaces the arbitrary
every-fold materiality veto with a stronger motion-level audit. The v2 default
requires at least 12 eligible motions, four per fold. Every selected motion is
held out once while the robot-camera transform is refitted from all other
motions. Both Shah and Li must retain median translation improvement of at
least 0.25 mm and 10%. The exact one-sided positive-motion sign probability is
Bonferroni-corrected for every nonzero offset in the search because the
candidate is selected from that same fixed motion set; the corrected value
must remain at or below 0.05. Per-fold materiality remains visible as a warning
so an operator can see an uneven bucket without allowing that bucket
assignment to decide an otherwise consistent camera. Insufficient motion, flat
or ambiguous timing evidence, boundary optima, excessive fold/method
disagreement, inadequate aggregate or motion-level improvement, and rotation
degradation fail closed. A real attempt saves the complete curve, fold
membership, per-motion evidence, correction count, checks, sign convention,
and decision in `time_offset_search.json`.

This replay shows that robot motion is used and that a constant effective
latency improves stationary-target consistency. It does not prove physical
clock synchronization: RealSense `global_time` exposure timestamps are paired
with host-receipt timestamps for robot pose packets, not controller measurement
timestamps. It also does not use a known target transform. All three run
configs recorded `calib00` placement as `unknown`; the
`aruco_grid -> template_base` companion transform is estimated from PnP and
robot motion.

Finally, the replay does not by itself resolve the 8.642 mm cross-run
difference. That number compares the original promoted profiles, whose
automatic selections changed between Tsai, Park, and Shah, while this replay
only evaluates timing with a fixed reference. A new full solver comparison at
the accepted offsets, preferably with a controlled common method for the
repeatability comparison, is required before claiming that the cross-run
camera-to-flange difference improved.

## 2026-07-25 Three-Camera v2 Calculation Follow-up

A new immutable calculation attempt on retained run
`working_data/calib00_test20260724` exercised the complete v2 timing estimator
and full common-bundle solver without opening cameras or contacting the robot:

- attempt ID: `1c6b0c9d00dc49ce8d0c14c18d43336b`
- implementation:
  `constant_latency_nearest_pose_motion_lomo_cv.v2`
- result: complete, with recommendations for all three required cameras
- common recommendation: `IPPE + Shah`
- passing common bundles: 15

The complete timing-identification result was:

| Camera | Offset | Aggregate translation improvement | Shah LOMO | Li LOMO | Search-corrected sign p |
|---|---:|---:|---:|---:|---:|
| `033422071805` | +70 ms | 1.007 mm / 29.1% | 17/17 positive; median 0.811 mm / 28.6% | 17/17 positive; median 0.792 mm / 29.4% | 0.000458 |
| `825412070181` | +85 ms | 1.430 mm / 35.6% | 17/17 positive; median 0.530 mm / 27.8% | 17/17 positive; median 0.621 mm / 24.7% | 0.000458 |
| `923322072633` | +45 ms | 0.430 mm / 11.6% | 16/17 positive; median 0.555 mm / 16.5% | 16/17 positive; median 0.603 mm / 19.8% | 0.008240 |

The p-value limit is 0.05 after correcting for all 60 searched nonzero
offsets. Camera `923322072633` retains a visible fold-materiality warning
because one arbitrary three-fold bucket improved by 8.27%, below the 10%
threshold. It passes because the aggregate gate and the stronger
leave-one-motion-out audit pass independently for both reference methods; the
warning is not silently discarded.

The recommended bundle quality was:

| Camera | Accepted views | Solver inliers | Mean reprojection | Held-out translation | Held-out rotation | Supported x/y span | Supported hull area | Legacy cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `033422071805` | 828 | 85 / 85 | 1.122 px | 2.116 mm | 0.518° | 53.1% / 43.3% | 13.25% | 5/9 warning |
| `825412070181` | 823 | 85 / 85 | 1.021 px | 1.961 mm | 0.395° | 55.7% / 48.0% | 14.39% | 7/9 |
| `923322072633` | 830 | 90 / 90 | 1.090 px | 2.491 mm | 0.409° | 52.4% / 45.4% | 13.90% | 6/9 |

All cameras exceed the continuous extrinsic-coverage gates of 45% x span,
35% y span, and 10% normalized supported centroid-hull area, with five views
supporting each coordinate tail. This matters for `033422071805`: its dense,
wide coverage happens to occupy only five absolute 3 × 3 cells. Treating that
partition count as the extrinsic gate rejected sound evidence even though the
continuous coverage is similar to the other cameras and the prior successful
campaigns. The cell count remains visible as a warning and still gates a
manual OpenCV intrinsic fallback at 6/9; it no longer substitutes for actual
extrinsic field-of-view diversity.

The common bundle's maximum stationary-companion closure is 3.172 mm / 0.450°,
well inside the 10 mm / 5° limits. This closes the calculation failure for all
three cameras with stronger evidence rather than by dropping a camera or
lowering materiality thresholds.

An operator retry later exposed a runtime-lifecycle problem rather than a
timing-estimator regression. Attempt
`e588682f5ad64e9aaf8ed39e7b02c623` was created by a still-running backend
whose imported constants remained at
`constant_latency_nearest_pose_motion_cv.v1`. Its immutable request therefore
recorded the old three-motions-per-fold rule, and the fresh worker correctly
honoured that saved v1 contract instead of silently changing the intent. It
reproduced the old `923322072633` rejection.

Fresh-process attempt `268c897e1baf49e7bd78a434a4569b99` then independently
repeated the complete calculation from the same preserved recordings. It
recorded v2 with four motions per fold and reproduced the offsets, all
per-camera residuals, the 15 passing bundles, and the selected `IPPE + Shah`
closure above. Its recommended three-camera bundle was explicitly promoted:

| Camera | Published profile | Saved `sync_delta_ms` |
|---|---|---:|
| `033422071805` | `033422071805_eye_in_hand_IPPE_shah_268c897e` | -70 ms |
| `825412070181` | `825412070181_eye_in_hand_IPPE_shah_268c897e` | -85 ms |
| `923322072633` | `923322072633_eye_in_hand_IPPE_shah_268c897e` | -45 ms |

The canonical `calibration_profiles.json` is `calibration.v2`, all three
profiles are `valid`, and `rewrite_calibration_validation.v1` reports ready at
3/3. The setup API now exposes its loaded timing implementation revision. The
packaged workflow compares it with the revision it was built for, blocks Auto
attempt submission on a missing/mismatched revision, and tells the operator to
restart PoseTestBot and reload. Existing legacy attempts are labeled as
immutable and non-upgradable instead of being described as if they had run the
v2 leave-one-motion-out gate.

## Historical Reusable-Profile Retirement and Replacement

The six top-level `calibration_profiles.json` and
`intrinsic_calibration_profiles.json` collections from these runs were removed
after this review. They predate the required saved Auto time-alignment
provenance and must not be selected by a new object-dataset run. Raw captures,
robot poses, target evidence, candidates, solver/validation reports, and
immutable attempt artifacts remain unchanged. The fresh, explicitly promoted
replacement is attempt `268c897e1baf49e7bd78a434a4569b99` described above.

## Historical Run 3 Camera-to-Flange Transforms

These were Run 3's published transforms before profile retirement. They map
`camera -> robot_flange`; quaternions use WXYZ order.

| Camera | Translation mm `[x, y, z]` | Quaternion WXYZ |
|---|---|---|
| `033422071805` | `[-59.2845247044336, -32.21735360915485, 61.70260149584767]` | `[0.7108405169646154, -0.0062257503947197, 0.007418511543901948, -0.7032865455559164]` |
| `825412070181` | `[6.1259163204552385, -249.71900988898255, 72.3722173002862]` | `[0.6887283428092116, -0.10884388197649904, -0.13273715590903892, -0.7044055129034289]` |
| `923322072633` | `[5.057470909268587, 263.0575652489657, 66.14933394292434]` | `[0.7067956720845577, 0.11011281650636943, 0.07545293821750294, -0.6947099392393336]` |

Exact homogeneous matrices, companion transforms, quality, synchronization,
and historical promotion provenance remain in each run's immutable attempt
artifacts.

## Intrinsic Comparison

Every run used 45 training and 15 guarded held-out views per camera. All nine
manual OpenCV estimates passed their calculation checks, while the compatible
factory color projection remained selected under the
`compare_factory_opencv` comparison-only policy. Across the three runs,
factory/manual held-out RMS ranges were:

| Camera | Factory RMS range | Manual RMS range |
|---|---:|---:|
| `033422071805` | 1.313–1.462 px | 1.123–1.262 px |
| `825412070181` | 1.216–1.259 px | 0.999–1.012 px |
| `923322072633` | 1.251–1.283 px | 1.055–1.096 px |

Factory SDK depth scale and SDK depth-to-color alignment were not
recalibrated.

## Guided Workflow Findings

The real journeys exposed three workflow defects and one missing expectation:

1. Selecting a brand-new run path left the workflow on a permanent skeleton
   because a missing `run_config.json` was treated as pending setup. The page
   now renders Run Setup as soon as overview loading completes.
2. The calculation step retained the pre-capture empty camera list. Capture
   now invalidates calibration setup, and setup polls while no recorded cameras
   are available.
3. The capture dialog queued only `capture_execution`, bypassing the canonical
   sequence that persists the hardware snapshot, capture-plan preflight,
   execution plan, and rewrite gate. It now submits
   `real_full_capture_validation` with fresh ephemeral acknowledgements for
   every required real-capture step.
4. A healthy three-camera exhaustive comparison took 882–1,011 seconds.
   The calculation card now states the expected 10–20 minute duration and that
   the background job survives navigation. Run 2 also proved real persisted
   recovery: after the UI harness timed out, reopening the same run restored
   the completed attempt and allowed review/promotion without duplicate
   capture or calculation.

The original three execution plans contain the full pre-START
`capture_plan_preflight.v1` reports with `ok` status. The full-capture gate now
accepts that embedded immutable report when the standalone report is absent,
while rejecting mismatched embedded status. Future guided captures write the
standalone report through the canonical sequence.

Desktop Playwright journeys ran at 1920 × 1080. The safety dialog kept the
exact run, robot endpoint, supervision envelope, and both acknowledgements
visible; the result page kept all three camera comparisons and the atomic save
action reachable. The journeys reported no page or browser-console errors.

## Old-Workflow Cleanup

Before this campaign, the superseded run roots
`working_data/eye_in_hand_calib00_3cam_20260722_1130` and
`working_data/eye_in_hand_calib00_3cam_repeat_20260722_1202`, plus their stale
calibration job evidence, were permanently removed at the operator's request.
The old dated validation document was replaced by this record and all tracked
references were updated. The immutable global `calib00` target bundle remains
because it defines the physical printed grid and is not a calibration result.
