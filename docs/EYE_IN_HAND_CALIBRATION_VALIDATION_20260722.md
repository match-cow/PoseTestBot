# Three-Camera Eye-in-Hand Calibration Validation — 2026-07-22

## Result

Eye-in-hand calibration of all three RealSense cameras completed successfully
using the immutable `calib00` target:

- target ID: `15b49f67-7cf5-4c00-9e7f-914aa6ed5da0`
- target geometry SHA-256:
  `3da681424ff77e55dc51c8c1c9bb58e0a425f7fa039b63d29c798aa2ad02b256`
- promoted bundle: `IPPE + Shah`
- promoted attempt: `12e6a40eff444b889870597b787bf016`
- promoted run:
  `working_data/eye_in_hand_calib00_3cam_repeat_20260722_1202`

All three profiles are valid and promoted. The full-capture gate is 10/10
ready and the calibration-validation gate is 3/3 ready. The BOP-export gate
remains intentionally blocked because this calibration-only run did not
produce a BOP dataset.

No iiwa `STOP` command was sent.

## Physical Runs

The first exhaustive run is preserved at:

`working_data/eye_in_hand_calib00_3cam_20260722_1130`

It captured 9,792 robot poses and 943, 919, and 893 frames respectively. The
exhaustive PnP/extrinsic search recommended `IPPE + Shah`, with maximum
three-camera companion closure of 9.903 mm / 0.338°. Because the translation
result was close to the 10 mm gate, it was not promoted without an independent
repeat.

The confirmation run is preserved at:

`working_data/eye_in_hand_calib00_3cam_repeat_20260722_1202`

It captured 9,772 robot poses and 963, 936, and 908 frames respectively. Every
camera became ready on its first startup attempt. The repeat reduced maximum
companion closure to 7.104 mm / 0.421° and supplied the promoted bundle.

First-to-repeat transform differences were:

| Camera | Translation | Rotation |
|---|---:|---:|
| `033422071805` | 4.060 mm | 0.184° |
| `825412070181` | 5.400 mm | 0.365° |
| `923322072633` | 1.780 mm | 0.106° |

The two cameras present in the earlier independent baseline also agreed with
the repeat within 1.008 mm / 0.129° and 0.612 mm / 0.094°.

## Promoted Camera-to-Flange Transforms

The transforms below map `camera -> robot_flange`. Quaternions use WXYZ order.

| Camera | Translation mm `[x, y, z]` | Quaternion WXYZ |
|---|---|---|
| `033422071805` | `[-59.14419588674232, -30.63511921047722, 63.13461552457343]` | `[0.7114568666802874, -0.0060777227108807395, 0.0074622729398983526, -0.702663861759439]` |
| `825412070181` | `[5.367930532985498, -249.6917403687268, 74.43316306660485]` | `[0.6880400434536754, -0.1081668900981757, -0.13326580932222082, -0.7050822977188174]` |
| `923322072633` | `[7.779365815509917, 261.9700428429465, 68.29049076661184]` | `[0.7043450571581864, 0.10992362906402096, 0.0746867903362383, -0.6973067614611932]` |

The canonical profiles, including exact homogeneous matrices and companion
grid-to-template-base transforms, are in the promoted run's
`calibration_profiles.json`.

## Candidate Quality

| Camera | Observations / inliers | Mean reprojection | Held-out translation | Held-out rotation |
|---|---:|---:|---:|---:|
| `033422071805` | 606 / 605 | 1.183 px | 3.052 mm | 0.628° |
| `825412070181` | 608 / 608 | 1.040 px | 3.241 mm | 0.473° |
| `923322072633` | 610 / 610 | 1.095 px | 3.226 mm | 0.425° |

Maximum pairwise companion residuals were 7.104 mm and 0.421°, below the
10 mm / 5° promotion thresholds.

## Intrinsic Comparison

Manual OpenCV color-intrinsic calibration used 45 training and 15 held-out
views per camera.

| Camera | Factory held-out RMS | Manual held-out RMS | Selected |
|---|---:|---:|---|
| `033422071805` | 1.260 px | 1.019 px | compatible factory |
| `825412070181` | 1.230 px | 0.964 px | compatible factory |
| `923322072633` | 1.268 px | 0.998 px | compatible factory |

The manual solutions passed quality and plausibility checks. Compatible
factory color profiles were retained intentionally under the
`compare_factory_opencv` comparison-only policy, while all manual profiles and
comparison evidence remain preserved. Factory SDK depth scale and SDK
depth-to-color alignment were not recalibrated.

## Multi-Camera Startup Reliability

Capture execution now starts cameras deterministically one at a time. Each
camera must write at least three valid `frame_metadata.jsonl` records before
the next camera starts. Startup allows three attempts by default, but retries
only when the failed attempt produced no sensor output; any partial raw output
fails closed and is preserved.

The robot pose receiver and iiwa `START` are delayed until every camera passes
a final readiness check. Failure and cancellation cleanup terminate local
child process groups only and never send iiwa `STOP`.

## Cell View and Software Validation

The Cell scene API returned all three promoted cameras parented to
`robot_flange`, with valid calibration state, exact transform matrices,
companion transforms, and no warnings. The focused WebGL and canvas Cell
Playwright regressions passed.

Capture startup, retry, partial-output protection, cancellation-race, pipeline
option, and sequence regressions passed. The complete repository suite passed
632 tests: 603 core tests plus 29 localhost Playwright tests.
