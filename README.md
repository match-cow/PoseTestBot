<picture>
  <source media="(prefers-color-scheme: dark)" srcset="posetestbot/web/static/cow_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="posetestbot/web/static/cow_light.png">
  <img src="posetestbot/web/static/cow_light.png" alt="PoseTestBot cow logo" width="96" align="right">
</picture>

# PoseTestBot

PoseTestBot is a local lab data-acquisition and BOP dataset export tool for
robot-mounted and static RGB-D sensors. Its finish line is a synchronized,
calibrated, inspectable BOP-format dataset with optional BlenderProc-generated
GT/masks.

This repository does not run pose-estimation runtimes, convert estimator output
to BOP result CSVs, run BOP Toolkit evaluation, or export metric reports. Those
downstream workflows should consume this repo's BOP output from a separate
project.

## What Is In Scope

- Fixed real lab iiwa profile with explicit execution safety gates.
- Capture planning, preflight, and supervised capture execution.
- RealSense, OAK-D Pro, and ZED 2i sensor registry/status/capture contracts.
- Run-scoped camera enable/disable controls that retain configured camera
  identity and calibration metadata while excluding disabled cameras from work.
- Non-destructive timestamp alignment and fail-closed mixed-mount RealSense
  depth-exposure hardware synchronization under `processed/synchronized/`.
- Sync quality reporting.
- Pinned PoseGridGen target preview/generation, immutable target bundles,
  legacy ArUcoGridGen import, split marker detection/pose solving,
  factory-vs-OpenCV RealSense color intrinsic comparison, explicit
  hand-eye/known-grid
  extrinsic solving, selection-gated promotion, and derived RGB-D rectification.
- Persistent JSON-backed Workpiece Catalogue with retained CAD assets, editable
  labels/tags/groups/attributes, 3D identification previews, lifecycle controls,
  revisioned metre/millimetre correction, and metadata import/export.
- Pinned PoseTemplateCreator stable grounded orientations, bounded isometric
  previews, exact footprint/layout validation, immutable printable pose
  templates sourced from filtered active catalogue workpieces, preview-rich run
  placement, and per-instance object GT.
- BlenderProc preparation/render planning for optional GT and masks.
- BOP dataset export, model metadata, targets, frame maps, authoritative
  hardware-synchronized frame sets, and optional multiview/COCO sidecars.
- React/shadcn operator console backed by the Flask API and local job runner.

## Operator Workflows

The web console starts with two guided outcomes: **Calibrate cameras** and
**Record an object dataset**. Each journey shows a numbered required path,
keeps optional authoring/rendering work off that path, and uses one visible
readiness step before a separately authorized physical capture. See
[`docs/OPERATOR_WORKFLOWS.md`](docs/OPERATOR_WORKFLOWS.md) for the complete
operator contract, including saved-calibration reuse and the exact Factory SDK
versus OpenCV intrinsic-selection policy.

## Quick Setup

Install the Python 3.12 dependencies and initialize the pinned target generator:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
```

Run Python entry points through `uv`:

```bash
uv run python scripts/robot_status.py --json
```

## Hardware Profile

Current lab expectations:

- 3 Intel RealSense D435-class cameras.
- 1 Luxonis OAK-D Pro.
- 1 Stereolabs ZED 2i.
- KUKA LBR iiwa at `172.31.1.147:30300`.
- Lab receiver IP on the robot subnet: `172.31.1.169`.
- Normal network IP on the same interface: `10.145.8.132`.

The sole robot profile is the real lab iiwa. Inspect it without sending UDP commands:

```bash
uv run python scripts/robot_status.py
uv run python scripts/robot_status.py --json
```

Check sensors and acquisition runtimes:

```bash
uv run python scripts/sensor_status.py --json
uv run python scripts/sensor_adapters.py --json
uv run python scripts/runtime_status.py --json
```

`runtime_status.py` checks acquisition-relevant external tools only:
BlenderProc for optional GT rendering and the ZED SDK Python module.

## Run Configs

Create an operator intent artifact:

```bash
uv run python scripts/create_run_config.py working_data/example_run
```

Defaults:

- real lab robot profile,
- current lab sensor list,
- objectless dataset mode until an immutable pose template is selected,
- `real_full_capture_validation` as the saved sequence,
- `plan_only=true`.

New run configs also make the robot stream frames explicit:
`robot_flange -> template_base` with `kuka_abc_radians`; optional fixed edges
can describe flange-to-TCP or template-base-to-physical-base transforms. Older
real-profile configs without frame metadata remain readable and receive a
`legacy_frames_inferred` warning; fake-profile configs are rejected.
New runs are objectless by default. Add and classify test objects on the
**Workpiece Catalogue** page, then use active workpieces on the **Pose
Templates** page or through the pose-template CLIs to create and select an
immutable bundle. Object-bearing runs resolve its physical instances and retain
stable catalog UUID and BOP `obj_id` provenance.
For the managed pose-template workflow, create the run with
`--dataset-mode pose_template`, then select and confirm an immutable template
in **Workflow → Object dataset**, step 2. See
[`docs/WORKPIECE_CATALOGUE.md`](docs/WORKPIECE_CATALOGUE.md) and
[`docs/POSETEMPLATECREATOR_OBJECT_GT.md`](docs/POSETEMPLATECREATOR_OBJECT_GT.md).

Step 1 of either guided workflow has a **Use for this recording** checkbox for
each configured camera. Disabling a camera retains
its identity, alias, mounting/orientation metadata, and calibration-profile
selection in `run_config.json`; it excludes that camera from capture planning
and preflight, calibration, rewrite-gate expectations, and the Cell scene. At
least one camera must remain enabled. Regenerate any already-written capture
plan and preflight after changing this selection so their evidence matches the
current run configuration.

`run_config.v3` also makes the acquisition timing claim explicit below
`capture.synchronization`. The default is
`capture_synchronization.v1` / `timestamp_aligned`. The current lab's supported
combined mode is deliberately narrower:

- `mode=hardware_trigger`
- `implementation=realsense_inter_cam_sync`
- `scope=depth_exposure`
- at least two exact-ID D435 cameras, including both a `static` and an
  `eye_in_hand` view
- exactly one selected master; every other enabled camera is a subordinate

For example:

```bash
uv run python scripts/create_run_config.py working_data/combined_run \
  --sensor realsense_d435:STATIC_SERIAL:static \
  --sensor realsense_d435:ROBOT_SERIAL:eye_in_hand \
  --hardware-trigger \
  --hardware-sync-group-id combined-d435-01 \
  --hardware-sync-master-sensor realsense_d435:STATIC_SERIAL
```

The command validates the identities and mounting mix before writing the run.
USB OAK-D Pro and USB ZED 2i cameras cannot join this hardware-trigger group,
and PoseTestBot does not silently downgrade an invalid request to timestamp
alignment. D435 inter-camera signaling synchronizes depth exposures only; the
associated rolling-shutter RGB images are retained, but are explicitly not
certified as simultaneous RGB exposures. The configured skew threshold is the
full earliest-to-latest depth timestamp span across every camera in a complete
group, not a separate allowance on either side of the master.

After the operator performs the physical harness and exposure-timing test
without robot motion, record its external evidence against that exact run:

```bash
uv run python scripts/record_hardware_sync_qualification.py \
  working_data/combined_run \
  --operator OPERATOR_ID \
  --method pulsed_light \
  --observed-max-depth-timestamp-skew-ms 0.8 \
  --evidence path/to/pulse-trace.csv \
  --confirm-passed
```

This command only copies evidence; it does not open cameras or contact the
robot. It writes `hardware_sync_qualification.json` plus hash-verified,
run-owned evidence. Hardware-trigger preflight, synchronization, and BOP export
fail closed if the evidence is missing, modified, or stale for the configured
resolution, FPS, policy, camera IDs, mounts, or roles.
Record or replace this qualification before acquisition begins. Once capture
status, report, logs, raw camera data, or raw robot-pose evidence exists, the
qualification is immutable and the recorder refuses to publish another one;
create a new run for a different qualification or hardware contract.

For example, a measured flange-to-TCP edge can be recorded at creation time:

```bash
uv run python scripts/create_run_config.py working_data/example_run \
  --fixed-transform-json '{"from":"robot_flange","to":"tcp","rotation_quaternion_wxyz":[1,0,0,0],"translation_mm":[0,0,125]}'
```

Preview the default physical validation workflow without executing it:

```bash
uv run python scripts/create_run_config.py working_data/real_run \
  --sequence real_full_capture_validation \
  --print-sequence-plan
```

Write and queue preflight-aware sequence plans:

```bash
uv run python scripts/run_preflight.py working_data/example_run --write
uv run python scripts/run_pipeline_sequence.py working_data/example_run \
  --sequence sync_to_bop_dry_run --plan-only
```

## Capture

Plan capture startup commands without opening hardware:

```bash
uv run python scripts/run_capture_plan_stage.py working_data/example_run
uv run python scripts/run_capture_plan_preflight.py working_data/example_run
uv run python scripts/run_capture_execution_plan.py working_data/example_run
```

Physical execution is explicitly gated and must be operator-triggered:

```bash
uv run python scripts/run_capture_execution_plan.py working_data/real_run \
  --allow-cameras --allow-real-robot --include-sensors
uv run python scripts/run_capture_execution_stage.py working_data/real_run \
  --allow-cameras --allow-real-robot --include-sensors
```

The supervised receiver gets those acknowledgements only in its runtime
command; they are never stored in `capture_plan.json` or a plan-only sequence.
Its default first-packet and inter-packet timeouts are 120 and 60 seconds, and
the supervisor default is 300 seconds. Before receiver bind or `START`, every
selected camera has 15 seconds to publish at least three valid, committed
`frame_metadata.jsonl` records. Override these bounds with `--startup-wait`,
`--receive-start-timeout-s`, `--receive-idle-timeout-s`, and `--timeout-s` when
the reviewed motion program requires different bounds.
Camera-stream freshness is deliberately independent of the robot packet
timeout: by default the supervisor allows 12 planned frame periods without new
metadata, clamped to 2–5 seconds. The optional
`--camera-metadata-idle-timeout-s` override is validated and cannot exceed five
seconds, so a shorter robot run cannot hide an alive-but-stalled camera.
Direct `start_iiwa.py` and `scripts/pose_receiver_udp_json.py` invocations also
require both fresh acknowledgement flags. Prefer the supervised capture stage
for coordinated camera startup and cleanup. The receiver refuses to bind or
send START when `raw_robot_ee_poses.json` already exists and preserves failed
or canceled streams as unique `raw_robot_ee_poses.partial.*.json` evidence.

## Synchronization And Quality

Synchronize without mutating raw capture folders:

```bash
uv run python scripts/sync_run_non_destructive.py working_data/example_run
uv run python scripts/run_sync_quality.py working_data/example_run
```

Derived sync folders are written below:

```text
processed/synchronized/<sensor>/
```

Synchronization is transactional and emits `sync_report.v3` per sensor plus
`sync_quality_report.v2` at run level. Synchronized metadata names the derived
frame paths, retains source-frame provenance, and records any timestamp-source
fallback. Raw frames and raw robot poses are never overwritten; start a new run
root when capture preflight reports existing raw data.

For a validated mixed-mount hardware-trigger run, capture starts the designated
D435 master before its subordinates and verifies the configured/read-back
inter-camera roles. The supervisor also requires every live camera to keep
appending monotonic frame metadata while cameras overlap and while the robot
receiver runs; a stalled or rewritten stream aborts local capture and preserves
the raw evidence. Immediately before receiver startup it revalidates the exact
run contract and physical qualification. A successful full-capture report
records the configuration SHA-256, the exact qualification-artifact SHA-256,
and that immediate revalidation; authoritative hardware-sync grouping rejects
any other capture status, missing execution gate, or changed binding.

Synchronization associates frames by global depth-sensor
timestamps whose full earliest-to-latest group span is within the configured
threshold and writes only complete cross-camera
sets to:

```text
processed/synchronized/multiview_frame_groups.json
```

Early master frames, incomplete sets, and unmatched frames remain in the raw
recording for audit and recovery; they are not presented as authoritative
combined observations. Complete groups certify a common depth-exposure instant
across static and robot-mounted views, including depth-visible robot occlusion.
Their associated D435 RGB images are not hardware-synchronized and must not be
described as sharing the same moving-robot or changing-illumination state.
Each durable group artifact is rebuilt and exactly compared with the current
run config, qualification, synchronized metadata, matched robot poses, and
referenced RGB-D bytes before it is accepted downstream.
The exact capture-report binding is carried from this artifact into
`bop/posetestbot_frame_sets.json`. The BOP rewrite gate compares the current
qualification, capture report, authoritative groups, frame sets, frame map, and
exported bytes; it fails if the original capture-time configuration or
qualification provenance no longer agrees.

For a dataset run with a selected hash-bound calibration, these commands
automatically apply each camera profile's saved sync delta, timestamp pair,
required timestamp domain, fallback policy, and nearest-pose threshold.
Per-camera reports bind the applied values to the calibration bundle/profile,
the quality report checks exact coverage, and rectification/BOP export reject
missing or contradictory timing evidence. Manual timing options cannot override
a selected calibration policy.

## Calibration

The preferred operator path is **Workflow → Camera calibration**. One form
selects exactly one geometry, one or more captured cameras, and one of two
authoritative modes:

- **Robot-mounted camera (eye-in-hand):** the target is stationary relative to
  `template_base`; the primary result is `camera → robot_flange`.
- **Static camera (eye-to-hand):** the target is attached to `robot_flange`;
  the primary result is `camera → template_base`.

**Run calibration** queues one CPU/disk parent job. It never opens cameras or
commands the robot. Only selected captured camera folders are synchronized into
an immutable derived attempt under
`processed/calibration/<attempt_id>/`. The five displayed phases are Prepare
data, Estimate target poses, Estimate time alignment, Compare robot-camera
solutions, and Validate and rank. Every PnP/extrinsic combination and failure
remains available for review.

The default **Auto compare — recommended** policy compares IPPE, ITERATIVE, and
SQPNP using a common robust point mask and LM refinement, then compares Tsai,
Park, Horaud, Andreff, Daniilidis, Shah, and Li robot-camera solutions with
deterministic robust-closure outlier rejection and leave-one-pose-out
validation. Each attempt also compares the captured RealSense factory color
projection with a manual OpenCV intrinsic fit. Both candidates and their
matrix/distortion deltas remain in `intrinsic_comparison.json`; the captured
factory profile is retained whenever its projection is OpenCV-compatible, and
the manual fit remains comparison evidence. RealSense
`inverse_brown_conrady` is forward-OpenCV-compatible only when every recorded
distortion coefficient is finite and exactly zero, where both directions
reduce to the same pinhole projection. Nonzero inverse coefficients are never
passed to OpenCV as forward distortion. When the factory projection is
unusable, the manual fit activates only after its 15-view, 6/9 coverage, 3 px
per-view, 1.5 px RMS, five-view held-out, and parameter-plausibility gates pass;
factory and manual evidence are retained either way.

RealSense calibration pairs color-frame `global_time` exposure timestamps with
robot-pose host-wall timestamps, forbids fallback, and excludes frames whose
nearest pose is more than 20 ms away. Guided attempts default to **Auto time
alignment**; use the explicit 0 ms baseline only after independently validating
that capture path.

Auto time alignment estimates one constant effective latency per camera:

```text
robot_pose_query_time = frame_time + robot_pose_time_offset_ms
sync_delta_ms = -robot_pose_time_offset_ms
```

A positive operator-facing offset uses a later robot record. The bounded
-150–150 ms search uses measured robot motion and stationary-target closure,
does not require a known target-to-base transform, and never rewrites raw
timestamps. Motion-disjoint consistency, improvement, boundary, ambiguity, and
rotation guards fail closed; `time_offset_search.json` retains the complete
decision. This estimates capture-path latency, not hardware-clock
synchronization.

Published profiles retain the accepted per-camera offset, timestamp fields,
clock-domain rule, fallback policy, and pose-gap limit. Object-dataset runs copy
those profiles into a hash-bound snapshot and apply the timing automatically;
manual timing cannot override it, and synchronization quality, rectification,
and BOP export recheck the provenance. See
[`docs/OPERATOR_WORKFLOWS.md`](docs/OPERATOR_WORKFLOWS.md) for the operator
contract and the dated validation record below for search details and limits.

Target poses require at least 12 common corner inliers, 50% support, four
markers with at least three supported corners each spanning two target rows and
columns, and at most 3 px whole-board mean reprojection error. Across accepted
views, at least 50% of target markers and 60% of target rows and columns must be
covered (`calib00`: 18/35 markers, 3/5 rows, 5/7 columns). Final passing also
requires at least 15 accepted views, 6/9 image-centroid cells, four distinct
motion poses, at least 20 mm translation span and 5° rotation span, rotations
of at least 2° with rotation-axis second/first singular ratio at least 0.15,
at least six hand-eye inliers, at most 10 mm mean translation residual, at most
5° mean rotation residual, at most 25% motion-balanced outliers, and at most
25% outliers within any repeated motion. Raw outlier density remains evidence,
not a promotion gate. Solver fitting uses at most five evenly spaced frames per
motion, then validates every accepted frame with motion-balanced
residual/outlier evidence. For two or more cameras estimating the same
stationary companion frame, ranking evaluates only complete bundles that use
the same PnP and extrinsic method for every camera. Every individual candidate
must pass and the maximum pairwise companion closure must be at most 10 mm and
5°. The best mean individual score establishes a quality baseline; bundles
within 0.01 normalized score are treated as quality-equivalent and ranked by
their normalized pairwise companion closure. Bundles outside that band remain
ordered by individual quality, so a poor per-camera solution cannot win on
closure alone. Deterministic numeric and method tie-breaks finish the ordering.
Normalized ranking values are rounded to six decimals before comparison; this
already corresponds to only 0.00001 mm for a translation-only difference or
0.000005° for a rotation-only difference. Smaller solver dust therefore falls
through to the canonical PnP/extrinsic method order.
If no common bundle passes, every recommendation and promotion fails closed;
an explicit override must still select one complete recorded passing bundle.
Single-camera ranking is unchanged. Recommendations remain inactive until the
operator accepts them.

The 2026-07-23 guided campaign produced three independent real capture and
calibration attempts for all three eye-in-hand RealSense cameras. Their
historical 0 ms results passed the then-current run gates and reached 7.847 mm /
1.115° maximum within-run companion closure. The 8.642 mm / 1.099° maximum
cross-run transform difference is a diagnostic requiring a controlled repeat,
not validated accuracy: the selected solver method changed between runs.
Exact transforms, residuals, workflow findings, and immutable attempt evidence
remain in
[`docs/EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md`](docs/EYE_IN_HAND_CALIBRATION_VALIDATION_20260723.md).

An offline retrospective replay of the new constant-offset search over all
nine saved camera/run combinations selected +70 to +75 ms for `033422071805`,
+80 to +85 ms for `825412070181`, and +45 to +55 ms for `923322072633`.
Cross-validated stationary-target translation closure improved by 14–38%
relative to 0 ms. This was saved-data analysis, not another capture, and its
5 ms search grid is finer than the approximately 11 ms median robot-pose
cadence. It neither proves hardware-clock accuracy nor resolves the
method-confounded cross-run difference.

The three top-level reusable profile collections were retired after this replay
because they predate the required saved time-alignment provenance. Raw
captures, robot poses, and immutable attempt evidence remain. A new calibration
must pass Auto time alignment and be explicitly published before an
object-dataset run.

The compatible factory color projections were retained under the
comparison-only policy while all 45-training/15-held-out OpenCV fits remain
immutable evidence. Across the three runs, factory/manual held-out RMS stayed
within 1.216–1.462 / 0.999–1.262 px. Depth scale and depth-to-color alignment
remain factory SDK provenance, not a depth recalibration. A saved-data
depth-plane diagnostic found a range-dependent metric-depth anomaly on
`923322072633`; metric depth from that camera remains unvalidated pending a
later cable/firmware and depth-specific check.

The **Calibration Targets** page previews and fits PoseGridGen ArUco boards and
stores immutable bundles below
`working_data/calibration_targets/<target_id>/`. Saved targets remain browsable
and selectable when PoseGridGen generation is unavailable. Generation never
selects a target automatically.

Placement choices are unknown, identity-aligned to `template_base`, or the
PoseGridGen board-to-base pose converted to PoseTestBot's millimetre/WXYZ frame
contract. Once target-dependent calibration or BOP artifacts exist, changing
the target or placement is blocked and a new run is required. See
[`docs/POSEGRIDGEN_CALIBRATION_TARGETS.md`](docs/POSEGRIDGEN_CALIBRATION_TARGETS.md)
for bundle, provenance, API, and recovery details.

The calibration import stage prefers that run-config selection. For an older
run it still accepts an exact legacy ArUcoGridGen 1.0 JSON or a PoseGridGen 2.0
source manifest:

```bash
uv run python scripts/run_calibration_target_import.py working_data/example_run \
  --source working_data/example_run/aruco_grid_config.json \
  --aligned-to-template-base
uv run python scripts/run_aruco_detection_stage.py working_data/example_run
uv run python scripts/run_intrinsic_calibration_stage.py working_data/example_run \
  --mode calibrate
uv run python scripts/run_aruco_pose_stage.py working_data/example_run
uv run python scripts/run_calibration_observations.py working_data/example_run \
  --target-spec calibration_target.json
```

The stage-level commands remain available as advanced diagnostics and retain
their existing behavior:

```bash
uv run python scripts/run_calibration_solver.py working_data/example_run \
  --mode compare
uv run python scripts/run_calibration_validation.py working_data/example_run \
  --select-profile realsense_123=PROFILE_ID
```

The intent-level promotion transaction mirrors accepted evidence into the
canonical calibration artifacts, merges exactly one valid `calibration.v2`
profile for each accepted camera, preserves unrelated profiles, and records
attempt, operator, target, PnP, and extrinsic provenance. The diagnostic CLI
promotion path remains explicit as well:

```bash
uv run python scripts/run_calibration_validation.py working_data/example_run \
  --select-profile realsense_123=PROFILE_ID --promote
uv run python scripts/run_camera_rectification.py working_data/example_run
```

Rectification writes only below `processed/rectified/<sensor>/`, using linear
RGB and nearest-neighbor aligned-depth remapping. BlenderProc preparation and
BOP export prefer that tree when present. `run_aruco_stage.py` remains the
factory-intrinsics compatibility wrapper.

## Workpiece Catalogue

The operator console's **Workpiece Catalogue** page is the persistent source
of test-object identity. Upload PLY, STL, or OBJ CAD plus an optional PNG
texture; the local job runner inspects the mesh and retains the original file,
a canonical PLY, hashes, and texture below
`working_data/object_catalog/objects/<catalog_uuid>/`. Operators can edit the
workpiece name, alias, description, tags, groups, and custom scalar key/value
attributes, search or filter the catalogue, and use bounded isometric card
thumbnails plus one orbitable exact 3D detail view for identification. The
selected detail loads the hash-revisioned canonical PLY, preserves its authored
orientation, and only centres/uniformly scales it for display. Cards use a
separately bounded, geometry-hash-bound
recognition mesh that keeps the welded source when it fits and otherwise
compares quadric and spatial candidates by component/Euler topology before
using a convex safety proxy. The cache records which strategy won and whether
topology was retained, so an unavoidable approximation is labelled rather than
silently presented as exact. Ranked orientations and exact contours remain in
the full derived analysis used by the template editor.

The catalogue manifest remains portable JSON at
`working_data/object_catalog/object_catalog.json`. Mutations are serialized
across web and worker processes, each committed state receives an atomic
numbered revision, and stable UUID/BOP `obj_id` values are never reused.
Archiving is reversible. Permanent deletion additionally requires an archived
record, explicit confirmation, and no references from any pose-template
bundle; an unreadable bundle also blocks deletion. Bundle publication and
catalogue deletion share one cross-process lock, and the tombstone manifest is
committed before unreferenced assets are removed. Tombstones retain asset
cleanup status and bounded error evidence, so a repeated confirmed delete can
retry an interrupted cleanup without reviving the retired identity.

JSON export/import is deliberately metadata-only. Exported JSON records asset
references and hashes but does not embed CAD, canonical PLY, or texture bytes;
import updates matching locally installed workpieces and reports absent local
assets as skipped. Export remains available for metadata recovery when an asset
is corrupt, while import skips the affected entry and continues with intact
records. Pose Templates selects only active workpieces from this same catalogue,
then immutable bundles and run selections preserve complete snapshots. New
bundle manifests keep card metadata bounded and place exact contours in the
hash-verified preview instead of duplicating them in every instance record.
Cards use the bounded template thumbnail; a selected version loads the exact
interactive preview. Selection is a strictly validated, journaled transaction
that recovers the prior run state after an interrupted promotion. See
[`docs/WORKPIECE_CATALOGUE.md`](docs/WORKPIECE_CATALOGUE.md).

## BOP Dataset Export

Prepare optional BlenderProc inputs and render plan:

```bash
uv run python scripts/run_blenderproc_prepare_stage.py working_data/example_run \
  --calibration-profiles working_data/example_run/calibration_profiles.json
uv run python scripts/run_blenderproc_render_stage.py working_data/example_run --dry-run
```

Export BOP dataset structure:

```bash
uv run python scripts/run_bop_export_stage.py working_data/example_run \
  --calibration-profiles working_data/example_run/calibration_profiles.json
```

The transactional export emits `bop_export_manifest.v4` and uses standard
BOP-scenewise paths:

```text
bop/
├── dataset_info.json
├── posetestbot_bop_frame_map.json
├── posetestbot_frame_sets.json          # hardware-trigger runs
├── posetestbot_pose_template.json       # pose-template runs
├── posetestbot_instance_map.json        # pose-template runs
├── models/
└── <split>/<scene_id>/
    ├── rgb/
    ├── depth/
    ├── scene_camera.json
    ├── scene_gt.json
    ├── scene_gt_info.json
    ├── mask/          # optional
    └── mask_visib/    # optional
```

The preparation and export CLIs accept the same repeatable `--object-name` /
`--objectless` choice. Objectless rendering writes a successful skipped plan
without invoking BlenderProc. Objectless BOP output retains RGB, depth, and
camera metadata, writes empty GT and targets, and contains no models or masks.

The export preserves:

- scene RGB/depth,
- `scene_camera.json`,
- explicit empty or imported `scene_gt*.json`,
- optional `mask/` and `mask_visib/`,
- `bop_export_manifest.json`,
- `posetestbot_bop_frame_map.json`,
- `posetestbot_frame_sets.json` for validated hardware-trigger runs,
- `models/obj_XXXXXX.ply`,
- `models/models_info.json`,
- `test_targets_bop19.json`.

Pose-template runs load every physical instance independently in BlenderProc
2.8.0. Duplicate instances share one stable numeric `obj_id` and one exported
model, while their masks and `scene_gt` rows retain distinct immutable instance
UUIDs in PoseTestBot sidecars.

`posetestbot_frame_sets.json` maps each authoritative complete hardware group
onto its per-camera BOP `scene_id`/`im_id` views and records that depth exposure
is hardware-synchronized while RGB exposure is not certified. Only the depth
observations may be treated as sharing the robot/scene instant; the associated
RGB images may differ during motion or changing light. BlenderProc's optional
synthetic GT/masks do not currently render the articulated iiwa, so they must
not be treated as robot-occluder visibility truth. The sidecar also retains the
successful capture report's exact configuration/qualification binding for
rewrite-gate verification.

## Pipeline Sequences

Useful presets:

- `real_full_capture_validation`
- `sync_aruco`
- `sync_aruco_calibration_observations`
- `sync_aruco_calibration_candidates`
- `sync_aruco_calibration_solver`
- `sync_aruco_calibration_validation`
- `aruco_grid_full_calibration`
- `calibrated_capture_to_bop_dataset_dry_run`
- `sync_to_bop_dry_run`
- `sync_to_bop_calibrated_dry_run`
- `capture_to_bop_dataset_dry_run`

List current sequences and stages through the Flask API:

```bash
curl http://127.0.0.1:5000/pipeline/stages
curl http://127.0.0.1:5000/pipeline/sequences
```

## Web UI

Start the Flask-backed operator console:

```bash
uv run posetestbot-web
# source-checkout compatibility entrypoint:
uv run python web_interface.py
```

The web server intentionally defaults to unauthenticated `0.0.0.0` for the
trusted lab LAN and still exposes deliberate real-robot controls. Do not expose
it to an untrusted network. Web run paths are confined to `working_data` by
default; add path-list entries with `POSETESTBOT_WEB_RUN_ROOTS`. Read-only
external inputs default to `scripts/default_data`; extend them with
`POSETESTBOT_WEB_INPUT_ROOTS`. Symlink escapes and output paths outside
the selected run are rejected. Installed deployments may set
`POSETESTBOT_APP_ROOT`; otherwise a source checkout is detected automatically
and an installed command uses its current working directory. CLI tools continue
to accept explicit paths.

The bundled console has desktop routes for Dashboard, Devices, Cell,
Calibration Targets, Workpiece Catalogue, Pose Templates, Workflow, and Jobs.
The catalogue entry appears directly below Calibration Targets and above Pose
Templates. The read-only Cell page renders the HRI template,
base/flange/TCP proxies, calibrated cameras, pose-template instances,
calibration target, and exact recorded trajectories in right-handed Z-up
millimetres.
Missing frame edges remain visibly unresolved, and a component/provenance list
remains available without WebGL. For each resolved camera that list includes
the exact calibration profile identity, camera-to-parent 4 × 4 matrix, WXYZ
quaternion, millimetre translation, stationary-target companion transform when
estimated, quality, solver, common-bundle promotion, target, and intrinsic and
synchronization provenance. It remembers the selected run,
system/light/dark theme, and manual IIWA target in the browser. The Devices
page distinguishes **Capture-ready**, **Not capture-ready**, and
**Disconnected** cameras and shows a human-readable readiness reason. It
blocks preview, snapshot, and new run selection for cameras that are not ready,
while still allowing a previously selected unavailable camera to be
deselected.

Physical capture remains separate from
ordinary stage forms: current preflight evidence is required, camera previews
are stopped first, and two fresh acknowledgements send `allow_cameras` and
`allow_real_robot` together in that one request. A non-plan-only capture
sequence is rejected by `/pipeline/run-config`; use Advanced Capture instead.
Manual **Start IIWA** controls also require a fresh target confirmation plus
both execution acknowledgements. **Stop IIWA** remains available without
motion-start gates, but it is not a safety stop and cannot interrupt active
motion. In the operator-reported running calibration program it exits the
waiting program and requires a manual application restart, so do not use it
between calibration captures.

The checked-in `posetestbot/web/static/ui/` build is what Flask and installed
wheels serve, so Bun is not a runtime dependency. Frontend development uses the
Bun-locked Vite project:

```bash
cd frontend
bun install --frozen-lockfile
bun run typecheck
bun run lint
bun run build
```

The production build clears the previous output before writing hashed assets.
Use `bash scripts/install.sh --with-web-build` to perform the locked install and
build through the project installer.

Important endpoints:

- `GET /ui/bootstrap`
- `GET /ui/runs`
- `GET /ui/cell-scene`
- `GET /ui/cell-scene/timeline`
- `GET /robot/status`
- `POST /run-command`
- `GET /sensors/status`
- `GET /runtime/status`
- `GET|POST /monitoring/webcam`
- `POST /monitoring/webcam/<job_id>/brightness/autocalibrate`
- `GET /calibration-targets/status`
- `GET /calibration-targets/capabilities`
- `POST /calibration-targets/fit`
- `POST /calibration-targets/preview`
- `GET /calibration-targets/bundles`
- `DELETE /calibration-targets/bundles/<target_id>`
- `POST /calibration-targets/bundles/<target_id>/select`
- `GET /calibration/setup`
- `GET|POST /calibration/attempts`
- `GET /calibration/attempts/<attempt_id>`
- `POST /calibration/attempts/<attempt_id>/promote`
- `GET /workpieces/status`
- `GET /workpieces/catalog`
- `GET /workpieces/catalog/<catalog_uuid>`
- `POST /workpieces/catalog/upload`
- `PATCH /workpieces/catalog/<catalog_uuid>`
- `POST /workpieces/catalog/<catalog_uuid>/unit-corrections`
- `POST /workpieces/catalog/<catalog_uuid>/archive`
- `POST /workpieces/catalog/<catalog_uuid>/restore`
- `DELETE /workpieces/catalog/<catalog_uuid>`
- `GET /workpieces/catalog/<catalog_uuid>/assets/<kind>`
- `GET /workpieces/catalog/export`
- `POST /workpieces/catalog/import`
- `GET /pose-templates/status`
- `GET /pose-templates/catalog`
- `POST /pose-templates/catalog/upload`
- `GET|POST /pose-templates/workpieces/<catalog_uuid>/orientations`
- `GET /pose-templates/workpieces/<catalog_uuid>/orientation-thumbnail`
- `POST /pose-templates/preview`
- `POST /pose-templates/validate`
- `POST /pose-templates/generate`
- `GET /pose-templates/library`
- `GET /pose-templates/library/<template_uuid>`
- `GET /pose-templates/library/<template_uuid>/preview`
- `GET /pose-templates/library/<template_uuid>/thumbnail`
- `GET /pose-templates/library/<template_uuid>/assets/<instance_uuid>/<kind>`
- `GET /pose-templates/library/<template_uuid>/download/<kind>`
- `GET|POST /pose-templates/runs/selection`
- `POST /hardware/status`
- `GET|POST /run-config`
- `GET|POST /pipeline/preflight`
- `POST /pipeline/run-config`
- `GET /pipeline/recommendations`
- `GET /capture/jobs`
- `GET /capture/status`
- `POST /capture/jobs/<job_id>/stop`

The older `/pose-templates/catalog...` read/upload/archive/restore/asset APIs
remain available for compatibility. New catalogue management uses
`/workpieces`; pose-template preview, generation, library, and run-selection
APIs remain under `/pose-templates`.

## Rewrite Gates

Current gates:

- `rewrite_full_capture.v1`
- `rewrite_calibration_validation.v1`
- `rewrite_bop_export_readiness.v1`

Run all gate status summaries:

```bash
uv run python scripts/run_rewrite_status.py working_data/example_run --write
```

## Removed Legacy Entry Points

The destructive multi-camera sync/capture wrappers, `realsense_multi.py`, the
duplicate `main.py` web launcher, legacy `aruco_pose_estimation.py`, shell
BlenderProc wrapper/preparation script, ROI generators, and superseded
transform scripts were removed. Use the supported replacements:

- capture: `run_capture_plan_stage.py`, `run_capture_plan_preflight.py`, and
  `run_capture_execution_stage.py`;
- sync: `sync_run_non_destructive.py` followed by `run_sync_quality.py`;
- BlenderProc: `run_blenderproc_prepare_stage.py` and
  `run_blenderproc_render_stage.py`;
- calibration transforms: the calibration observation, solver, validation, and
  explicit promotion stages.

## Validation

Recommended local validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
git diff --check
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m playwright \
  tests/test_web_console_playwright.py tests/test_web_preview_playwright.py
```

The default test selection excludes the explicitly marked Playwright modules;
install Chromium and use `-m playwright` when running browser coverage.

The single source of truth for unfinished rewrite work is
[`docs/REWRITE_REMAINING_WORK.md`](docs/REWRITE_REMAINING_WORK.md).
