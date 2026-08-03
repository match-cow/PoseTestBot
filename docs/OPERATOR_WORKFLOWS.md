# Operator Workflows

PoseTestBot presents two outcome-oriented workflows in the operator console:

1. **Calibrate cameras** to create a reviewed, reusable calibration.
2. **Record an object dataset** using a previously saved calibration.

The numbered steps are the normal operator path. Low-level pipeline stages are
available under **Advanced tools** for diagnosis and recovery, but they do not
replace the guided order and may produce incomplete evidence when run alone.
Workflow completion and status come from run-owned artifacts. Browser-local
state remembers only the selected run's last viewed workflow step for fast
return from supporting pages. An unsaved step draft remains intact while the
operator reviews another step in the same journey, but changing the active run
folder clears run-setup and placement drafts instead of carrying them into the
new run.

Camera names follow the same scope boundary. **Devices** stores reusable lab
defaults; step 1 snapshots an editable **Operator alias for this run** into
`run_config.json`. Existing runs always hydrate that saved value rather than a
newer lab default. Capture planning carries the label into `capture_plan.json`
and `dataset_manifest.json`, but sensor type and device ID remain the durable
physical identity.

The Devices mounting default follows the same rule. A newly configured run
copies each selected camera's explicit `static` or `eye_in_hand` value into its
own `capture.sensors[]` entry in `run_config.json`; an existing run keeps its
saved value even if the reusable Devices default changes later.

## Required and optional work

- **Required** steps form the required workflow spine. A later untouched step
  is shown as not started; **blocked** is reserved for a real failed check or
  missing prerequisite.
- **Optional** work is explicitly labeled and does not block the required
  outcome. Most optional actions are shown separately; dataset step 6 remains
  in the numbered rail so optional annotation evidence has a clear home.
  Other examples include creating a new grid when a suitable one already
  exists, editing catalogue data, and inspecting alternate solver evidence.
- **Automatic processing** means a queued local job performs the computation.
  It never means that PoseTestBot may open cameras or move the robot without a
  fresh operator action.
- Raw RGB-D frames and robot poses are never rewritten by synchronization,
  calibration, rectification, rendering, or export. Those steps create derived
  artifacts, normally below `processed/`.

## Journey 1: calibrate cameras

Use this journey when a camera, mounting mode, resolution, or workcell geometry
needs a new reusable calibration.

1. **Configure the run and cameras — required.** Choose every camera to
   calibrate and confirm its serial/device identity, run-owned operator alias,
   mounting mode, orientation, resolution, frame rate, and supervised robot
   speed. Saving setup does not open hardware.
2. **Choose the printed calibration grid — required.** Select the immutable
   target bundle that exactly matches the board in the cell. Its dictionary,
   marker dimensions, spacing, printable PDF, geometry hash, and placement are
   treated as one identity. Creating another grid is optional when a suitable
   saved grid already exists.
3. **Check readiness — required.** Run the single visible readiness check after
   setup and target selection. It validates saved configuration, target
   provenance, current sensor/runtime visibility, and the planned software
   stages without opening cameras or moving the robot. It does not reserve a
   device or replace the live identity, empty-output, and safety checks at
   capture startup.
4. **Record calibration images — required.** Place the board according to the
   selected calibration mode, clear the cell, and explicitly authorize the
   supervised camera-and-robot capture. Varied, sharp grid views and sufficient
   robot motion are required for a useful solution.
5. **Calculate, review, and publish — required.** Queue derived
   synchronization, target detection, intrinsic comparison, robot-camera
   solving, validation, and ranking. Review the recommended complete camera
   bundle, then explicitly publish only passing profiles. Calculation alone
   does not make a candidate reusable. The analysis inherits the hash-verified
   grid from step 2 and preselects the mounting interpretation saved in step 1;
   it cannot silently substitute another grid. Robot-mounted and static
   cameras are calculated as separate mounting groups, and the server rejects
   any camera/mode contradiction.

The result is a promoted `calibration_profiles.json` collection, its intrinsic
profiles, and retained attempt evidence. Per-view diagnostics and alternate
PnP/extrinsic candidates are optional review tools unless a quality check needs
investigation.

The two supported extrinsic interpretations are:

- **Eye in hand:** the camera moves with `robot_flange`, the grid is stationary
  relative to `template_base`, and the primary result is
  `camera -> robot_flange`.
- **Eye to hand:** the camera is static, the grid moves rigidly with
  `robot_flange`, and the primary result is `camera -> template_base`.

## Journey 2: record an object dataset

Use this journey to acquire an object-bearing dataset after a compatible camera
calibration has been promoted.

1. **Configure cameras and select calibration — required.** Choose the enabled
   cameras, their run-owned operator aliases, and acquisition settings, then
   assign a promoted calibration source to every camera identity and mounting
   mode. Static and robot-mounted cameras may use different source runs. When
   one source covers the complete setup, PoseTestBot retains that exact pair;
   when several sources are assigned, it deterministically combines only the
   selected camera/lens profile pairs. The resulting
   `calibration_profiles.json` and `intrinsic_calibration_profiles.json` live in
   `processed/calibration_inputs/<bundle_sha256>/`. The run-owned
   `calibration_profile_selection.json` binds the combined bundle and
   per-camera mapping. Its v2 form also records every source-run bundle hash
   and the cameras assigned to it. Later changes to any source run cannot
   silently alter this dataset run. Switching an existing selection requires
   an explicit confirmation and a matching current bundle hash. Once capture
   or derived dataset material exists, start a new run instead of rebinding
   that evidence to another calibration. The selected profiles must
   also contain a verified per-camera robot-pose time offset, timestamp pair,
   clock-domain/fallback rule, and maximum pose gap.
   Choose the capture synchronization policy in the same setup. The general
   choice is `timestamp_aligned`. The research combined-view choice is the
   exact `capture_synchronization.v1` contract
   `hardware_trigger` / `realsense_inter_cam_sync` / `depth_exposure`: it
   requires at least two enabled exact-ID D435 cameras, both `static` and
   `eye_in_hand` mounting modes, exactly one selected master, and subordinate
   roles for the rest. USB OAK-D Pro and USB ZED 2i cannot join that group.
   Saving an invalid combination fails; it never becomes timestamp alignment
   through a silent fallback. D435 RGB remains timestamp-associated and is not
   certified as a simultaneous cross-camera exposure.
   Before acquisition, physically qualify the exact harness, camera
   membership, mounts, roles, resolution, and FPS without robot motion, then
   record the operator-confirmed external exposure-timing evidence in
   `hardware_sync_qualification.json`. The recorder only copies evidence; it
   does not open cameras or contact the robot. Once capture status, logs, raw
   camera data, or raw robot-pose evidence exists, the qualification cannot be
   published or replaced. Start a new run when either the contract or
   qualification must change.
2. **Choose the pose template and placement — required.** Select the immutable
   printed pose-template version that is physically present, enter its measured
   pose in `template_base`, and confirm the placement. Creating or editing
   workpieces and publishing a new template are optional prerequisites only
   when the needed immutable template does not already exist.
3. **Check readiness — required.** Run the same single readiness facade. For an
   object dataset it additionally fails closed when the calibration snapshot is
   missing, unreadable, incompatible, or lacks a valid profile for an enabled
   camera, or when the template placement is unconfirmed. Live camera identity
   and output protection are checked again by the gated recording action.
4. **Record the object dataset — required.** Place the physical objects exactly
   as confirmed, clear the cell, and explicitly authorize supervised capture.
   The selected calibration and template are provenance; they do not authorize
   hardware by themselves. For a hardware-trigger run, capture also requires
   exact master/subordinate SDK configuration and read-back plus global depth
   timestamps before the robot may start. Every camera must then continue
   appending monotonic metadata. Its default liveness deadline is 12 planned
   frame periods clamped to 2–5 seconds, independent of the robot UDP
   first/inter-packet timeouts; a stalled or rewritten stream aborts capture
   while preserving raw evidence. Immediately before receiver startup, the
   supervisor revalidates the contract and qualification and records their
   exact hashes in the successful capture report.
5. **Process frames and create the base BOP export — required.** One
   recoverable background job applies each selected profile's saved timing;
   manual values and generic defaults cannot override it. It writes derived
   frame-to-pose matches below `processed/synchronized/`, verifies sync quality
   and calibration provenance, rectifies the RGB-D frames, and writes the base
   BOP scenes, camera data, models, targets, frame map, and PoseTestBot
   provenance sidecars. Per-camera `sync_report.json` files and the run-level
   `sync_quality_report.json` retain the exact applied timing policy. Raw
   capture data remains untouched.

   Hardware-trigger runs additionally associate global depth timestamps whose
   full earliest-to-latest group span is within the configured threshold and
   write only complete mixed-mount sets to
   `processed/synchronized/multiview_frame_groups.json`. Early master frames,
   incomplete groups, and unmatched frames remain preserved as raw evidence
   but are not authoritative combined observations. The export maps every
   authoritative complete group onto its per-camera BOP scene/image views in
   `bop/posetestbot_frame_sets.json` and carries forward the capture-report
   binding. The BOP rewrite gate compares it across the current qualification,
   capture report, authoritative groups, frame sets, frame map, and exported
   files. The job continues after navigation and remains available in
   **Jobs** for progress, logs, and cancellation.
6. **Add optional BOP ground-truth evidence — optional.** After the base
   image/model export is verified, optionally choose **Plain pose ground
   truth** or **Pose + object masks and ROI**. Both modes use BlenderProc 2.8.0
   to validate the immutable scene and derive standard model-to-camera pose
   rows. The complete mode also uses the pinned official BOP Toolkit renderer
   and captured depth to write `scene_gt_info.json`, full masks, visible masks,
   ROI, pixel counts, and visibility fractions. Ground-truth work is one
   recoverable CPU/render/disk job and remains visible in **Jobs** after
   navigation. The workflow marks this step complete only when the selected
   annotation output has verified durable evidence.

Annotation generation is explicitly optional. A synchronized calibrated base
export remains valid without GT or masks, while plain pose GT deliberately is
not BOP19 evaluation-ready. Pose-estimator execution and proprietary-result
conversion remain outside this repository. The sole metric exception is the
run-scoped **Inspect → BOP Evaluation** path: it accepts an already compatible
standard BOP19 CSV or a deterministic test-only slight GT perturbation, queues
the pinned official VSD/MSSD/MSPD scripts, and writes derived evidence only
below `processed/bop_evaluation/`. It is not a pipeline stage.

The real static and robot-mounted depth observations in an authoritative
complete group share the synchronized depth-exposure instant, including
depth-visible robot occlusion. Associated D435 RGB images are not
hardware-synchronized and must not be claimed to share a moving-robot or
changing-illumination instant. BlenderProc does not currently render the
articulated iiwa, so the full model mask must not be presented as robot
occlusion truth. In the complete annotation mode, the visible mask is compared
with captured depth and therefore reflects measured occluders, including the
robot where valid depth observed it. Missing or invalid captured depth still
requires explicit inspection.

## One readiness step, two fresh capture gates

The guided UI intentionally shows only one **Check readiness** step per journey.
It is a facade over the relevant run-level checks and writes durable
`run_preflight_report.json` evidence:

- **Not checked** means no report exists for the run.
- **Setup changed** means the saved report no longer matches `run_config.json`.
- **A required check failed** means at least one prerequisite did not pass.
- **Evidence unreadable** means the report cannot be validated and must be
  replaced.

Passing readiness is necessary but is not permission to capture. The final
recording action always requires two fresh acknowledgements in the capture
dialog:

1. the robot cell is clear, the physical target or object arrangement is
   correct, and supervised real-robot motion is authorized; and
2. the selected cameras may be opened and active previews may be stopped.

Those acknowledgements exist only in the capture request. They are not stored
in `run_config.json`, a reusable plan, or browser storage. Capture then repeats
time-sensitive plan, sensor, and empty-output checks before receiver bind or
robot `START`. Readiness and ordinary processing jobs never send those gates.

For a selected reusable calibration, readiness re-hashes both run-owned
profile snapshots, validates their pairwise camera/lens projection identity,
checks the saved camera mapping, and compares the selection record with
`run_config.json`. Rectification, BlenderProc preparation, and BOP export repeat
that verification immediately before consuming the profiles, so a passing old
readiness report cannot hide a later file or path change.

## Factory SDK versus OpenCV intrinsics

These labels describe two candidates for the camera's color projection; they
do not describe the robot-camera extrinsic transform.

- **Factory SDK intrinsics** are the camera matrix, distortion model, and
  coefficients reported by the camera SDK and recorded with capture. “Factory”
  does not mean PoseTestBot recalibrated depth scale or depth-to-color alignment.
- **OpenCV intrinsics** are a new color-camera model fitted from this run's
  printed-grid observations. They are an independently checked fallback and
  comparison, not an automatic replacement for the SDK values.
- **Existing intrinsics** are an exact compatible run-local profile reused by
  the reuse policy. “Existing” is not a third calibration algorithm.

Selection is deliberately conservative:

1. If the captured Factory SDK projection is usable as a forward OpenCV model,
   it remains selected. The manual OpenCV fit and the matrix/distortion deltas
   remain comparison evidence even when the manual fit has a lower RMS error.
   A lower RMS, including the recorded absolute and relative improvement, never
   replaces a compatible factory projection.
2. Forward-compatible SDK models are `none`, `brown_conrady`, and
   `modified_brown_conrady`. RealSense `inverse_brown_conrady` is accepted only
   when every coefficient is finite and exactly zero, because only then are the
   forward and inverse mappings the same pinhole projection. Nonzero inverse
   coefficients are never passed to OpenCV as forward distortion.
3. Only when the factory projection is unusable may the manual OpenCV result be
   activated. It must pass all training, held-out, and plausibility gates:
   at least 15 training views, coverage of at least 6 of 9 image cells, at least
   five held-out views, no more than 3 px error for any accepted training or
   held-out view, no more than 1.5 px RMS, finite parameters, positive focal
   lengths, a principal point inside the image, at most 10% focal-length change,
   principal-point movement of at most 5% of image width/height, at most 5%
   pixel-aspect change, and absolute `[k1, k2, p1, p2, k3]` distortion bounded
   by `[1, 3, 0.05, 0.05, 5]`.
4. If the factory projection is unusable and the OpenCV candidate does not pass
   every activation gate, calibration fails closed for that camera. It cannot
   be promoted as a valid reusable profile.

The complete decision, both candidates, held-out results, quality gates,
selection reason, and deltas are retained in attempt-level
`intrinsic_comparison.json`.

## Operator and implementation surfaces

- Guided pages: `/workflow/calibration` and `/workflow/dataset`.
- Expert recovery controls: `/workflow/advanced`.
- Machine-readable descriptions: `GET /pipeline/workflows`, schema
  `operator_workflows.v1`.
- Legacy workflow URLs redirect into the matching guided step; they are route
  compatibility aliases, not separate workflows.

The Dashboard derives its five-step calibration or six-step dataset overview
from the selected run's saved `dataset_mode`, links every tile to the exact
guided step, and uses durable run evidence for progress. It also keeps the room
monitor, acquisition-disk capacity, active jobs, and recent failures visible.
An unconfigured run is sent to the two-outcome chooser instead of being
presented as calibration.

The sidebar groups supporting pages by purpose. Devices and the reusable
Calibration Target, Workpiece Catalogue, and Pose Template libraries prepare
inputs; Cell, BOP Evaluation, and Jobs inspect evidence and background work.
Each page shows its workflow handoff because visiting or editing a reusable
library does not by itself mutate the selected run. The global **Operator
console guide** summarizes these scopes and the physical-execution boundary
without replacing the step-local prerequisites and safety text.

The machine-readable contract mirrors the same compact five- and six-step
spines. Its `required`, `optional`, and `automatic` fields let other clients
preserve the distinction between operator decisions and queued computation
without exposing raw stage identifiers as the primary workflow.

Pipeline stages and reusable sequences remain the implementation layer beneath
these journeys. New operator-facing work should describe its purpose and
requirement in a journey first, while raw stage parameters remain advanced.
