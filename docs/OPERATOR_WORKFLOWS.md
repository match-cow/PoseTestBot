# Operator Workflows

PoseTestBot presents two outcome-oriented workflows in the operator console:

1. **Calibrate cameras** to create a reviewed, reusable calibration.
2. **Record an object dataset** using a previously saved calibration.

The numbered steps are the normal operator path. Low-level pipeline stages are
available under **Advanced tools** for diagnosis and recovery, but they do not
replace the guided order and may produce incomplete evidence when run alone.
Workflow progress comes from run-owned artifacts, not browser state.

## Required and optional work

- **Required** steps form the numbered spine. A later step remains blocked when
  its required inputs or durable evidence are missing.
- **Optional** actions are shown separately and do not block the required
  outcome. Examples include creating a new grid when a suitable one already
  exists, editing catalogue data, inspecting alternate solver evidence, and
  rendering masks with BlenderProc.
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
   calibrate and confirm its serial/device identity, mounting mode, orientation,
   resolution, frame rate, and supervised robot speed. Saving setup does not
   open hardware.
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

## Journey 2: record an object-template dataset

Use this journey to acquire an object-bearing dataset after a compatible camera
calibration has been promoted.

1. **Configure cameras and select calibration — required.** Choose the enabled
   cameras and acquisition settings, then select a promoted calibration that
   covers every camera identity and mounting mode. PoseTestBot copies the exact
   selected `calibration_profiles.json` and
   `intrinsic_calibration_profiles.json` into
   `processed/calibration_inputs/<bundle_sha256>/`. The run-owned
   `calibration_profile_selection.json` binds the source bundle, copied-file
   hashes, and per-camera profile mapping. Later changes to the source run
   cannot silently alter this dataset run. Switching an existing selection
   requires an explicit confirmation and a matching current bundle hash. Once
   capture or derived dataset material exists, start a new run instead of
   rebinding that evidence to another calibration. The selected profiles must
   also contain a verified per-camera robot-pose time offset, timestamp pair,
   clock-domain/fallback rule, and maximum pose gap.
2. **Choose the object template and placement — required.** Select the immutable
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
   hardware by themselves.
5. **Synchronize and verify frames — required.** PoseTestBot applies each
   selected profile's saved timing automatically; manual values and generic
   defaults cannot override it. It writes derived frame-to-pose matches below
   `processed/synchronized/` and rejects missing matches, excessive pose gaps,
   incompatible timestamps, or calibration-provenance differences.
   Per-camera `sync_report.json` files and the run-level
   `sync_quality_report.json` retain the exact applied policy. Raw capture data
   remains untouched.
6. **Export the BOP dataset — required.** Revalidate the selected calibration
   and template identities, then write the BOP scenes, camera data, object
   poses, models, targets, frame map, and PoseTestBot provenance sidecars.

BlenderProc preparation and rendering of GT, masks, or derived COCO annotations
is explicitly optional. A synchronized calibrated recording remains valid
without those rendered products. Pose-estimator execution and metric evaluation
remain outside this repository.

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

The machine-readable contract mirrors the same compact five- and six-step
spines. Its `required`, `optional`, and `automatic` fields let other clients
preserve the distinction between operator decisions and queued computation
without exposing raw stage identifiers as the primary workflow.

Pipeline stages and reusable sequences remain the implementation layer beneath
these journeys. New operator-facing work should describe its purpose and
requirement in a journey first, while raw stage parameters remain advanced.
