# PoseTestBot Agent Notes

These notes are for Codex and other coding agents working in this repository.
They complement the system overview in
`docs/SYSTEM_OVERVIEW_REWRITE_BASELINE.md`.

## Operating Rules

- Use `uv` for Python environment and package management.
- Run scripts as `uv run python ...`.
- Add dependencies with `uv add ...`; do not hand-edit dependency locks unless
  a tool-generated update is impossible.
- The current Flask UI uses `posetestbot.jobs.runner.LocalJobRunner`; do not add
  new blocking `subprocess.check_output` request handlers. Declare resources on
  long-running or hardware-touching jobs so the runner can reject unsafe
  concurrent work.
- Keep the rewrite fake-iiwa-first unless the user explicitly asks to target the
  physical robot.
- Preserve raw capture data. New synchronization/export work should create
  derived artifacts rather than destructively renaming or deleting the only copy
  of frames.
- Keep progress current in `docs/REWRITE_PROGRESS.md`.

## Current Lab Hardware

- 3 Intel RealSense cameras.
- 1 Luxonis OAK-D Pro.
- 1 Stereolabs ZED 2i.
- KUKA LBR iiwa at `172.31.1.147:30300`.
- Lab receiver IP on the robot subnet: `172.31.1.169`.
- Normal network IP on the same interface: `10.145.8.132`.

The default robot profile is fake:

```bash
uv run python iiwa/fake_iiwa_controller.py --receiver-ip 127.0.0.1
uv run python scripts/pose_receiver_udp_json.py /tmp/posetestbot_fake_run --test
```

Use the real robot only intentionally:

```bash
POSETESTBOT_ROBOT_MODE=real uv run python scripts/pose_receiver_udp_json.py working_data/test_run
```

Inspect the selected fake/real profile without sending UDP commands:

```bash
uv run python scripts/robot_status.py
uv run python scripts/robot_status.py --json
```

The Flask transition API exposes the same read-only profile contract at
`GET /robot/status`, and the index page has a manual Robot refresh panel. Keep
robot status checks read-only; do not probe by sending start/stop commands.

Check connected camera status with:

```bash
uv run python scripts/sensor_adapters.py
uv run python scripts/sensor_adapters.py --json
uv run python scripts/sensor_status.py
uv run python scripts/sensor_status.py --json
```

`posetestbot.sensors.registry` is the static single source of truth for
supported RGB-D sensor families, display names, SDK module names, capture
scripts, folder prefixes, supported resolutions, and mounting modes. It does
not open hardware. `capture_plan.py`, `sensor_status.py`, the
`scripts/sensor_adapters.py` CLI, and Flask `GET /sensors/adapters` should all
flow through this registry. Update it first when adding or renaming a sensor
adapter.

The default expected counts are the current lab profile: 3 RealSense D435-class
cameras, 1 OAK-D Pro, and 1 ZED 2i.
The Flask transition API exposes the same status contract at
`GET /sensors/status`, the static adapter registry at `GET /sensors/adapters`,
and the index page has manual Sensors controls for both. Keep live status
checks side-effect-light and JSON-friendly; discovery errors should be reported
in the payload rather than crashing the page.

Run-scoped hardware snapshots are handled by
`posetestbot.pipeline.hardware_status`. Invoke
`uv run python scripts/run_hardware_status_stage.py <run>` or
`POST /hardware/status` to write `hardware_status_report.json`, record a
manifest `hardware_status` stage, and preserve the selected robot profile,
sensor visibility, and external runtime readiness before capture launch. The
stage is read-only and must not start robot motion or camera capture.

Check external runtime readiness with:

```bash
uv run python scripts/runtime_status.py
uv run python scripts/runtime_status.py --json
```

The Flask transition API exposes the same runtime contract at
`GET /runtime/status`, and the index page has a manual Runtimes refresh panel.
Runtime checks should remain lightweight: check executables, modules, env vars,
and checkout paths, but do not start Docker containers, BlenderProc render jobs,
or BOP Toolkit evaluation as part of status refresh. MegaPose and SAM6D runtime
checks currently validate wrapper-script availability through `MEGAPOSE_WRAPPER`
and `SAM6D_WRAPPER`, falling back to `scripts/megapose_wrapper.py` and
`scripts/sam6d_wrapper.py`.

## Rewrite Direction

The target architecture is a local lab webapp with a typed Python backend,
hardware adapters, a job runner, manifest-backed storage, BOP scenewise export,
and BOP Toolkit evaluation. Current script compatibility is useful, but avoid
adding new long-lived behavior only as prompt-driven scripts.

Near-term implementation should favor:

- `posetestbot.*` importable modules for contracts, configuration, artifact
  names, robot UDP helpers, sensors, synchronization, calibration, BOP export,
  jobs, and evaluation.
- Thin CLI wrappers around importable modules.
- JSONL or JSON sidecars that are backward-compatible with the existing folder
  layout.
- Explicit fake adapters and fixtures for hardware-free validation.
- Non-destructive derived outputs under `processed/` before replacing legacy
  destructive stages.

## Important Existing Contracts

- Robot start legacy command: `{"start": 0.2}`.
- Robot stop legacy command: `{"stop": true}`.
- Robot pose fields: `motion`, `X`, `Y`, `Z`, `A`, `B`, `C`.
- Raw robot pose artifact: `raw_robot_ee_poses.json`.
- Matched robot pose artifact: `match_robot_ee_poses.json`.
- Frame timestamp sidecar: `frame_metadata.jsonl`.
- Run manifest artifact: `dataset_manifest.json`.
- Run configuration artifact: `run_config.json`.
- Run preflight artifact: `run_preflight_report.json`.
- Derived sync report: `sync_report.json`.
- BlenderProc render plan artifact: `blenderproc_render_plan.json`.
- FoundationPose plan artifact: `foundationpose_plan.json`.
- MegaPose/SAM6D adapter plan artifacts: `megapose_plan.json`,
  `sam6d_plan.json`.
- BOP export manifest artifact: `bop_export_manifest.json`.
- BOP frame map artifact: `posetestbot_bop_frame_map.json`.
- Calibration profile schema: `calibration.v1`.
- Legacy RGB/depth folders: `rgb/`, `depth/`.

`posetestbot.sensors.frame_writer` owns the shared capture output contract:
legacy `rgb/` and `depth/` PNG files plus compact `frame_metadata.jsonl`
records. RealSense, OAK-D Pro, and ZED 2i capture scripts should write frames
through `write_legacy_rgbd_frame` or `write_aligned_rgbd_frame` so timestamp
field names, paths, and sensor types stay consistent for non-destructive sync.
Use `write_legacy_camera_sidecars` for `cam_K.txt`, `depthscale.txt`,
`camera.json`, and `camera_data.json`; these sidecars are still the bridge to
current estimator wrappers and BOP export.

Calibration profiles live in `posetestbot.calibration.profiles`. Eye-in-hand
profiles must use extrinsics from `camera` to `end_effector`; static profiles
must use extrinsics from `camera` to `robot_base` or `cell_world`. The legacy
`camera_ee_transform.json` is a migration source, not the long-term schema.
`run_blenderproc_prepare_stage.py --calibration-profiles ...` resolves
matching profiles for synchronized sensor folders and writes the derived
BlenderProc camera transform map under `processed/calibration/`. Eye-in-hand
profiles chain camera-to-EE through robot poses; static profiles repeat the
camera-to-robot-base or camera-to-cell-world transform for every synchronized
frame.
`run_bop_export_stage.py --calibration-profiles ...` records matching
calibration profiles in `bop_export_manifest.json` and per-frame
`scene_camera.json` metadata. When supplied, profile intrinsics and depth scale
are the BOP camera source instead of legacy `cam_K.txt`/`depthscale.txt`.
`posetestbot.calibration.preflight` owns the run-level
`calibration_preflight_report.json` gate. It loads the run config, resolves the
configured profile collection or `<run>/calibration_profiles.json`, matches
profiles to enabled sensors, and reports profile status, observation count, and
mean reprojection quality. Use
`uv run python scripts/run_calibration_preflight.py <run>` or
`POST /calibration/preflight` before stages that depend on calibrated
extrinsics. Warnings mean the data may still be structurally usable, but robust
calibration quality is not proven.
If `blenderproc/output/scene_gt*.json` or sensor-level `masks/` are present,
the BOP export imports them into the scene; otherwise it writes explicit empty
GT placeholders.
If `scene_gt.json` is present but `scene_gt_info.json` is absent, BOP export
derives basic bbox, pixel-count, and visibility metadata from `masks/` and
optional `blenderproc/output/mask_visib/` images.
By default BOP export reads `object_models/objects.json`, copies `.ply` models
to `bop/models/obj_XXXXXX.ply`, writes `models_info.json`, normalizes string
`obj_id` values to numeric BOP IDs, and writes `test_targets_bop19.json`.
`models_info.json` includes geometry metadata such as diameter/min/size when
vertices can be loaded from the PLY. Use `--no-model-export` only for structural
tests where models/targets are not desired.

`scripts/run_bop_result_export_stage.py` converts estimator outputs into BOP19
result CSVs under `results/bop/`, writes `bop_result_export_manifest.json`, and
records a `bop_result_export` stage in `dataset_manifest.json`. It reads
`bop_export_manifest.json` for sensor-to-scene IDs and BOP model metadata for
object IDs. FoundationPose is the default source and converts
`foundationpose*_output/ob_in_cam` matrices, scaling translations from meters to
millimeters by default to match the legacy evaluators. `--source aruco`
converts synchronized `aruco_pose_estimation.json` OpenCV `rvec`/`tvec` entries
with `--aruco-object-name` mapped to BOP model metadata; ArUco translation
values are left unchanged by default unless `--translation-scale-to-mm` is set.
`--source megapose` converts `megapose*_output/megapose_poses.json` entries,
assuming MegaPose translations are meters by default. `--source sam6d` converts
`sam6d*_output/detections_pem/*.json`, uses the highest-scoring detection per
frame, and leaves translations unchanged by default. Explicit output folders can
be repeated with `--foundationpose-output`, `--aruco-pose-file`,
`--megapose-output`, or `--sam6d-output`.

`scripts/run_bop_evaluation_stage.py` is the current BOP Toolkit bridge. It
validates a BOP19 pose-result CSV, writes `bop_evaluation_plan.json` and
`bop_evaluation_report.json`, sets `BOP_PATH` in the planned command
environment, and records a `bop_evaluation` stage in `dataset_manifest.json`.
The report captures prerequisite checks, command/env, result metadata, any files
discovered under the evaluation output folder, and numeric metrics harvested
from `scores*.json` files such as `scores_bop19.json`. Use `--dry-run` unless a
BOP Toolkit checkout and matching Python runtime are intentionally configured.
For the default `<run>/bop` export, result filenames must use dataset name
`bop`, such as `foundationpose_bop-test.csv`, because BOP Toolkit parses the
dataset name from the result filename.

`scripts/run_metric_report_export_stage.py` exports discovered legacy metric
artifacts and BOP Toolkit score rows into `results/metrics/metric_report.json`,
`results/metrics/metric_methods.csv`, and
`results/metrics/metric_report.xlsx`, then records a `metric_report_export`
stage in `dataset_manifest.json`. The exporter reuses
`metric_dashboard_summary` for parsing `accuracy_HRC-Hub.json`,
`accuracy_ArUco_HRC-Hub.json`, `all_results.json`, and
`bop_evaluation_report.json` score summaries; keep it as a reporting bridge
rather than a second metric evaluator.

`web_interface.py` now queues button actions through
`posetestbot.jobs.runner.LocalJobRunner`. Jobs write `job.json` and `log.txt`
under `working_data/jobs/<job_id>/`; the runner reloads saved job records on
startup, marks interrupted queued/running jobs as failed, and rejects new jobs
whose declared resources overlap a queued/running job. Job records include a
`parameters` snapshot for command configuration, and cancellation terminates the
process group on POSIX systems so child processes are not left running. The
Flask API exposes `/jobs`, `/jobs/<job_id>`, `/jobs/<job_id>/log`, and
`/jobs/<job_id>/cancel`; it also exposes `/robot/status` for the selected
fake/real iiwa profile, `/sensors/status` for the shared RealSense/OAK-D Pro/ZED
2i status snapshot, `/runtime/status` for external runtime readiness,
`/hardware/status` for run-scoped status snapshots, and
`/pipeline/recommendations` for read-only next-step suggestions based on the
current run artifacts. The transition index page has compact robot, sensor,
runtime/hardware snapshot, recommended-steps, and job/resource panels wired to
those endpoints.
This is a
bridge toward the baseline
FastAPI/Jinja2/HTMX job architecture, not the final web stack.

`posetestbot.pipeline.recommendations` owns the first artifact-driven next-step
contract. It inspects run artifacts such as `run_config.json`,
`run_preflight_report.json`, `capture_plan.json`, `sync_quality_report.json`,
calibration reports, BOP manifests, and BOP result manifests, then returns
suggested `uv run ...` commands, endpoints, expected artifacts, and resource
hints. It recommends writing run preflight reports when the snapshot is missing,
failed, or stale relative to `run_config.json`, and only recommends queueing the
saved sequence once that snapshot is fresh, using
`posetestbot.pipeline.preflight.run_preflight_queue_summary` as the shared
queue-readiness source of truth. Keep this helper read-only; do not make
recommendation lookup launch hardware or mutate run folders.

`posetestbot.pipeline.stages` defines the current typed pipeline stage registry
for the transition web API. Use it instead of constructing shell strings in web
handlers. It builds command arrays such as `uv run python
scripts/run_bop_export_stage.py <run_root> ...`, validates known options, stores
the normalized options in the job `parameters` snapshot, and declares resources
before submission. Flask exposes `/pipeline/stages`, `/pipeline/stages/<stage>`,
and `/pipeline/run`. Expensive external stages should keep safe dry-run defaults
in this registry unless the user explicitly requests execution.

`scripts/run_foundationpose_stage.py` is the manifest-tracked FoundationPose
bridge. It validates synchronized sensor folders with prepared
`blenderproc/objects.json` files, writes `foundationpose_plan.json`, records a
`foundationpose` stage, and in dry-run mode does not start Docker. Non-dry-run
execution launches the legacy wrapper as `uv run python
scripts/foundationpose_wrapper_multi.py ...`. Keep the pipeline registry default
for this stage dry-run-first unless the user explicitly asks to run the
FoundationPose runtime.

`scripts/run_megapose_stage.py` and `scripts/run_sam6d_stage.py` are
manifest-tracked dry-run-first adapter scaffolds for the legacy MegaPose/SAM6D
paths. They validate synchronized sensor folders, write `megapose_plan.json` or
`sam6d_plan.json`, record `megapose`/`sam6d` manifest stages, and plan
`uv run python <wrapper> ...` commands. Non-dry-run execution is intentionally
gated on the configured wrapper script existing; the current repository does not
contain `scripts/megapose_wrapper.py` or `scripts/sam6d_wrapper.py`.
The artifact browser treats `foundationpose_plan.json`, `megapose_plan.json`,
and `sam6d_plan.json` as known run artifacts and summarizes estimator ID,
dry-run state, object ID, sensor names, wrapper availability when present, and
whether the recorded command starts with `uv run`.

`posetestbot.pipeline.sequences` composes stage specs into dependency-aware
workflows such as `sync_aruco`, `sync_to_bop_dry_run`,
`capture_to_bop_foundationpose_dry_run`,
`foundationpose_to_bop_eval_dry_run`, `aruco_to_bop_eval_dry_run`,
`megapose_to_bop_eval_dry_run`, and `sam6d_to_bop_eval_dry_run`.
Sequence plans write `pipeline_sequence_plan.json`, snapshot per-step
commands/resources/options, and can be created through
`scripts/run_pipeline_sequence.py --plan-only` or queued through
`/pipeline/run-sequence`. Sequence option defaults and caller overrides may use
`{run_root}` when a later step needs a path produced inside the same run folder.
The capture-to-BOP/FoundationPose preset starts from an existing captured run,
keeps BlenderProc rendering and FoundationPose Docker execution in dry-run mode,
and is meant to bridge capture folders toward BOP dataset export and estimator
planning before real runtime execution is enabled.
The MegaPose/SAM6D-to-BOP-eval presets include the dry-run estimator adapter
step before result export, so result-export output paths usually need caller
overrides until the wrappers are confirmed and producing outputs on this lab
machine.
Sequences that start from non-destructive sync run the `sync_quality` stage
immediately after `sync_run`, before ArUco, BlenderProc preparation, or BOP
export. Keep that quality gate in the dependency chain unless there is a clear
operator-facing reason to bypass it.
`sync_aruco_calibration_observations` extends that path with ArUco pose
estimation and `calibration_observations`, producing solver inputs without
claiming a solved calibration profile.
Prefer adding reusable workflow ordering there instead of open-coding command
chains in Flask handlers or shell scripts.

`posetestbot.pipeline.run_config` owns the first versioned operator intent
artifact, `run_config.v1`. It records the fake-or-real robot profile, capture
defaults, intended sensor list, object folder, optional calibration profiles,
and the default pipeline sequence/options for a run. Create it with
`uv run python scripts/create_run_config.py <run>`. The CLI defaults to the
current lab profile of 3 RealSense D435 cameras, 1 OAK-D Pro, and 1 ZED 2i,
with fake iiwa and `plan_only=true`. Use `--robot-mode real` only when the user
explicitly wants the physical iiwa target. The Flask transition API exposes
`GET /run-config?run_root=...` for validating/previewing the saved config plus
derived sequence plan and saved-preflight queue readiness, `POST /run-config`
for creating/updating it, with the transition page rendering that queue
readiness after save/load/queue attempts,
`GET /pipeline/preflight?run_root=...` for robot/sensor/runtime readiness
checks before queueing, `POST /pipeline/preflight` or
`uv run python scripts/run_preflight.py <run> --write` for writing
`run_preflight_report.json` plus a manifest `run_preflight` stage, and
`POST /pipeline/run-config` for queueing the saved sequence without repeating
options in the request body. `POST /pipeline/run-config` rejects a missing
`run_preflight_report.json` unless `allow_missing_preflight: true` is supplied,
rejects a saved report whose `overall_status` is `error` unless
`allow_failed_preflight: true` is supplied, and rejects stale preflight snapshots
whose embedded config no longer matches `run_config.json` unless
`allow_stale_preflight: true` is supplied. The transition page exposes those
three queue overrides as default-off checkboxes. Keep the saved-preflight
queue-readiness summary in `posetestbot.pipeline.preflight` so UI/API callers
share the same missing/invalid/failed/stale/ready contract. The object folder is
operator-facing in the CLI and transition UI/API, must not be blank, and should
point at the object registry consumed later by BlenderProc, BOP export, and
estimator stages. The current Flask index page is still transitional, but it now
has direct controls for saving/loading, preflighting, writing a preflight
snapshot, and queueing the run config, plus compact robot/sensor/runtime,
job/resource, artifact, metric, and BOP frame/result inspector panels.

`posetestbot.pipeline.capture_plan` owns the non-executing capture startup plan,
`capture_plan.v1`. It turns `run_config.json` into explicit `uv run python ...`
command arrays for the fake iiwa controller when configured, the robot pose
receiver, and each enabled RealSense/OAK-D Pro/ZED 2i capture process. Generate
it with `uv run python scripts/run_capture_plan_stage.py <run>` or through the
Flask transition API, `POST /capture-plan`, then inspect it with
`GET /capture-plan?run_root=...`. The current index page has compact
write/load controls that render command order for operators. The stage records
`capture_plan.json`, planned sensor folders, and a manifest `capture_plan`
stage without opening cameras or sending robot commands. RealSense and OAK-D
Pro planning is currently 720p-only; ZED planning passes the configured
`--resolution` through to `scripts/capture_zed_2i.py`.

`posetestbot.pipeline.capture_plan_preflight` validates a capture plan before
execution. It checks `uv run python` command shape, script existence,
robot-controller role counts, fake/real robot safety, static adapter support for
the configured resolution, duplicate/nonempty sensor output folders, and optional
sensor SDK and device readiness, then writes `capture_plan_preflight_report.json`
through `uv run python scripts/run_capture_plan_preflight.py <run>` or
`POST /capture-plan/preflight`. The typed `capture_plan_preflight` stage is part
of `fake_capture_rehearsal`, with sensor discovery skipped there because that
sequence remains pose-only and should not depend on camera availability.

`posetestbot.pipeline.capture_execution` owns the non-executing capture command
selection artifact, `capture_execution_plan.v1`. It consumes the saved
`run_config.json`/`capture_plan.json`, reuses capture-plan preflight, and writes
`capture_execution_plan.json` with selected commands, skipped commands, safety
gates, resources, and the planned supervisor/stop policy. Default
`pose_only_fake` mode selects only `robot_controller` plus
`robot_pose_receiver`, skips all camera commands, and requires fake iiwa mode.
`full` mode is still planning-only and requires `--allow-cameras` before camera
commands are selected. Generate it with
`uv run python scripts/run_capture_execution_plan.py <run>` or
`POST /capture-plan/execution`. The transition page exposes the same mode,
camera allowance, sensor-check, and real-robot allowance gates with fake-only
defaults; POST handlers parse string booleans through `_truthy` so values like
`"false"` do not accidentally unlock cameras. The typed
`capture_execution_plan` stage sits between capture-plan preflight and the
executable fake step in both `fake_capture_rehearsal` and
`fake_capture_execution`.

`scripts/run_capture_execution_stage.py` is the first general supervised
capture runner. It consumes the same execution-plan contract, starts selected
background commands in process groups, runs the robot pose receiver as the
controlling foreground command, stops remaining background process groups after
the receiver exits, and writes `capture_execution_report.json` plus per-command
logs under `capture_execution_logs/`. The default `pose_only_fake` mode only
runs the fake iiwa controller plus pose receiver. Full camera execution is
code-gated behind `--allow-cameras`; report process statuses and
`termination_reason` fields are the current evidence for camera startup/teardown
behavior. The supervisor also updates `capture_execution_status.json` at key
phases so operators and tests can inspect active process counts, per-process
status, log paths, selected roles, and raw-pose counts before the final report is
written. The Flask transition layer now exposes `/capture/jobs`,
`/capture/status`, and `/capture/jobs/<job_id>/stop` as capture-focused views
over the local job runner, so operators can see active capture jobs, held
resources, latest supervisor status, log links, and stop supervised capture runs
through process-group cancellation. Full mode still needs hardware validation
and richer per-process live telemetry before it should be treated as production
capture orchestration.

`posetestbot.pipeline.capture_rehearsal` owns the first executable fake capture
bridge. It builds a `capture_plan.v1` variant, selects only the
`robot_controller` and `robot_pose_receiver` commands, starts
`iiwa/fake_iiwa_controller.py` plus `scripts/pose_receiver_udp_json.py`, writes
`raw_robot_ee_poses.json` plus `capture_rehearsal_report.json`, and records a
`capture_rehearsal` manifest stage. It intentionally refuses real-robot configs
and does not start sensor capture scripts. Invoke it with
`uv run python scripts/run_capture_rehearsal_stage.py <run>` or queue the typed
`capture_rehearsal` stage through `/pipeline/run`. The
`fake_capture_rehearsal` sequence writes the capture plan, preflights it, writes
the fake execution plan, then runs this pose-only fake rehearsal.

`posetestbot.calibration.observations` owns `calibration_observations.json`.
It reads synchronized `aruco_pose_estimation.json` files, keeps only frames with
enough detected markers, valid OpenCV `rvec/tvec`, and matched
`robot_ee_pose`, then records rejected frames with reasons. This is solver input
for future robust hand-eye/static-camera calibration, not a solved calibration
profile. Generate it with
`uv run python scripts/run_calibration_observations.py <run>` or
`POST /calibration/observations`.

`posetestbot.aruco.coverage` owns `aruco_coverage_report.json`. It summarizes
existing synchronized `aruco_pose_estimation.json` files without rerunning
OpenCV detection, records frame/detection/pose/valid-pose counts and coverage
ratios per sensor, and writes a manifest `aruco_coverage` stage. Generate it
with `uv run python scripts/run_aruco_coverage_stage.py <run>` or queue the
typed `aruco_coverage` stage. Treat it as an inspection/readiness artifact
before calibration observation extraction or ArUco BOP result export.

`posetestbot.calibration.candidates` owns `calibration_candidates.json` and
`calibration_profiles_from_observations.json`. It consumes observation reports,
uses the legacy ArUco grid/template transform by default, averages per-frame
camera-to-end-effector or camera-to-robot-base transforms, rejects candidate
outliers with configurable residual thresholds, records per-frame residuals plus
inlier/outlier counts, and emits profiles with status `needs_validation`. Treat
these as inspection artifacts until a later validation/promotion step marks a
profile `valid`.
Generate them with `uv run python scripts/run_calibration_candidates.py <run>`
or `POST /calibration/candidates`.

`posetestbot.calibration.solver` owns `calibration_solver_report.json` and
`calibration_profiles_solved.json`. It consumes `calibration_observations.json`;
eye-in-hand sensors use OpenCV `calibrateHandEye`, while static sensors use the
configured calibration-target-to-robot-base transform and residual consistency
filtering. `--holdout-fraction` reserves a deterministic train/held-out split
when enough observations are present; holdout residuals are written to the
solver report and solved-profile metadata. `--compare-hand-eye-methods`
evaluates all OpenCV hand-eye methods for eye-in-hand sensors and records
method comparison residuals in `method_comparisons` without changing the
selected profile method. Solver profiles are still `needs_validation`; validate
them with the solved profile collection before promotion:

```bash
uv run python scripts/run_calibration_validation.py <run> \
  --candidates calibration_solver_report.json \
  --profiles calibration_profiles_solved.json
```

Run the solver with
`uv run python scripts/run_calibration_solver.py <run> --holdout-fraction 0.2 --compare-hand-eye-methods`
or `POST /calibration/solver`.

`posetestbot.calibration.validation` owns `calibration_validation_report.json`
and explicit promotion to `calibration_profiles.json`. It consumes candidate
or solver reports/profile collections, checks inlier count, mean residuals, and
outlier ratio, then writes active `valid` profiles only when `--promote` or
`promote: true` is explicitly supplied and every validation gate passes. Run it
with `uv run python scripts/run_calibration_validation.py <run>` or
`POST /calibration/validation`.

`posetestbot.io.artifact_browser` is the transition artifact browser for run
folders. It collects manifest-listed artifacts, stage artifacts, sequence plans,
BOP export/result/evaluation files, sync quality summaries for scan-friendly UI
listings, legacy accuracy metric summaries, BOP Toolkit score summaries from
evaluation reports, safe previews for paths under a run root, and
dashboard-ready run metric summaries via `metric_dashboard_summary`, plus
frame-level BOP scene drill-down via `bop_scene_detail` and BOP19 result CSV
drill-down via `bop_result_detail`. `bop_frame_detail` joins one BOP frame with
RGB/depth metadata, camera/GT/mask records, frame-map provenance, and optional
matching BOP19 result rows for side-by-side inspection. Flask exposes
`/artifacts`, `/artifacts/preview`, `/artifacts/file`, `/artifacts/metrics`,
`/artifacts/bop-scene`, `/artifacts/bop-result`, `/artifacts/bop-frame`, and
`/artifacts/bop-frame-overlay`; the transition page can render metric and BOP
Toolkit score summaries, load BOP result CSV summaries, and inspect one-frame
RGB/depth/mask/GT/provenance plus pose-row bundles without hand-building URLs.
Run-config and run-preflight artifact summaries should surface saved-preflight
queue readiness from `posetestbot.pipeline.preflight.run_preflight_queue_summary`
so artifact listings, recommendations, and queue APIs share
missing/invalid/failed/stale/ready vocabulary.
`render_bop_frame_overlay_png` composites masks onto RGB, draws GT boxes, and
adds BOP19 result score labels; BOP19 rows do not carry 2D result boxes, so do
not imply projected result geometry unless a future output contract adds it. Job
logs are linked from artifact listings but are still read through
`/jobs/<job_id>/log`. Keep preview/download/drill-down behavior path-safe and
scoped to the run root unless a future API adds an explicit external-artifact
trust model.

Validate or migrate profiles with:

```bash
uv run python scripts/validate_calibration_profiles.py <profile-or-collection.json>
uv run python scripts/validate_calibration_profiles.py --legacy-camera-ee scripts/default_data/camera_ee_transform.json --legacy-sync-data scripts/default_data/sync_data.json --output /tmp/posetestbot_calibration_profiles.json
uv run python scripts/create_run_config.py <run> --sequence sync_to_bop_dry_run --print-sequence-plan
```

Run non-destructive sync as:

```bash
uv run python scripts/sync_non_destructive.py <run>/<sensor>
uv run python scripts/sync_run_non_destructive.py <run>
uv run python scripts/run_sync_quality.py <run>
uv run python scripts/run_calibration_observations.py <run>
uv run python scripts/run_calibration_solver.py <run> --holdout-fraction 0.2 --compare-hand-eye-methods
uv run python scripts/run_calibration_candidates.py <run>
uv run python scripts/run_calibration_validation.py <run>
```

This command writes under `processed/synchronized/<sensor>/` and should not
modify raw `rgb/` or `depth/` files.

`posetestbot.sync.quality` owns the run-level `sync_quality_report.json`
contract. It aggregates per-sensor `sync_report.json` files, records matched and
dropped frame counts, checks match ratio, optional dropped-frame thresholds,
nearest-pose delta thresholds, and optional timestamp-source expectations, then
updates the manifest as `sync_quality`. Flask exposes the same write path at
`POST /sync/quality` and a read-only build path at `GET /sync/quality`.

The derived sync folder copies legacy camera sidecars so downstream scripts can
be pointed at `processed/synchronized/<sensor>`:

```bash
uv run python scripts/run_aruco_stage.py <run>
uv run python scripts/run_blenderproc_prepare_stage.py <run> --calibration-profiles <calibration_profiles.json>
uv run python scripts/run_blenderproc_render_stage.py <run> --dry-run
uv run python scripts/run_foundationpose_stage.py <run> --dry-run
uv run python scripts/run_bop_export_stage.py <run> --calibration-profiles <calibration_profiles.json> --object-folder object_models
```

## Validation

Use `uv` for tests too:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

The current tests cover robot profile defaults, UDP command shapes, fake iiwa
command parsing, artifact constants, the run manifest helper, and
non-destructive synchronization. They also cover the BlenderProc prep fallbacks
needed for derived synchronized folders, the manifest-tracked BlenderProc
prep/render stages, the sensor status JSON contract, and BOP export including
calibration metadata. They also cover `calibration.v1` profile validation and
legacy calibration migration plus BlenderProc preparation from calibration
profiles. Sync quality coverage checks per-sensor aggregation, thresholds,
manifest updates, CLI output, pipeline sequences, Flask API, and artifact
summaries. BOP fixture coverage includes importing BlenderProc GT JSON and masks
when present.

## Known Sharp Edges

- The real KUKA Sunrise app in `iiwa/HRC_Hub_Cap.java` appears to parse only
  legacy `start` commands and does not clearly implement stop handling.
- The legacy sync script is destructive by default. Prefer the
  `scripts/sync_*non_destructive.py` path for new work.
- On 2026-06-16, an unsandboxed `uv run python scripts/sensor_status.py --json`
  saw all 3 RealSense cameras but reported 0 OAK-D Pro devices because DepthAI
  warned that udev permissions were insufficient for an unbooted device.
- The ZED SDK Python module is provided by Stereolabs, not ordinary PyPI package
  management. On 2026-06-16, `pyzed.sl` was not available in the uv environment.
- FoundationPose still runs in its Docker/runtime environment; PoseTestBot-side
  wrappers should still be launched through `uv`.
