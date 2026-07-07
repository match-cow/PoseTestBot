# PoseTestBot Rewrite Progress

Last updated: 2026-07-07

## Current Phase

Phase 1.5: prove a narrow rewrite golden path before adding more surface area.

The immediate goal is to stop expanding transition UI/reporting surface until
the repo can prove one concrete path with run artifacts:

```bash
uv run python scripts/run_rewrite_gate.py <run> --write
```

Use the aggregate milestone view when asking how far the rewrite is:

```bash
uv run python scripts/run_rewrite_status.py <run> --write
```

It audits all current rewrite gates and writes `rewrite_status_report.json`
with ready/blocked gate counts, ready/blocked check counts, and the first
blockers to clear next. The report also records `next_gate` and
`next_actions` command arrays, so a blocked status points at the next concrete
operator command instead of only describing missing evidence.
When a single-root status report has already proved
`rewrite_fake_end_to_end.v1`, the default `rewrite_full_capture.v1` evidence
root is a sibling named `<status-root>_real_full_capture`; this keeps the next
real-run commands from rewriting the fake proof run.

When milestone evidence lives in different run folders, keep the aggregate
status in a separate root and pass per-gate evidence roots:

```bash
uv run python scripts/run_rewrite_status.py /tmp/posetestbot_rewrite_status --write \
  --gate-run-root rewrite_fake_end_to_end.v1=/tmp/posetestbot_gate_full_smoke \
  --gate-run-root rewrite_full_capture.v1=<real-capture-run>
```

This avoids rewriting the fake smoke run just to prove a real-capture gate.

Run the full hardware-free smoke path as:

```bash
uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_gate_full_smoke --overwrite
```

This smoke opens local UDP sockets for the fake iiwa controller and pose
receiver, so sandboxed coding environments may need explicit permission for
that one step. It does not touch the physical robot or camera SDKs.

The first gate, `rewrite_fake_end_to_end.v1`, requires current-run evidence for
`run_config.json`, `run_preflight_report.json`, a succeeded fake
`capture_execution_report.json` with raw poses, `synthetic_rgbd_report.json`,
`sync_quality_report.json`, `bop/bop_export_manifest.json`,
`bop_result_export_manifest.json`, `bop_evaluation_report.json`, and
`results/metrics/metric_report.json`. Plan files alone are not enough to pass
this gate. Missing or structurally weak evidence is reported as a blocker in
`rewrite_gate_report.json`.

The next gate, `rewrite_full_capture.v1`, keeps that fake proof distinct from
real lab capture validation:

```bash
uv run python scripts/run_rewrite_gate.py <run> --gate rewrite_full_capture.v1 --write
```

It requires `run_config.json` to target `robot_profile.mode=real` with enabled
sensors, `capture_execution_report.json` to prove a succeeded `mode=full`
supervised capture with camera commands selected and raw robot poses recorded,
and each enabled raw sensor folder to contain RGB/depth PNGs plus
`frame_metadata.jsonl`.

Keep new work pointed at whichever gate is next. Defer richer panels, additional
dry-run sequence variants, and optional BOP/COCO/multiview polish unless they
directly unblock one of these proof gates.

Current rewrite distance: the repo has a passing hardware-free fake golden path,
but it does not yet have current evidence for real full capture, real estimator
runtime execution, production calibration, or live BOP Toolkit scoring. The
latest aggregate status is blocked with 1/4 rewrite gates ready and 12/26
checks ready; `rewrite_full_capture.v1` is blocked with 3/12 checks ready.
Treat it as roughly at the "artifact contracts, explicit gates, and fake proof
are in place" milestone, not at the "lab workflow is validated" milestone.

Read-only status checks on 2026-06-19 in this managed workspace selected the
fake robot profile, saw no connected cameras through the sandboxed discovery
path, and reported missing external runtimes for BlenderProc, FoundationPose,
MegaPose, SAM6D, BOP Toolkit, and ZED SDK Python. These checks do not prove the
lab hardware itself is absent, but they do prove this workspace currently cannot
claim the real full-capture or runtime gates are passing.

Read-only RealSense status on 2026-07-07 still reports `pyrealsense2` as
importable but blocks discovery with
`RuntimeError: could not initialize udev monitor`, so the next lab action is
USB/udev/container visibility rather than robot motion. Use the RealSense-only
gate while OAK-D Pro, ZED, and real iiwa work stay out of scope:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json \
  --expected realsense_d435=3 --expected oak_d_pro=none --expected zed_2i=none \
  --check-expected
```

## Done

- Added a `posetestbot` package skeleton for shared rewrite code.
- Added canonical artifact constants in `posetestbot.io.artifacts`.
- Added fake-first robot profile defaults in `posetestbot.config`.
- Recorded the current real iiwa profile:
  - Robot IP: `172.31.1.147`
  - Receiver IP on robot subnet: `172.31.1.169`
  - Normal network IP: `10.145.8.132`
- Added UDP helpers for legacy and `robot_command.v1` start/stop messages.
- Added `posetestbot.robot.status` and `scripts/robot_status.py` for read-only
  fake/real iiwa profile snapshots, lab robot endpoint visibility, environment
  override reporting, and supported command-protocol metadata.
- Added Flask `GET /robot/status` and a manual Robot panel on the transition
  page so operators can confirm fake-first development mode without sending UDP
  commands to the real iiwa.
- Updated `start_iiwa.py`, `stop_iiwa.py`, and
  `scripts/pose_receiver_udp_json.py` to use robot profiles.
- Updated `iiwa/fake_iiwa_controller.py` to understand legacy and v1 robot
  commands.
- Updated the pose receiver to bind before sending a start command and to record
  monotonic receive timestamps.
- Updated script orchestration paths to prefer `uv run python`.
- Fixed the capture wrapper pose artifact mismatch:
  `raw_robot_ee_poses.json` is now copied into sensor folders for sync.
- Added sensor contracts for RealSense D435, OAK-D Pro, and ZED 2i aligned
  RGB-D frames.
- Added `posetestbot.sensors.realsense` and
  `scripts/run_realsense_capture_smoke.py` as the first RealSense-only camera
  proof path. The smoke runner reads a RealSense-only `run_config.json`,
  requires three explicit visible D435/D435i serials, refuses nonempty raw
  sensor folders, captures short sequential RGB-D samples, writes
  `realsense_capture_smoke_report.json`, and records a manifest
  `realsense_capture_smoke` stage.
- Hardened `scripts/capture_realsense_720p.py` for headless lab capture:
  preview windows are opt-in via `--preview`, `--warmup-frames` discards
  startup frames before writing, and startup/device failures now return clear
  stderr messages instead of exiting from the middle of the capture script.
- Added best-effort sensor discovery helpers.
- Added `posetestbot.sensors.registry` and `scripts/sensor_adapters.py` as a
  static adapter registry for supported RGB-D families. It centralizes display
  names, SDK modules, capture scripts, folder prefixes, supported resolutions,
  mounting modes, and command construction for RealSense D435, OAK-D Pro, and
  ZED 2i without opening hardware.
- Added `frame_metadata.jsonl` sidecars to RealSense and Luxonis capture.
- Added `posetestbot.sensors.frame_writer` as the shared legacy RGB-D image-pair
  and `frame_metadata.jsonl` writer, including `AlignedRgbdFrame` support for
  future adapter implementations.
- Updated RealSense D435, OAK-D Pro, and ZED 2i capture scripts to use the
  shared frame writer while preserving legacy camera sidecar outputs.
- Extended the shared frame writer with `write_legacy_camera_sidecars` for
  `cam_K.txt`, `depthscale.txt`, `camera.json`, and `camera_data.json`, and
  migrated all three capture scripts to that common sidecar writer.
- Fixed Luxonis capture device selection (`--device`) and output folder
  ownership.
- Added an initial ZED 2i capture script with the same legacy folder and
  metadata contract.
- Added `trimesh` with `uv add` because the FoundationPose wrapper imports it.
- Added `pytest` as a uv-managed development dependency.
- Added `dataset_manifest.json` helpers for manifest-backed run metadata.
- Added `posetestbot.pipeline.rewrite_gate` and
  `scripts/run_rewrite_gate.py` as the first explicit rewrite milestone audit.
  The gate checks concrete run artifacts for the fake end-to-end proof path and
  deliberately treats plan-only evidence as incomplete.
- Added `posetestbot.pipeline.synthetic_rgbd` and
  `scripts/create_synthetic_rgbd_fixture.py` to close the fake-capture data gap.
  It writes a manifest-tracked `realsense_synthetic/` folder with legacy
  RGB/depth PNGs, camera sidecars, and `frame_metadata.jsonl` records aligned to
  existing `raw_robot_ee_poses.json`, so the non-destructive sync and BOP export
  bridge can be exercised without camera hardware.
- Added `posetestbot.evaluation.synthetic_bop_results` and
  `scripts/create_synthetic_bop_results.py` as explicit fixture evidence for
  the fake BOP evaluation path. It writes deterministic BOP19 result CSV rows
  from `bop_export_manifest.json`, refreshes BOP19 target rows for those
  synthetic predictions, records `bop_result_export_manifest.json`, and labels
  the source as synthetic rather than estimator output.
- Added `scripts/run_rewrite_fake_e2e_smoke.py` to orchestrate the current
  hardware-free golden path through the existing stage CLIs, ending with
  `rewrite_gate_report.json`.
- Added `rewrite_full_capture.v1` to `scripts/run_rewrite_gate.py` so the first
  real lab capture validation has an explicit audit target and cannot be implied
  by the fake/synthetic smoke gate.
  The gate now requires each enabled raw sensor folder to have matching positive
  RGB/depth PNG counts plus `frame_metadata.jsonl`, so a partial RGB-only or
  depth-only capture cannot satisfy full-capture evidence.
- Added a typed `rewrite_gate` pipeline stage and made pipeline recommendations
  mode-aware for capture execution: fake run configs still recommend the safe
  pose-only path, while real run configs now recommend full capture planning and
  execution with explicit `--allow-cameras`/`--allow-real-robot` gates, then
  recommend auditing `rewrite_full_capture.v1` before treating hardware capture
  as validated.
- Added artifact-browser support for `rewrite_gate_report.json`, including
  scan-friendly gate ID, ready/blocker counts, next-blocker names, and display
  labels such as `rewrite_gate=ready` or `rewrite_gate=blocked_rewrite_gate`.
- Added `scripts/run_rewrite_status.py`, the typed `rewrite_status` stage, and
  artifact-browser support for `rewrite_status_report.json` so operators can
  answer "how far are we?" across all rewrite gates in one report instead of
  inferring progress from scattered plan artifacts.
- `rewrite_status_report.json` now includes `next_gate` and `next_actions`.
  For the current fake-smoke evidence, the first action is creating an
  intentional real lab run config for `rewrite_full_capture.v1` in a separate
  full-capture evidence root with `--sequence real_full_capture_validation`,
  followed by a plan-only sequence preview, an explicit sequence execution
  command, individual capture plan/preflight/full execution commands, and the
  gate-audit command.
- Added the `real_full_capture_validation` pipeline sequence as the first
  reusable workflow for the blocked real-capture milestone. It now writes a
  checked run preflight snapshot, records a run-scoped hardware status snapshot,
  writes the capture plan, preflights with `--allow-real-robot`, writes a full
  execution plan with `--allow-cameras --allow-real-robot --include-sensors`,
  runs full supervised capture with the same explicit gates, then audits
  `rewrite_full_capture.v1`. This is intentionally not the default fake
  development path and still requires real lab evidence before the gate can
  pass.
- Added a typed `run_preflight` pipeline stage for
  `scripts/run_preflight.py`, so sequence execution can record
  `run_preflight_report.json` with the same command-builder and resource
  declaration path as other transition stages.
- Artifact-browser summaries for `rewrite_status_report.json` now expose the
  `next_gate_id`, first `next_action_label`, first `next_action_command`, and a
  compact `next_gate=...` display label so the transition UI can show the next
  concrete command without parsing the full report body.
- `rewrite_status_report.json` now supports per-gate run roots through repeated
  `--gate-run-root GATE_ID=RUN_ROOT` CLI options. The report records
  `gate_run_roots`, each gate summary records its evidence `run_root`, and
  next actions use the blocked gate's own run root. This keeps repo-level
  progress honest when fake-smoke, real-capture, runtime, and calibration
  proofs are produced by separate runs.
- When no explicit full-capture root is supplied and the fake end-to-end gate is
  ready, aggregate rewrite status now records
  `rewrite_full_capture.v1=<status-root>_real_full_capture` by default. This
  avoids suggesting a real-mode `run_config.json` write into the fake-smoke run
  folder.
- Full-capture next actions now skip the plan-only recommendation when the
  blocked gate root already has `pipeline_sequence_plan.json`, then proceed
  through the first failing safety prerequisite instead of jumping directly to
  execution.
- Full-capture next actions now stop at the first unresolved safety blocker in
  gate order: run preflight, hardware status, capture plan, capture-plan
  preflight, and capture execution plan must be acceptable before status
  suggests supervised full capture.
- `rewrite_full_capture.v1` now requires `hardware_status_report.json` to have
  selected the real robot profile, so a real-mode run config cannot be
  validated against a fake-profile hardware snapshot.
- When an existing `hardware_status_report.json` is blocked by sensor discovery
  errors, aggregate rewrite status now recommends the read-only
  `uv run python scripts/sensor_status.py --json --check-expected` diagnostic
  instead of simply rewriting the same hardware snapshot. The command keeps the
  JSON status payload on stdout and exits nonzero until expected camera counts
  are visible, making the recommendation usable by automation as well as by an
  operator.
- `rewrite_full_capture.v1` now audits the guarded workflow prerequisites, not
  only terminal capture output. The gate requires acceptable
  `run_preflight_report.json`, `hardware_status_report.json`,
  `capture_plan.json`, `capture_plan_preflight_report.json`, and
  `capture_execution_plan.json` evidence before the supervised full-capture
  report and raw sensor frame folders can satisfy the milestone.
- `rewrite_gate_report.json` and aggregate `rewrite_status_report.json`
  next-blocker entries now preserve prerequisite-report details, including
  blocked checks and flattened sensor diagnostics from hardware and
  capture-plan preflight reports. A blocked full-capture gate can therefore
  show RealSense/OAK-D/ZED visibility and SDK hints without requiring a second
  manual artifact lookup.
- The default human-readable `scripts/run_rewrite_gate.py` and
  `scripts/run_rewrite_status.py` outputs now print compact blocker details
  from that same diagnostic payload: blocked sub-checks, sensor diagnostic
  messages, and the first remediation hints. Operators can see why a gate is
  blocked without switching to `--json` for the common hardware-visibility
  failure path.
- Artifact-browser summaries for `rewrite_gate_report.json` and
  `rewrite_status_report.json` now expose the same compact blocker context as
  structured fields: `next_blocker_messages`, `next_blocker_diagnostics`,
  `next_blocker_hints`, and `next_blocker_checks`. The transition
  `/artifacts` API can therefore surface the current RealSense/OAK-D/ZED
  blocker causes without requiring operators to preview the full report JSON.
- Artifact display labels now include bounded `next_blocker=...` and
  `next_diag=...` fields for blocked rewrite gate/status artifacts. The
  transition page's artifact list can show the current full-capture blocker and
  first hardware diagnostic inline while keeping the row scan-friendly.
- Pipeline recommendations now surface the bounded `next_actions` sequence from
  a current but blocked `rewrite_status_report.json` as command
  recommendations. The first recommendation keeps the compatibility ID
  `follow_rewrite_status_next_action`; follow-up actions use numbered IDs such
  as `follow_rewrite_status_next_action_2`. Each recommendation preserves the
  exact command and action-specific reason recorded by the gate status report
  and includes the first blocker message/diagnostic in its reason, so a current
  aggregate status no longer leaves the recommendation panel empty while the
  rewrite is still blocked, hides the explicit follow-up command, or gives the
  follow-up action the wrong generic context.
- Pipeline recommendations now also preserve rewrite-status action
  `blocks_on` lists, and the transition recommendation panel renders them as a
  compact blocker line. The recommendation API and artifact API therefore agree
  on the RealSense/OAK-D/ZED blocker set for the current full-capture path.
- When a current blocked aggregate rewrite status has a guided next action,
  pipeline recommendations now suppress generic run-config setup for the
  aggregate status folder and rank the recorded rewrite-status action first.
  For the current mixed-root status, the only top recommendation is the
  read-only sensor diagnostic command that blocks `rewrite_full_capture.v1`.
- Run-scoped hardware status now resolves the selected robot profile from a
  saved `run_config.json` when present. Real full-capture evidence roots
  therefore record the intended real iiwa profile without requiring
  `POSETESTBOT_ROBOT_MODE=real` in the agent process environment, while ad hoc
  status checks remain fake-first.
- Rewrite-status next actions now use that run-scoped hardware-status behavior:
  stale fake-profile hardware snapshots are refreshed with
  `uv run python scripts/run_hardware_status_stage.py <run>` instead of an
  environment-variable wrapper.
- When sensor discovery errors block `rewrite_full_capture.v1`, aggregate
  rewrite status now records a two-step operator path: inspect
  `sensor_status.py --json --check-expected`, then rerun
  `run_hardware_status_stage.py <run>` after USB/SDK visibility is fixed. The
  human-readable `run_rewrite_status.py` output prints all recorded next
  actions instead of hiding the follow-up command.
- `scripts/sensor_status.py --json` now keeps stdout parseable as pure JSON
  even when vendor SDKs print hardware-discovery warnings to stdout; those
  messages are redirected to stderr while the status snapshot is collected.
  This keeps the current rewrite-status next action usable in shell and UI
  automation.
- Artifact-browser summaries for `rewrite_status_report.json` now expose bounded
  `next_action_labels` and `next_action_commands` lists in addition to the
  first-action compatibility fields, so the transition UI/API can show the same
  two-step sensor-blocker path without previewing raw JSON.
- Artifact-browser summaries for `rewrite_status_report.json` now also expose
  bounded `next_action_blocks_on` lists, so the transition API can show that the
  current strict sensor diagnostic and follow-up hardware refresh both block on
  RealSense, OAK-D Pro, and ZED 2i visibility without requiring a raw JSON
  preview.
- The transition artifact list now renders those next-action blockers inline
  with each rewrite-status next action, so operators can see the exact
  RealSense/OAK-D/ZED blocker set next to the command.
- The transition artifact list now renders those multi-step rewrite-status next
  actions inline under the artifact row, while retaining click-to-preview
  behavior. Operators can see the sensor-status diagnostic command and the
  follow-up hardware-snapshot refresh command directly in the page.
- Rewrite gate blocker extraction now accepts both `checks` and `gates`
  arrays, so `capture_execution_plan.json` reports can surface their failing
  safety gate. The current full-capture status now explains
  `capture_execution_plan` as blocked by `capture_plan_preflight`, instead of
  falling back to a generic command-selection message.
- BOP evaluation artifact summaries and pipeline recommendation facts now treat
  empty or absent `checks` arrays in older succeeded/planned
  `bop_evaluation_report.json` files as unrecorded prerequisite diagnostics
  rather than failed prerequisites. Partial check lists still block when
  critical checks are missing or failed, but legacy reports with score metrics
  can still feed the metric dashboard.
- Pipeline recommendations now reuse `gate_run_roots` from an existing
  `rewrite_status_report.json` before deciding whether the report is stale, so
  a valid mixed-root aggregate report is not incorrectly refreshed as a
  single-root report.
- The typed `rewrite_status` stage now accepts repeated
  `gate_run_root` options and emits repeated `--gate-run-root` CLI flags. When
  recommendations do need to refresh a stale mixed-root status report, the
  suggested command preserves those saved evidence roots.
- Pipeline recommendations now expose aggregate rewrite-status facts and
  recommend writing or refreshing `rewrite_status_report.json` when rewrite
  evidence exists but the aggregate report is missing, invalid, or stale. A
  current report suppresses that recommendation so the panel does not become
  permanent noise.
- Added `rewrite_foundationpose_runtime.v1` to `scripts/run_rewrite_gate.py`
  and the typed `rewrite_gate` stage. This estimator-runtime gate requires a
  non-dry-run `foundationpose_plan.json` with `ob_in_cam` pose files for every
  planned job, a FoundationPose-sourced BOP result manifest with existing CSV
  rows, and a succeeded non-dry-run BOP evaluation report with score metrics.
  It is an audit target for real runtime validation, not proof that this
  workspace has already run FoundationPose or BOP Toolkit.
- Pipeline recommendations now surface `rewrite_foundationpose_runtime.v1`
  readiness facts and recommend the gate audit after FoundationPose/BOP
  evaluation evidence appears but still falls short of real runtime proof, such
  as a dry-run BOP evaluation report.
- Added `rewrite_calibration_validation.v1` to `scripts/run_rewrite_gate.py`
  and the typed `rewrite_gate` stage. This calibration gate requires an
  `overall_status=ok` validation report with explicit successful promotion plus
  a promoted `calibration_profiles.json` collection whose profiles are marked
  `valid` and retain inlier/residual quality fields. Solver or candidate
  profiles that remain `needs_validation` do not satisfy the gate.
- Pipeline recommendations now surface `rewrite_calibration_validation.v1`
  readiness facts and recommend the gate audit when calibration validation has
  run but promoted production profiles are still missing or incomplete.
- Added `posetestbot.pipeline.run_config` with `run_config.v1`, a versioned
  operator-intent artifact for robot profile, capture defaults, intended
  sensors, object/calibration inputs, and default pipeline sequence options.
- Added `scripts/create_run_config.py` to write `run_config.json`, record it in
  `dataset_manifest.json`, and optionally print the derived sequence plan.
- `scripts/create_run_config.py --sequence` now uses the typed pipeline
  sequence registry for argparse choices, so CLI help and early validation stay
  aligned with `/pipeline/sequences`.
- The default generated run config is fake-iiwa-first and captures the current
  lab profile of 3 RealSense D435 cameras, 1 OAK-D Pro, and 1 ZED 2i.
- Added run-config helpers for loading `run_config.json` from a run root,
  checking run-root consistency, and building the configured sequence job from
  the saved artifact.
- Added shared run-config sensor parsing for text tokens and JSON sensor
  objects, plus a helper that writes `run_config.json` and records the
  manifest stage consistently for CLI and web API callers.
- Exposed `object_folder` through the transition run-config UI/API and tightened
  validation so CLI/API-created run configs reject blank object-folder values
  before downstream BlenderProc, BOP, or estimator stages consume them.
- Added `posetestbot.pipeline.capture_plan` and
  `scripts/run_capture_plan_stage.py` to turn `run_config.json` into a
  manifest-tracked `capture_plan.json` with explicit `uv run python ...`
  startup commands for fake iiwa, robot pose receiving, and the configured
  RealSense/OAK-D Pro/ZED 2i sensor captures.
- Added artifact-browser summary support for `capture_plan.json` so planned
  command count, sensor count, robot mode, and command roles are visible beside
  other run artifacts.
- Added Flask `GET /capture-plan` and `POST /capture-plan` APIs plus compact
  transition-page controls for writing/loading and rendering the capture startup
  command order from `run_config.json`.
- Added `posetestbot.pipeline.capture_rehearsal`,
  `scripts/run_capture_rehearsal_stage.py`, a typed `capture_rehearsal`
  pipeline stage, artifact-browser summary support, and a transition-page queue
  action for a pose-only fake iiwa rehearsal. The stage runs
  `iiwa/fake_iiwa_controller.py` plus `scripts/pose_receiver_udp_json.py`,
  writes `raw_robot_ee_poses.json` and `capture_rehearsal_report.json`, and
  refuses real-robot configs.
- Refactored `capture_rehearsal` to derive its fake controller and pose receiver
  commands from the same `capture_plan.v1` command model used by
  `capture_plan.json`, then added the `fake_capture_rehearsal` sequence that
  writes a capture plan before running the pose-only rehearsal.
- Added `posetestbot.pipeline.capture_plan_preflight`,
  `scripts/run_capture_plan_preflight.py`, the typed `capture_plan_preflight`
  stage, Flask `/capture-plan/preflight`, and artifact-browser support for
  `capture_plan_preflight_report.json`. It validates command shape, script
  availability, fake/real robot safety, static adapter/resolution support,
  duplicate or nonempty planned sensor output folders, and optional sensor
  SDK/device readiness before any capture process starts.
- Capture-plan preflight now reports unsupported configured sensor resolutions
  and duplicate output folders as structured checks without throwing before a
  report can be written, so operators get readable launch-blocking evidence even
  when `capture_plan.json` cannot be built.
- Added `posetestbot.pipeline.capture_execution`,
  `scripts/run_capture_execution_plan.py`, the typed `capture_execution_plan`
  stage, Flask `/capture-plan/execution`, and artifact-browser support for
  `capture_execution_plan.json`. It reuses capture-plan preflight, selects the
  fake iiwa controller plus pose receiver in default `pose_only_fake` mode,
  skips camera commands explicitly, and gates full capture command selection
  behind `--allow-cameras` while still remaining non-executing.
- Exposed capture execution mode, camera allowance, sensor-check, and real-robot
  allowance controls in the transition UI, keeping fake-only execution as the
  default and preserving the same safety gates as the CLI/API.
- Tightened `/capture-plan/execution` POST boolean parsing so string values such
  as `"false"` remain false and cannot accidentally pass camera or real-robot
  gates.
- Added `scripts/run_capture_execution_stage.py`, the typed
  `capture_execution` stage, artifact-browser support for
  `capture_execution_report.json` and `capture_execution_logs/`, and a
  `fake_capture_execution` sequence. The default supervised execution path
  starts only fake iiwa plus the pose receiver from the selected execution plan,
  writes per-command logs, terminates remaining process groups, records
  return codes/log tails/raw pose count, and remains camera-free unless full
  mode is explicitly allowed.
- Extended `capture_execution_report.json` process records with
  `termination_reason` and added mocked full-mode coverage proving a selected
  sensor capture command is started, supervised while the pose receiver runs,
  and reported as `stopped_after_receiver_exit` when the supervisor stops it.
- Extended capture execution process telemetry in both
  `capture_execution_status.json` and `capture_execution_report.json` with PID
  when available, start/end timestamps, elapsed time, and artifact-browser
  summaries for process timing and termination reasons.
- Added a transition Capture Activity UI/API bridge over the local job runner:
  `GET /capture/jobs` lists capture-related jobs, active counts, resource
  holders, log links, and stop endpoints scoped by run root, while
  `POST /capture/jobs/<job_id>/stop` delegates supervised capture shutdown to
  process-group cancellation and refuses unrelated jobs.
- Added `capture_execution_status.json`, updated during supervised capture
  startup/receiver/cleanup/final phases with active process counts,
  per-process status/log paths, selected roles, raw-pose counts, and surfaced it
  through artifact discovery plus Flask `/capture/status` and `/capture/jobs`.
- Extended pipeline recommendations and artifact summaries with capture
  execution report readiness, so missing, invalid, or non-succeeded
  `capture_execution_report.json` files expose a concrete blocker and prompt
  rerunning the safe fake capture execution path instead of silently unblocking
  downstream recommendations. Artifact scan labels now surface the same
  capture readiness/blocker state for quick UI inspection. Recommendation
  lookup also suppresses raw-sync suggestions for runs that have entered the
  supervised capture-execution path until the execution report is ready, while
  preserving legacy raw-pose sync recommendations for older captures.
- Updated the capture wrapper and standalone pose receiver to write/update
  run manifests with robot profile, capture config, sensor records, stages, and
  raw robot pose artifacts.
- Added automated tests for robot profiles, UDP command helpers, fake iiwa
  command parsing, artifact names, and manifest writing/loading.
- Added `posetestbot.sync.non_destructive`, which consumes `frame_metadata.jsonl`
  plus `raw_robot_ee_poses.json` and writes derived synchronized frames to
  `processed/synchronized/<sensor>/`.
- Added `scripts/sync_non_destructive.py`, which updates `dataset_manifest.json`
  with per-sensor sync stages and artifacts.
- Added automated tests proving non-destructive sync preserves raw frames while
  writing copied synchronized RGB/depth frames, `match_robot_ee_poses.json`, and
  `sync_report.json`.
- Updated non-destructive sync to copy legacy camera sidecars into derived
  synchronized folders so ArUco and BlenderProc prep can consume them.
- Hardened BlenderProc prep for derived synchronized folders:
  - three-line `cam_K.txt` files now get zero distortion coefficients;
  - serial-specific sensor folder names such as `realsense_123`,
    `luxonis_ABC`, and `zed_2i_42` fall back to sensor-type calibration keys.
- Added downstream compatibility tests for derived sync output and BlenderProc
  prep helpers.
- Added `scripts/sync_run_non_destructive.py` to synchronize every discovered
  sensor folder in a run and record aggregate/per-sensor manifest stages.
- Added `posetestbot.sync.quality` and `scripts/run_sync_quality.py` to
  aggregate per-sensor `sync_report.json` files into
  `sync_quality_report.json`, checking match ratio, dropped frames, nearest
  robot-pose deltas, and optional timestamp-source expectations while recording
  a `sync_quality` manifest stage.
- Added the typed `sync_quality` pipeline stage, Flask `/sync/quality`, a
  transition-page Sync Quality control, and artifact-browser summaries for
  `sync_quality_report.json`.
- Updated sync-backed pipeline sequences so `sync_quality` runs immediately
  after `sync_run` before ArUco, BlenderProc, BOP export, or FoundationPose
  planning.
- Added `scripts/run_aruco_stage.py` to run ArUco estimation on one derived
  synchronized sensor folder or all synchronized sensors in a run, recording
  `aruco_pose_estimation.json` as a manifest artifact.
- Added tests for run-level sync and manifest-tracked ArUco stage execution.
- Added `posetestbot.aruco.coverage` and
  `scripts/run_aruco_coverage_stage.py`, writing
  `aruco_coverage_report.json` with per-sensor detection counts, valid-pose
  counts, marker-count thresholds, coverage ratios, manifest tracking,
  artifact-browser summaries, typed pipeline stage support, and next-step
  recommendations.
- Added `scripts/run_blenderproc_prepare_stage.py` to prepare BlenderProc inputs
  from `processed/synchronized` and record generated `blenderproc/` folders in
  `dataset_manifest.json`.
- Fixed legacy BlenderProc prep to convert object transforms loaded from JSON
  into NumPy arrays before transform math.
- Added fixture tests proving the BlenderProc prep stage writes camera matrices,
  distortion coefficients, object files/transforms, camera poses, and manifest
  artifacts for derived synchronized folders.
- Added `scripts/run_blenderproc_render_stage.py` to validate prepared
  synchronized BlenderProc folders, write `blenderproc_render_plan.json`, and
  optionally execute BlenderProc rendering while recording the stage in
  `dataset_manifest.json`.
- Added render-stage tests for dry-run plan/manifest output and BlenderProc
  output cleanup into per-sensor `masks/` and `blenderproc/output/` folders.
- Added `posetestbot.sensors.status` and `scripts/sensor_status.py` for
  JSON-friendly camera SDK/device status across RealSense D435-class, OAK-D Pro,
  and ZED 2i families.
- Added sensor-status tests for expected lab counts, discovery errors,
  operator diagnostics, and expected-count override parsing.
- Sensor status now emits per-family diagnostics with hints for discovery
  errors, missing SDK modules, expected-count misses, USB/udev access, and ZED
  SDK Python availability.
- Added Flask `GET /sensors/status` and a manual Sensors panel on the transition
  page, reusing the shared sensor-status contract for RealSense/OAK-D Pro/ZED
  2i SDK availability, connected-device counts, expected-count checks, and
  discovery errors.
- Added Flask `GET /sensors/adapters` and a Sensors-panel adapter listing so
  operators can inspect registered capture scripts and supported resolutions
  without opening camera SDKs.
- Added `posetestbot.runtime.status` and `scripts/runtime_status.py` for
  lightweight external runtime readiness across BlenderProc, Docker-backed
  FoundationPose, MegaPose/SAM6D wrapper scripts, BOP Toolkit, and the ZED SDK
  Python module.
- Added Flask `GET /runtime/status` and a manual Runtimes panel on the
  transition page so operators can see missing external executables/checkouts
  without starting containers, render jobs, or BOP Toolkit evaluation.
- Added `posetestbot.pipeline.hardware_status`,
  `scripts/run_hardware_status_stage.py`, typed `hardware_status` pipeline stage,
  Flask `/hardware/status`, artifact-browser support, and a transition-page
  Snapshot action for run-scoped `hardware_status_report.json` files. The report
  records selected fake/real robot profile, sensor visibility, and external
  runtime readiness without starting robot motion, camera capture, containers,
  renders, or evaluations.
- Added `posetestbot.bop.writer` and `scripts/run_bop_export_stage.py` for a
  first manifest-tracked BOP-shaped export from derived synchronized sensor
  folders.
- Added a BOP export fixture test covering copied scene RGB/depth frames,
  `scene_camera.json`, empty ground-truth placeholders, frame provenance, and
  manifest stage artifacts.
- Added `posetestbot.calibration.profiles` with the `calibration.v1` profile
  schema for eye-in-hand and static-camera calibration records.
- Added `scripts/validate_calibration_profiles.py` to validate profile files or
  migrate legacy `camera_ee_transform.json` plus `sync_data.json` into
  `calibration.v1` profile collections.
- Added calibration profile tests for baseline JSON keys, transform-direction
  validation, legacy migration, and the migration CLI.
- Added calibration profile lookup helpers that match synchronized sensor folder
  names such as `realsense_123` to exact, serial, or legacy profile identities.
- Added `posetestbot.calibration.preflight`,
  `scripts/run_calibration_preflight.py`, the typed `calibration_preflight`
  stage, Flask `/calibration/preflight`, and artifact-browser support for
  `calibration_preflight_report.json`. It validates run-level calibration
  profile coverage, explicit profile IDs, mounting-mode matches, profile
  status, observation counts, and mean reprojection metrics before downstream
  stages consume calibrated extrinsics.
- Added `posetestbot.calibration.observations`,
  `scripts/run_calibration_observations.py`, the typed
  `calibration_observations` stage, Flask `/calibration/observations`, and
  artifact-browser support for `calibration_observations.json`. It extracts
  solver-ready observation pairs from synchronized target-pose detections and
  matched robot end-effector poses, while preserving rejected-frame reasons.
  The extractor keeps legacy `aruco_pose_estimation.json` compatibility and can
  also ingest `charuco_pose_estimation.json`,
  `checkerboard_pose_estimation.json`, or
  `calibration_target_pose_estimation.json` records with target-specific pose
  keys and feature counts.
- Added explicit calibration target metadata normalization in
  `posetestbot.calibration.targets` and threaded it through
  `calibration_observations.json`, CLI flags, typed pipeline stage options,
  Flask `/calibration/observations`, and artifact summaries. The default
  remains the current 4x3 `DICT_5X5_50` ArUco grid, while reports can now record
  ChArUco/checkerboard target metadata for future detector/capture workflows.
- Added `sync_aruco_calibration_observations`, a dependency-aware sequence that
  runs non-destructive sync, sync quality, ArUco estimation, then calibration
  observation extraction without claiming a solved calibration profile.
- Added `posetestbot.calibration.solver`,
  `scripts/run_calibration_solver.py`, the typed `calibration_solver` stage,
  Flask `/calibration/solver`, artifact-browser support for
  `calibration_solver_report.json` plus `calibration_profiles_solved.json`, and
  the `sync_aruco_calibration_solver` sequence. Eye-in-hand observations use
  OpenCV `calibrateHandEye`; static observations use target/reference transform
  consistency. Optional deterministic `--holdout-fraction` splits observations
  into train/held-out sets, reports held-out residuals, and records the summary
  in solved-profile metadata. Optional `--compare-hand-eye-methods` evaluates
  all OpenCV hand-eye solver methods for eye-in-hand sensors and writes
  `method_comparisons` to the solver report/transition UI summaries. Solver
  profiles remain `needs_validation`.
- Added `posetestbot.calibration.candidates`,
  `scripts/run_calibration_candidates.py`, the typed `calibration_candidates`
  stage, Flask `/calibration/candidates`, and artifact-browser support for
  `calibration_candidates.json` plus
  `calibration_profiles_from_observations.json`. It averages per-frame
  observation transforms into `needs_validation` profile candidates with
  residual-threshold outlier filtering, inlier/outlier counts, per-frame
  translation/rotation residuals, and profile quality metadata for inspection.
- Added `sync_aruco_calibration_candidates`, a dependency-aware sequence that
  runs sync, sync quality, ArUco estimation, calibration observation extraction,
  then validation-gated candidate generation.
- Added `posetestbot.calibration.validation`,
  `scripts/run_calibration_validation.py`, the typed `calibration_validation`
  stage, Flask `/calibration/validation`, and artifact-browser support for
  `calibration_validation_report.json`. It gates candidate profiles by inlier
  count, mean translation/rotation residuals, and outlier ratio, then promotes
  profiles to `calibration_profiles.json` only when explicitly requested.
- Added `sync_aruco_calibration_validation`, a dependency-aware sequence that
  runs sync, sync quality, ArUco estimation, observation extraction, candidate
  generation, then validation without automatic promotion.
- Updated `scripts/run_blenderproc_prepare_stage.py` to accept
  `--calibration-profiles`, generate
  `processed/calibration/camera_ee_transform_from_calibration_profiles.json`,
  and run the existing BlenderProc prep path from resolved calibration profiles.
- Extended the BlenderProc prep path to handle static calibration profiles by
  repeating camera-to-robot-base/cell-world poses for every synchronized frame,
  while preserving legacy eye-in-hand camera-to-EE behavior.
- Added BlenderProc prep fixtures proving eye-in-hand and static
  calibration-profile extrinsics feed into generated `camera_poses.npy` and
  manifest artifacts.
- Updated `scripts/run_bop_export_stage.py` to accept `--calibration-profiles`.
  Matching profiles now contribute BOP `scene_camera.json` intrinsics/depth
  scale and PoseTestBot calibration metadata, and are recorded in
  `bop_export_manifest.json`.
- Added a BOP export fixture proving calibration-profile metadata and
  intrinsics/depth scale are exported to scene camera and manifest artifacts.
- Extended BOP export to import BlenderProc `scene_gt.json`,
  `scene_gt_info.json`, sensor-level `masks/`, and optional `mask_visib/` output
  when those artifacts already exist.
- Added a BOP export fixture proving BlenderProc GT JSON and masks are copied
  into the BOP scene and recorded in `bop_export_manifest.json`.
- Extended BOP export to derive `scene_gt_info.json` bbox, pixel-count, and
  visibility metadata from mask images when BlenderProc produced `scene_gt.json`
  but did not produce `scene_gt_info.json`.
- Added a BOP export fixture proving mask-derived `scene_gt_info.json` metadata
  is written without requiring model export.
- Extended BOP export with object registry support from `objects.json`, model
  copying to `bop/models/obj_XXXXXX.ply`, `models_info.json`, numeric BOP
  object-ID normalization for string BlenderProc object names, and
  `test_targets_bop19.json` generation.
- Added a BOP export fixture proving model files, `models_info.json`, normalized
  `scene_gt.json`, target files, and manifest artifacts are written.
- Added optional PoseTestBot multiview target summaries for BOP export via
  `--write-multiview-targets`. The exporter writes
  `posetestbot_multiview_targets.json`, records it in
  `bop_export_manifest.json` and `dataset_manifest.json`, exposes the flag
  through the typed `bop_export` stage, and artifact discovery can list the
  manifest-linked summary.
- Added optional PoseTestBot COCO-style annotations for BOP export via
  `--write-coco-annotations`. The exporter writes
  `posetestbot_coco_annotations.json` from BOP RGB frames, `scene_gt.json`,
  `scene_gt_info.json`, and copied masks, records it in
  `bop_export_manifest.json` and `dataset_manifest.json`, exposes the flag
  through the typed `bop_export` stage, and artifact discovery reports image,
  annotation, and category counts.
- Extended `models_info.json` export with geometry-derived metadata when PLY
  vertices are readable: `diameter`, bounding-box minimums, extents, vertex
  count, and the diameter method.
- Added a BOP export fixture proving exact model diameter and bounding-box
  metadata are written for a small PLY fixture.
- Added `posetestbot.evaluation.bop_toolkit` and
  `scripts/run_bop_evaluation_stage.py` as the first BOP Toolkit evaluation
  bridge. It validates BOP19 result CSVs, builds the `eval_bop19_pose.py`
  command with `BOP_PATH`, writes `bop_evaluation_plan.json`, and records a
  `bop_evaluation` stage in `dataset_manifest.json`.
- Added BOP evaluation fixtures for dry-run plan/manifest output and local
  BOP19 result-row validation.
- Added `bop_evaluation_report.json` for the BOP Toolkit bridge. The report
  records dry-run/execution status, command/env, validated result metadata,
  prerequisite checks, and a capped inventory of files discovered under the
  evaluation output folder. It now also harvests numeric metrics from
  `scores*.json` files such as `scores_bop19.json` into `score_summary`, which
  is surfaced by artifact-browser summaries. It is recorded as a manifest
  artifact and surfaced by the artifact browser.
- Added `posetestbot.evaluation.bop_results` and
  `scripts/run_bop_result_export_stage.py` to convert FoundationPose
  `ob_in_cam` matrices into BOP19 pose-result CSVs under `results/bop/`, using
  `bop_export_manifest.json` for sensor-to-scene IDs and BOP model metadata for
  object IDs.
- Added BOP result-export fixtures covering FoundationPose output parsing,
  object/scene mapping, meters-to-millimeters translation scaling, BOP19 CSV
  validation, `bop_result_export_manifest.json`, and manifest stage artifacts.
- Extended BOP result export with `--source aruco` to convert synchronized
  `aruco_pose_estimation.json` OpenCV `rvec`/`tvec` entries into BOP19 result
  CSV rows using BOP export sensor/object metadata.
- Added ArUco result-export fixtures for object/scene mapping, marker-count
  filtering, default translation scaling, BOP19 CSV validation,
  `bop_result_export_manifest.json`, and manifest stage artifacts.
- Extended BOP result export with `--source megapose` and `--source sam6d`.
  MegaPose conversion reads `megapose_poses.json`, treats translations as meters
  by default, and exports BOP19 rows. SAM6D conversion reads
  `detections_pem/*.json`, picks the highest-scoring detection per frame,
  leaves translations unchanged by default, and exports BOP19 rows.
- Added MegaPose/SAM6D result-export fixtures covering source discovery by
  explicit output folder, object/scene mapping, source-specific translation
  scaling, BOP19 CSV validation, `bop_result_export_manifest.json`, and typed
  pipeline-stage options.
- Added `posetestbot.jobs.runner.LocalJobRunner` for structured command-array
  background jobs with status snapshots, log files, exit-code capture,
  cancellation, persisted job-record reload, interrupted-job marking, and
  declared-resource locking.
- Extended the job runner with POSIX process-group cancellation so child
  processes from a canceled job are terminated too, plus persisted job
  `parameters` snapshots for command configuration.
- Updated `web_interface.py` so the existing Flask buttons submit jobs through
  the local runner instead of blocking on `subprocess.check_output`, and added
  `/jobs`, `/jobs/<job_id>`, `/jobs/<job_id>/log`, and
  `/jobs/<job_id>/cancel` APIs.
- Added job-runner and Flask endpoint tests for successful jobs, failed jobs,
  cancellation, persisted history reload, interrupted-job reload handling,
  process-group cancellation, configuration snapshots, resource-lock rejection,
  command submission, status polling, log retrieval, and unknown command
  rejection.
- Added `posetestbot.pipeline.stages`, a typed transitional pipeline-stage
  registry that builds validated `uv run python ...` command arrays for
  non-destructive sync, ArUco, BlenderProc prepare/render, BOP export, BOP
  result export, and BOP Toolkit evaluation.
- Added `scripts/run_foundationpose_stage.py` as a manifest-tracked
  FoundationPose stage wrapper. It validates prepared synchronized sensor
  folders, writes `foundationpose_plan.json`, records the `foundationpose`
  stage, and stays dry-run-first unless Docker/runtime execution is requested.
- Added the `foundationpose` stage to the typed pipeline registry with safe
  dry-run defaults and tests for command generation, plan writing, object-ID
  validation, and manifest artifacts.
- Added `posetestbot.estimation.legacy_estimators` plus
  `scripts/run_megapose_stage.py` and `scripts/run_sam6d_stage.py` as
  manifest-tracked dry-run-first adapter scaffolds for the legacy MegaPose and
  SAM6D wrapper paths. They validate synchronized sensor folders, write
  `megapose_plan.json`/`sam6d_plan.json`, record manifest stages, and only
  execute when dry-run is disabled and the configured wrapper script exists.
- Added `megapose` and `sam6d` stages to the typed pipeline registry with safe
  dry-run defaults and tests for plan writing, object-ID validation, missing
  wrapper reporting, and command generation.
- Extended runtime status with MegaPose/SAM6D wrapper-script checks. The checks
  use `MEGAPOSE_WRAPPER`/`SAM6D_WRAPPER` when set and otherwise look for the
  legacy wrapper names under `scripts/`, without starting either estimator.
- Added `megapose_to_bop_eval_dry_run` and `sam6d_to_bop_eval_dry_run`
  sequence templates. They compose BOP export, the dry-run estimator adapter
  stage, BOP result export, and dry-run BOP Toolkit evaluation with caller
  override support for `{run_root}` output paths.
- Exposed `megapose_to_bop_eval_dry_run` and `sam6d_to_bop_eval_dry_run` in the
  transition page run-config sequence dropdown and verified the
  `/pipeline/sequences` API lists them.
- Added artifact-browser discovery and summary support for
  `foundationpose_plan.json`, `megapose_plan.json`, and `sam6d_plan.json` so
  estimator adapter dry-run plans show estimator ID, object ID, sensor names,
  wrapper availability, option keys, and whether the staged command uses
  `uv run`.
- Added Flask `/pipeline/stages`, `/pipeline/stages/<stage_id>`, and
  `/pipeline/run` APIs backed by the local job runner. Pipeline submissions
  snapshot normalized stage options, declare resources, and keep BlenderProc
  render/FoundationPose/BOP evaluation dry-run by default.
- Added tests for pipeline command generation, option validation, repeated
  FoundationPose output options, ArUco result-export options, stage listing,
  and pipeline job submission.
- Added `posetestbot.pipeline.recommendations` plus Flask
  `/pipeline/recommendations` and a compact transition-page Recommended Steps
  panel. The read-only helper inspects current run artifacts and suggests
  safe next `uv run ...` commands/endpoints, expected artifacts, and resources
  for run config, run preflight snapshots, capture planning, fake execution,
  sync quality, ArUco coverage, calibration, BOP export/result export,
  FoundationPose/ArUco/MegaPose/SAM6D result conversion, dry-run BOP
  evaluation, and metric report export.
- Added a run-preflight recommendation when `run_config.json` exists but
  `run_preflight_report.json` is missing, so operators are pointed at the
  persisted readiness snapshot before queueing saved workflows.
- Run-preflight recommendations now also refresh saved reports whose
  `overall_status` is `error` or whose embedded config no longer matches the
  current `run_config.json`, keeping the recommended-step panel aligned with
  the queue-time safety gate.
- The queue-saved-sequence recommendation is now held back until the saved
  `run_preflight_report.json` is fresh for the current `run_config.json`, so
  the Recommended Steps panel presents the readiness snapshot before queueing.
- Tightened estimator result-conversion recommendations so FoundationPose,
  MegaPose, and SAM6D suggestions require converter-ready output signatures
  (`ob_in_cam/`, `megapose_poses.json`, or `detections_pem/`) instead of
  output-folder names alone.
- Added a calibration-preflight recommendation when `run_config.json` declares
  a calibration profile collection, the run root already contains
  `calibration_profiles.json`, or the saved sequence includes a
  `calibration_preflight` stage, so operators are pointed at
  `calibration_preflight_report.json` before calibrated downstream stages.
- Updated pipeline recommendations to recognize calibration target pose outputs
  beyond legacy ArUco. `charuco_pose_estimation.json`,
  `checkerboard_pose_estimation.json`, and
  `calibration_target_pose_estimation.json` now set explicit recommendation
  facts and can suggest `calibration_observations` without requiring the
  ArUco-specific coverage report first.
- Added `posetestbot.pipeline.sequences` for dependency-aware workflow
  composition. Current sequences include `sync_aruco`,
  `sync_to_bop_dry_run`, and `sync_to_bop_calibrated_dry_run`, and plans write
  `pipeline_sequence_plan.json`.
- Added `sync_to_bop_calibrated_dry_run`, a calibrated sync-to-BOP preset that
  runs `calibration_preflight` before BlenderProc/BOP stages and threads
  `{run_root}/calibration_profiles.json` into both calibrated downstream
  stages by default.
- Added `scripts/run_pipeline_sequence.py` to build sequence plans, record
  `pipeline_sequence:<sequence_id>` in `dataset_manifest.json`, and either plan
  only or execute dependency-ordered stage commands.
- `scripts/run_pipeline_sequence.py --sequence` now uses the typed sequence
  registry for argparse choices, matching the run-config CLI, transition UI,
  and `/pipeline/sequences`.
- Added Flask `/pipeline/sequences`, `/pipeline/sequences/<sequence_id>`, and
  `/pipeline/run-sequence` APIs so a configured workflow can be queued as one
  local job with combined resource declarations.
- Plan-only sequence jobs now declare only `disk_io` to the local job runner,
  avoiding unnecessary camera/render/estimator/evaluation resource locks while
  still recording the full planned resource set in `pipeline_sequence_plan.json`
  and preserving both `locked_resources` and `planned_resources` in the job
  parameter snapshot.
- Updated the transition run-config sequence selector to render from
  `posetestbot.pipeline.sequences.PIPELINE_SEQUENCES` instead of a hand-written
  option list, keeping the UI aligned with newly added typed workflows.
- Added Flask `GET /run-config` and `POST /pipeline/run-config` APIs so a
  saved `run_config.json` can be inspected with its derived sequence plan and
  queued without repeating options in the request body.
- Run-config-derived sequence plans now thread a saved `calibration_profiles`
  path into calibrated BlenderProc preparation and BOP export steps when those
  stages are present and no caller override is supplied.
- Added Flask `POST /run-config` and transition page controls for saving,
  loading, and queueing `run_config.json` from the browser.
- Run-config save/load/queue responses and the transition page now include a
  compact saved-preflight queue-readiness summary, reporting missing, invalid,
  failed, stale, or ready `run_preflight_report.json` state before or during
  queue attempts.
- Moved saved-preflight queue-readiness classification into
  `posetestbot.pipeline.preflight`, keeping missing/invalid/failed/stale/ready
  semantics reusable outside the Flask transition layer.
- Updated pipeline recommendations to consume the shared saved-preflight
  queue-readiness helper and expose the queue blocker in recommendation facts,
  keeping recommended next steps aligned with the queue API.
- Added saved-preflight queue-readiness fields to `run_config.json` and
  `run_preflight_report.json` artifact summaries and scan labels, so artifact
  listings show missing/invalid/failed/stale/ready preflight state with the same
  shared vocabulary.
- Added `posetestbot.pipeline.preflight`, `scripts/run_preflight.py`, Flask
  `GET /pipeline/preflight`, and a transition-page Preflight panel for checking
  saved run configs, sequence plans, selected robot mode, live sensor status,
  and external runtime readiness before queueing a sequence.
- Added persisted run preflight snapshots: `scripts/run_preflight.py --write`
  and Flask `POST /pipeline/preflight` now write `run_preflight_report.json`,
  record a manifest `run_preflight` stage, expose a transition-page Write
  Preflight action, and preserve error-status evidence without launching stages.
- Tightened `POST /pipeline/run-config` so the transition queue path rejects a
  missing `run_preflight_report.json` unless the request explicitly sets
  `allow_missing_preflight: true`, rejects a saved report with
  `overall_status: error` unless `allow_failed_preflight: true` is supplied,
  and rejects stale snapshots whose embedded config no longer matches
  `run_config.json` unless `allow_stale_preflight: true` is supplied.
- Invalid saved preflight snapshots are now classified as `invalid_preflight`
  by the shared queue-readiness helper and are rejected until a fresh preflight
  report is written.
- Added default-off transition page Queue Config checkboxes for those
  missing/failed/stale preflight overrides, so operator bypasses are explicit
  in the same UI that shows the saved-preflight readiness summary.
- Run-config preflight sequence-plan checks now include `non_dry_run_steps`
  so installed-runtime workflows expose FoundationPose, MegaPose/SAM6D, or BOP
  Toolkit execution intent before any stage is launched.
- Run-config preflight now adds a `runtime_requirements` check that maps
  non-dry-run steps to specific runtime IDs (`foundationpose`, `megapose`,
  `sam6d`, `blenderproc`, `bop_toolkit`) and reports missing runtimes as
  warnings for plan-only inspection or errors for execution configs.
- Run-config preflight now also reports `calibration_profile_inputs`, resolving
  run-config and per-stage calibration profile paths and warning or erroring on
  missing profile files according to the saved plan-only/execution intent.
- Added compact transition page panels for job/resource status, job
  cancellation/opening, artifact listing, artifact previews, and metric
  dashboard rendering through the existing Flask APIs.
- Added a compact transition page BOP inspector for loading BOP result CSV
  summaries and one-frame RGB/depth/mask/GT/provenance plus pose-row bundles
  from the BOP result/frame endpoints without hand-building URLs.
- Added tests for sequence topological ordering, option merging, sequence job
  command generation, plan JSON output, plan-only manifest updates, and sequence
  web submission, plus run-config creation and run-config-driven sequence
  submission. Web tests also assert the transition page exposes run-config,
  job, and artifact controls.
- Added estimator-to-evaluation sequence presets:
  `foundationpose_to_bop_eval_dry_run` and `aruco_to_bop_eval_dry_run`. These
  compose BOP export, BOP result export, and dry-run BOP Toolkit evaluation
  planning with default `{run_root}` result paths and overrideable per-step
  options.
- Added `foundationpose_runtime_to_bop_eval`, the first installed-runtime
  estimator-to-BOP-evaluation sequence. It composes BOP export, non-dry-run
  FoundationPose execution, FoundationPose BOP19 result export, and non-dry-run
  BOP Toolkit evaluation while preserving explicit option overrides for runtime
  checkout paths and evaluation settings.
- Added `megapose_runtime_to_bop_eval` and `sam6d_runtime_to_bop_eval` as
  installed-runtime sequence templates for the MegaPose and SAM6D adapter
  bridges. They compose BOP export, non-dry-run estimator execution, BOP19
  result export, and non-dry-run BOP Toolkit evaluation while keeping wrapper
  paths and evaluation settings explicit.
- Added `capture_to_bop_foundationpose_dry_run`, a captured-run bridge that
  chains non-destructive sync, BlenderProc preparation, BlenderProc render
  planning, BOP export, and FoundationPose execution planning with external
  runtimes kept in dry-run mode by default.
- Added sequence tests for default BOP result paths, ArUco source options,
  caller override plumbing, calibrated sync-to-BOP ordering,
  capture-to-BOP/FoundationPose ordering, sequence listing, and web API
  exposure.
- Added `posetestbot.io.artifact_browser` to collect manifest-listed artifacts,
  stage outputs, sequence plans, BOP export/result/evaluation references, and
  safe file/directory previews scoped to a run root.
- Added Flask `/artifacts` and `/artifacts/preview` APIs. Artifact listings
  include run artifacts plus log links for jobs whose saved parameters reference
  the run root; file previews reject path escapes outside the run root.
- Extended artifact records with scan-friendly summaries for dataset manifests,
  pipeline sequence plans, BOP export/result/evaluation metadata, BOP scene
  folders, CSV result files, and readable images. Image previews now include a
  small PNG thumbnail.
- Added BOP scene drill-down through `posetestbot.io.artifact_browser`
  `bop_scene_detail` and Flask `/artifacts/bop-scene`, reporting frame-level
  RGB/depth files, camera metadata, GT records, GT info, mask filenames, and
  `posetestbot_bop_frame_map.json` provenance.
- Added Flask `/artifacts/file` for path-safe full artifact file responses under
  a run root, and extended BOP scene detail with run-root-relative RGB/depth and
  mask artifact paths for image/mask viewers.
- Added canonical legacy metric artifact names for `accuracy_HRC-Hub.json`,
  `accuracy_ArUco_HRC-Hub.json`, and `all_results.json`, plus artifact-browser
  discovery and scan-friendly summaries for method counts, motion names,
  `all_motions` metrics, sample counts, combined result groups, and best method
  by `AP_p`.
- Added `metric_dashboard_summary` and Flask `/artifacts/metrics`, returning a
  run-level dashboard contract for discovered legacy metric artifacts, direct
  methods, combined result groups, method names, best result by `AP_p`, and BOP
  Toolkit score rows from `bop_evaluation_report.json`.
- Added a compact transition page metric dashboard for artifact/method counts,
  best `AP_p`, best BOP19 average recall, direct method rows, BOP Toolkit score
  rows, and combined result groups while preserving raw JSON output for
  debugging.
- Added `posetestbot.evaluation.metric_reports` and
  `scripts/run_metric_report_export_stage.py`, exporting discovered legacy
  metric artifacts and BOP Toolkit score rows to
  `results/metrics/metric_report.json`, `metric_methods.csv`, and
  `metric_report.xlsx` with a manifest-tracked `metric_report_export` stage.
- Added BOP19 result CSV drill-down through `bop_result_detail` and Flask
  `/artifacts/bop-result`, validating result files, parsing row-level
  scene/image/object IDs, scores, rotation/translation/time values, and linking
  rows back to BOP scene folders from `bop_export_manifest.json`.
- Added BOP frame drill-down through `bop_frame_detail` and Flask
  `/artifacts/bop-frame`, joining one scene frame with RGB/depth image metadata,
  camera records, GT annotations, masks, frame-map provenance, and optional
  matching BOP19 result rows for side-by-side inspection.
- Added `render_bop_frame_overlay_png`, Flask `/artifacts/bop-frame-overlay`,
  and transition-page overlay rendering for BOP frames. The overlay composites
  masks onto RGB, draws GT boxes, adds BOP19 result score labels, and projects
  BOP19 result translations through `scene_camera.cam_K` as object-origin
  markers. When BOP model PLY files are mapped in `bop_export_manifest.json`,
  result model vertices are also projected through the result pose to draw an
  estimated model bbox while keeping file reads scoped to the run root.
- Added artifact-browser discovery and summary support for `run_config.json` so
  configured robot mode, intended sensor count, and default sequence are visible
  beside manifests and sequence plans.
- Extended artifact-browser summaries for calibrated workflows: `run_config.json`
  now surfaces object folder and calibration profile path presence, while
  `pipeline_sequence_plan.json` shows planned resources plus steps and paths
  that carry calibration profile inputs.
- Extended `calibration_preflight_report.json` artifact summaries with profile
  path, check status counts, matched profile IDs, and quality gate settings so
  warning/error causes are visible without opening the raw report.
- Added artifact API `display_label` scan lines and transition-page rendering
  for them, so calibrated run configs, sequence plans, and calibration
  preflight reports expose sequence IDs, resources, object/calibration inputs,
  status, matched profiles, and check counts directly in the artifact list.
- Added artifact-browser summary support for `run_preflight_report.json`,
  including sequence ID, step count, robot mode, included sensor/runtime
  snapshots, and preflight check status counts.
- Added artifact-browser tests for discovery from manifests/BOP metadata,
  run configs, JSON/text/directory/image previews, typed summaries,
  outside-run-root records, path-escape rejection, BOP scene drill-down, file
  serving, legacy metric summaries, dashboard aggregation, BOP result/frame
  drill-down, and Flask artifact endpoints.
- Expanded `README.md` with uv, hardware, fake iiwa, and capture guidance.
- Added this progress ledger and root `AGENTS.md`.

## In Progress

- Keeping legacy script compatibility while moving reusable behavior into
  importable `posetestbot.*` modules.
- Using metadata sidecars as the bridge from timestamp-named legacy frames to
  the future manifest-backed, non-destructive synchronization pipeline.
- Using `posetestbot.sensors.registry` as the first shared sensor-adapter
  registry so status, capture planning, CLI help, and the transition UI agree
  on supported sensor families and script-backed capture capabilities.
- Using `calibration_preflight_report.json` as the first run-level calibration
  readiness gate. It makes missing, ambiguous, low-quality, or not-yet-valid
  profiles visible before BlenderProc and BOP export rely on them.
- Using `dataset_manifest.json` as the first manifest-backed storage bridge.
- Using `run_config.json` as the first versioned operator-intent bridge between
  UI/API configuration and typed stage/sequence job submission. The transition
  UI/API can now create, inspect, load, validate required object-folder input,
  preserve preflight snapshots, block queueing after failed or stale preflight
  evidence unless explicitly overridden, and queue the configured sequence from
  the saved config.
- Using `capture_plan.json` as the first non-executing capture orchestration
  bridge. It records the exact fake-first robot receiver and sensor capture
  commands from a saved config without opening cameras or sending robot UDP
  commands, and the Flask transition UI/API can now write and inspect it beside
  run-config preflight and queueing controls.
- Using `capture_rehearsal_report.json` as the first executable fake-capture
  bridge. It exercises fake iiwa command/pose receiver startup and artifact
  writing without opening camera SDKs or touching the real iiwa, and now uses
  the capture-plan command model so later camera process supervision can build
  on the same structure.
- Using `capture_plan_preflight_report.json` as the launch gate between capture
  planning and execution. The fake rehearsal sequence now writes a capture plan,
  preflights the command model, writes a `capture_execution_plan.json` command
  selection artifact, then runs the pose-only fake rehearsal. The preflight gate
  now catches adapter/resolution mismatches and output-folder collisions before
  launch planning touches camera SDKs.
- Using `capture_execution_plan.json` as the next bridge from non-executing
  capture planning toward supervised process startup. It records selected
  commands, skipped camera commands, safety gates, selected resources, and the
  planned process-group stop policy without starting hardware.
- Using `capture_execution_report.json` as the first general supervised capture
  execution report. The runner still defaults to the safe fake iiwa pose path,
  but full mode now has mocked coverage for selected sensor process startup and
  supervisor stop reporting, and the transition UI/API now exposes active
  capture job status, latest supervisor status artifacts, and capture-specific
  stop controls through the local job runner.
- Keeping the old destructive sync script available while the full downstream
  pipeline is migrated toward the new derived synchronized folders.
- Treating the BOP export/estimator/evaluation path as an incremental bridge;
  FoundationPose now has a manifest-tracked dry-run stage, and BOP result export
  can convert FoundationPose, ArUco, MegaPose, and SAM6D outputs into BOP19 CSVs.
  MegaPose/SAM6D now have manifest-tracked dry-run adapter scaffolds, but real
  FoundationPose/BOP Toolkit runtime execution, installed MegaPose/SAM6D wrapper
  confirmation, and optional richer dataset metadata still need fuller
  implementation. A captured-run-to-BOP/FoundationPose dry-run sequence ties
  the existing bridge stages together for planning.
- Treating the Flask UI as a transition layer while the local job runner moves
  long-running command execution, typed stage submission, and dependency-aware
  sequence submission toward the baseline webapp architecture. The page now has
  basic robot/sensor/runtime status, run-scoped hardware snapshots, run-config,
  job/resource, artifact, recommended-step, metric, and BOP frame/result controls
  with compact preflight, metric, and BOP inspection panels, but it remains a
  bridge rather than the final
  FastAPI/Jinja2/HTMX UI.
- Treating the artifact browser as the first transition API for surfacing
  manifest-backed outputs, plans, BOP files, evaluation files, and job logs from
  one central UI. It now provides lightweight summaries for run configs,
  sequence plans, estimator plans, sync quality, ArUco coverage, BOP
  export/result/evaluation files, BOP Toolkit score summaries, metric report
  files, and metrics, plus BOP scene drill-down with scoped file serving, BOP
  result/frame drill-down, and a
  dashboard-ready metric summary endpoint. Richer UI-specific views are still
  intentionally
  incremental.

## Next Implementation Steps

- Keep `rewrite_fake_end_to_end.v1` green as the first anti-drift checkpoint.
  When touching capture, sync, BOP, result export, BOP evaluation, or metric
  export contracts, rerun the hardware-free smoke and inspect
  `rewrite_gate_report.json`, then run `scripts/run_rewrite_status.py` to keep
  the aggregate rewrite status honest before adding new UI/reporting surface.
- Run a real full-camera lab capture and make `rewrite_full_capture.v1` pass.
  This is the next boundary after the fake smoke: use real robot mode only
  intentionally, run full capture with `--allow-cameras` and the lab safety
  gates satisfied, then audit the resulting raw sensor folders before treating
  hardware capture as validated.
- Validate real FoundationPose execution through
  `rewrite_foundationpose_runtime.v1` on the installed Docker/runtime checkout,
  then run BOP Toolkit on the converted FoundationPose result CSV and make the
  gate pass with non-dry-run score metrics.
- Extend supervised capture execution from the current fake-safe and mocked
  full-process coverage to validated full `capture_plan.json` camera process
  groups on the lab hardware, including per-process live telemetry and
  operator-facing teardown evidence, while keeping fake iiwa as the default
  development path.
- Confirm installed MegaPose/SAM6D wrappers and runtime outputs on the lab
  machine, then execute the new manifest-tracked adapters against fixture runs
  and feed their outputs through the new sequence templates.
- Implement actual robust ChArUco/ArUco/checkerboard calibration capture and
  real held-out lab validation workflows that make
  `rewrite_calibration_validation.v1` pass with production-ready
  `calibration.v1` profiles, building on the current observation, solver,
  candidate, solver-method comparison, held-out residual, and validation
  contracts.
- Keep richer BOP/COCO/multiview variants, artifact browser views, metric
  charts, and mask/result comparison tools behind the golden-path gate unless a
  downstream consumer need directly unblocks validation.
- Eventually replace the Flask transition bridge with the FastAPI/Jinja2/HTMX
  architecture described in the baseline.

## Open Risks

- The real KUKA Sunrise app still needs a confirmed implementation for the
  structured v1 command protocol and stop behavior.
- The legacy sync script still destructively renames/deletes frames.
- ZED SDK Python runtime availability is missing from the uv environment
  (`pyzed.sl` unavailable on 2026-06-16).
- Current unsandboxed sensor status sees all 3 RealSense cameras, but the
  OAK-D Pro is blocked by DepthAI udev permissions.
- Real hardware capture has not yet been validated after the wrapper and
  metadata changes.
- The full-capture rewrite gate now exists, but no current lab run evidence in
  this workspace proves `rewrite_full_capture.v1` passing. It remains the
  explicit blocker before claiming RealSense/OAK-D/ZED full capture
  orchestration is validated.
- The latest read-only status snapshot in this managed workspace still selects
  fake iiwa mode. Sensor discovery reported 0/3 RealSense, 0/1 OAK-D Pro, and
  0/1 ZED 2i visible through the sandboxed path, with a RealSense udev monitor
  error, a DepthAI USB warning, and missing `pyzed.sl`. Runtime status reported
  BlenderProc, FoundationPose, MegaPose, SAM6D, BOP Toolkit, and ZED SDK Python
  as missing. That keeps the real capture/runtime milestones blocked here until
  they are validated on the lab host with the needed device and runtime access.
- The rewrite now has an explicit fake end-to-end gate and a synthetic RGB-D
  bridge plus synthetic BOP result bridge for fake runs, but no current evidence
  proves real hardware capture or estimator execution. A local hardware-free
  smoke has proven the full fake gate with run config, preflight, supervised
  fake capture, synthetic RGB-D, sync quality, BOP export, synthetic BOP result
  export, dry-run BOP evaluation, metric export, and `rewrite_gate_report.json`.
- `capture_execution_plan.json` now records selected commands and safety gates,
  and `capture_execution_report.json` records supervised fake iiwa plus robot
  pose receiver runs. Full camera process supervision has mocked coverage,
  explicit stop reasons, and per-process PID/timing telemetry. The transition
  UI/API now has capture-job status and a
  capture-specific stop endpoint backed by process-group cancellation, and
  `capture_execution_status.json` provides latest per-process supervisor
  snapshots. Pipeline recommendations and artifact labels now treat missing,
  invalid, empty, or receiver-less `capture_plan.json` files as blockers before
  capture-plan preflight, and failed or invalid `capture_execution_plan.json`
  files as blockers before supervised capture launch. Artifact labels now also
  distinguish fake capture rehearsal reports that actually produced raw poses
  from failed or empty rehearsal reports before operators treat them as sync
  inputs. Run-scoped hardware snapshots now expose ready/blocker labels so
  failed robot/sensor/runtime status reports are visible before capture launch.
  Full mode still needs real hardware validation.
- The capture-plan preflight can flag the current lab's RealSense udev,
  OAK-D Pro visibility, and ZED SDK Python gaps, and now carries those hints on
  the failing sensor checks. Those hardware/runtime issues still need
  system-level resolution before full camera capture can be launched reliably.
  It also blocks unsupported resolution/sensor combinations and duplicate
  planned output folders before command execution. Pipeline recommendations and
  artifact labels now treat failed or invalid
  `capture_plan_preflight_report.json` files as blockers before planning
  capture execution.
- Calibration preflight now verifies profile coverage and recorded quality
  fields, calibration solver output can produce `needs_validation` profiles
  from eye-in-hand OpenCV hand-eye or static target/reference observations,
  optional solver held-out residuals are now recorded, calibration candidates
  can still reject residual-threshold outliers, and calibration validation can
  explicitly promote passing profiles. Pipeline recommendations and artifact
  labels now treat failed or invalid calibration preflight reports as blockers
  before calibrated BlenderProc/BOP recommendations, treat solver reports with
  `overall_status=error` or invalid
  structure as blockers before candidate generation, and candidate reports with
  `overall_status=error` or invalid structure as blockers before validation.
  Validation reports now get the same treatment: failed or invalid
  `calibration_validation_report.json` files keep the validation stage
  recommended instead of being treated as final profile evidence. Artifact
  labels now also distinguish non-empty calibration profile collections that can
  be inspected by validation from invalid or empty profile collections.
  `rewrite_calibration_validation.v1` now defines the minimum promoted-profile
  evidence needed before claiming production calibration is validated.
  PoseTestBot still needs actual robust calibration capture and real held-out
  lab validation before profiles should be considered production quality.
- Calibration observations now provide solver-ready target/robot pose pairs,
  rejected-frame reasons, explicit target metadata, and target-specific
  ChArUco/checkerboard ingestion once detector outputs exist, and both the
  solver and candidate paths now write `needs_validation` profile collections.
  Pipeline recommendations and artifact labels now treat failed or invalid
  ArUco coverage reports as blockers before ArUco-derived observation
  extraction or ArUco BOP result conversion, and treat observation reports with
  `overall_status=error` or invalid structure as blockers before solver and
  candidate generation. Actual robust ChArUco/checkerboard
  detection/capture, profile-quality evidence beyond residual gates, optional
  solver holdout checks, and method-comparison rows are still future work.
- FoundationPose now has a manifest-tracked dry-run/execution bridge, but the
  real Docker/runtime path has not been validated from the new stage wrapper.
  `rewrite_foundationpose_runtime.v1` now defines the minimum evidence needed
  before claiming that runtime path is validated.
  Pipeline recommendations and artifact labels now treat missing, invalid, or
  empty FoundationPose/estimator plan job lists as re-plan blockers instead of
  treating plan-file existence alone as readiness.
- BOP Toolkit integration now has a manifest-tracked dry-run/execution bridge
  with plan/report artifacts plus FoundationPose, ArUco, MegaPose, and SAM6D
  BOP CSV converters, and reports can summarize numeric `scores*.json` metrics
  from completed Toolkit runs. Pipeline recommendations and artifact labels now
  distinguish usable BOP dataset export manifests from missing, invalid, or
  empty export manifests before planning result conversion; require BOP object
  model metadata before recommending estimator-to-BOP result conversion;
  distinguish usable BOP result export manifests from invalid manifests or
  missing result CSVs before planning evaluation; require usable BOP19 target
  rows before recommending or writing BOP Toolkit evaluation plans; report BOP
  model-folder, `models_info.json`, and model-PLY readiness in evaluation
  prerequisite checks; label BOP evaluation plans with structural ready/blocker
  state before operators treat them as executable Toolkit commands; and
  distinguish planned/succeeded BOP evaluation reports from failed, invalid,
  critical-prerequisite-failed, or stale partial-check reports before treating
  them as ready metric evidence. MegaPose/SAM6D adapter scaffolds can write
  dry-run plans, but
  PoseTestBot still needs a confirmed installed BOP Toolkit runtime and
  confirmed MegaPose/SAM6D wrapper/runtime execution.
- The Flask job runner stores/reloads job snapshots and enforces declared
  resource locks, and POSIX cancellation terminates the process group. Sequence
  presets now cover several dependency-aware workflows, and pipeline
  recommendations plus artifact labels now treat missing, invalid, or empty
  `pipeline_sequence_plan.json` files as blockers before considering the saved
  workflow queued/planned. Invalid `run_config.json` files now stop downstream
  recommendation fan-out and are labeled as root config blockers before
  preflight, sequence, or capture planning. The runner is still an in-process
  bridge without the final web stack.
- Artifact browsing now has lightweight BOP scene drill-down, scoped file
  serving, BOP result/frame drill-down, legacy metric summaries, a
  dashboard-ready metric API with BOP Toolkit score rows, compact
  transition-page BOP frame/mask/result inspection and overlays with BOP19
  object-origin markers and model-projected result bboxes, compact
  transition-page metric rendering, sync quality summaries, and CSV/JSON/image
  summaries. Artifact labels now distinguish usable BOP export sidecars from
  empty or invalid `models_info.json`, BOP19 targets, PoseTestBot multiview
  targets, and COCO-style annotation files. Pipeline recommendations and
  artifact labels now distinguish missing, invalid, empty, and dashboard-ready
  metric report exports, and BOP Toolkit evaluation reports count as metric
  sources when usable score metrics exist even if no legacy accuracy JSON
  exists, while failed critical BOP evaluation prerequisites or score-less
  dry-run plans keep those rows out of metric dashboards and exports. Richer
  metric charts, more ergonomic visual comparison tools, and external trusted
  artifact handling are still future work.
- Sync quality now produces non-destructive aggregate evidence, but its
  thresholds are still conservative defaults and should be tuned against real
  multi-sensor capture runs once hardware capture validation resumes. Pipeline
  recommendations and artifact labels now treat `overall_status=error` as a
  downstream blocker before ArUco while allowing `ok` and `warning` reports to
  proceed.

## Latest Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_preflight.py /tmp/posetestbot_gate_full_smoke_real_full_capture --check --write`:
  wrote `run_preflight_report.json` with `overall_status=warning`; this
  workspace selected the fake robot profile, saw 0 connected sensors, and had
  0/6 external runtimes available, while the sequence itself had no non-dry-run
  runtime requirements.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_hardware_status_stage.py /tmp/posetestbot_gate_full_smoke_real_full_capture`:
  wrote `hardware_status_report.json` with `overall_status=error`, confirming
  the current workspace cannot claim lab hardware/runtime readiness. The
  run-scoped hardware snapshot now selects the real robot profile from the
  saved `run_config.json`, so an extra `POSETESTBOT_ROBOT_MODE=real` wrapper is
  no longer needed for this real full-capture evidence root.
- `env POSETESTBOT_ROBOT_MODE=real UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_preflight.py /tmp/posetestbot_gate_full_smoke_real_full_capture --check --write`:
  refreshed `run_preflight_report.json` with configured real mode matching the
  selected real profile, plus nested sensor diagnostics for the camera
  visibility blockers.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_stage.py /tmp/posetestbot_gate_full_smoke_real_full_capture`:
  wrote `capture_plan.json` with five sensor capture commands plus the real
  robot pose receiver command targeting `172.31.1.147:30300` and receiver
  `172.31.1.169:8080`; no commands were launched.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_preflight.py /tmp/posetestbot_gate_full_smoke_real_full_capture --allow-real-robot`:
  wrote `capture_plan_preflight_report.json` with `overall_status=error` in
  this workspace, preserving the safety blocker before real capture execution.
  The failed sensor checks now carry the same structured diagnostics and hints
  as the nested sensor-status snapshot, including RealSense USB/udev access,
  expected OAK-D Pro visibility, and ZED SDK Python availability.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_execution_plan.py /tmp/posetestbot_gate_full_smoke_real_full_capture --mode full --allow-cameras --allow-real-robot --include-sensors`:
  wrote `capture_execution_plan.json` with status `error`, selected the six
  full-capture commands, and kept `ready_to_execute=false`; no robot or camera
  process was started.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_gate.py /tmp/posetestbot_gate_full_smoke_real_full_capture --gate rewrite_full_capture.v1 --write`:
  reported `rewrite_full_capture.v1` blocked with 3/12 checks ready. Ready
  evidence is the real-mode run config, warning-level run preflight, and capture
  plan; blockers are sensor-readiness-backed hardware status, capture-plan
  preflight, execution-plan readiness, the missing supervised capture report,
  and missing raw RGB-D frame folders.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_status.py /tmp/posetestbot_rewrite_status --write --gate-run-root rewrite_fake_end_to_end.v1=/tmp/posetestbot_gate_full_smoke --gate-run-root rewrite_full_capture.v1=/tmp/posetestbot_gate_full_smoke_real_full_capture`:
  reported blocked rewrite status with 1/4 gates ready and 12/26 checks ready;
  because the existing hardware snapshot has sensor discovery errors, the next
  action is now the read-only sensor diagnostic
  `uv run python scripts/sensor_status.py --json`. The default text output now
  also prints the first hardware and capture-plan preflight blocker details,
  including the RealSense udev monitor error and USB/udev hints.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... collect_run_artifacts('/tmp/posetestbot_rewrite_status') ... PY`:
  confirmed the live `rewrite_status_report.json` artifact summary includes
  `next_blocker_diagnostics`, `next_blocker_hints`, and `next_blocker_checks`
  for the current RealSense/OAK-D/ZED blockers, and its display label includes
  `next_blocker=hardware_status` plus the first RealSense diagnostic.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' ... build_pipeline_recommendations('/tmp/posetestbot_rewrite_status') ... PY`:
  confirmed the live recommendation payload now includes
  `follow_rewrite_status_next_action` with label `Inspect sensor status`,
  command `uv run python scripts/sensor_status.py --json`, and the RealSense
  discovery diagnostic in the reason.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_plan_preflight.py tests/test_sensor_status.py`:
  11 passed, including capture-plan preflight propagation of sensor diagnostic
  hints into the failing per-sensor checks.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_plan_preflight.py tests/test_sensor_status.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  246 passed, confirming the diagnostic propagation, rewrite gates, CLI
  blocker output, recommendations, artifact summaries, typed stages, sequence
  contracts, and transition web API still agree.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json`:
  reported 0/5 expected cameras visible in this workspace; RealSense discovery
  failed with `RuntimeError: could not initialize udev monitor`, OAK-D Pro had
  0/1 connected, and ZED 2i SDK Python was unavailable. The JSON now includes
  per-family diagnostics and hints for USB/udev access, expected device counts,
  and `pyzed.sl` installation.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py`:
  4 passed, including udev/USB and SDK-unavailable diagnostic coverage.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py`:
  26 passed, including next-action coverage that skips redundant full-capture
  sequence planning when `pipeline_sequence_plan.json` already exists.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_gate_full_smoke_real_full_capture --robot-mode real --sequence real_full_capture_validation --print-sequence-plan`:
  wrote the separated real-mode `run_config.json` and printed the guarded
  seven-step real full-capture validation plan without starting hardware.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_pipeline_sequence.py /tmp/posetestbot_gate_full_smoke_real_full_capture --sequence real_full_capture_validation --plan-only`:
  wrote `/tmp/posetestbot_gate_full_smoke_real_full_capture/pipeline_sequence_plan.json`
  with the checked run preflight, hardware snapshot, capture plan, real
  preflight, full execution plan, full supervised capture, and full-capture gate
  audit steps; it did not start hardware.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py`:
  25 passed, including default full-capture evidence-root separation after the
  fake gate is ready and CLI output that points real run-config creation at the
  sibling `<status-root>_real_full_capture` folder instead of the fake-smoke
  proof root.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  94 passed, confirming recommendation stale-status handling and artifact
  summaries still accept the updated `gate_run_roots` contract.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_status.py /tmp/posetestbot_gate_full_smoke --write`:
  reported blocked rewrite status with 1/4 gates ready and pointed the next
  action at
  `/tmp/posetestbot_gate_full_smoke_real_full_capture` for real run-config
  creation, preserving the fake-smoke proof root.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_stages.py tests/test_pipeline_sequences.py`:
  54 passed, including the typed `run_preflight` stage and the updated
  seven-step `real_full_capture_validation` sequence with checked run
  preflight, hardware snapshot, full capture planning/execution, and
  `rewrite_full_capture.v1` audit.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_pipeline_sequence.py /tmp/posetestbot_real_full_capture_config_smoke --sequence real_full_capture_validation --plan-only`:
  wrote a seven-step plan-only sequence with `run_preflight --check --write`,
  `hardware_status`, capture plan, real capture-plan preflight, full execution
  plan, full supervised capture, and the `rewrite_full_capture.v1` audit; it
  did not start hardware.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py`:
  23 passed, including aggregate `next_gate`/`next_actions` coverage and CLI
  next-action output.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_rewrite_gate.py`:
  50 passed, including `real_full_capture_validation` sequence coverage and
  aggregate status actions pointing to that sequence.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_run_config.py tests/test_pipeline_sequences.py`:
  60 passed, including the full-capture status action creating a real run config
  with `--sequence real_full_capture_validation` and CLI coverage that this
  config prints the real capture validation plan.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py`:
  37 passed, including `rewrite_status_report.json` summary coverage for
  `next_gate_id`, first next-action command, and display label.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  117 passed, including per-gate run-root rewrite status aggregation, stale
  status detection that compares `gate_run_roots`, and artifact summaries with
  `next_gate_run_root`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  56 passed, including recommendation handling for current mixed-root
  `rewrite_status_report.json` files.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_stages.py tests/test_pipeline_recommendations.py`:
  83 passed, including repeated `--gate-run-root` command generation for the
  typed `rewrite_status` stage and stale mixed-root refresh recommendations
  that preserve saved gate roots.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "...build_pipeline_recommendations('/tmp/posetestbot_mixed_status_smoke')..."`:
  confirmed the live mixed-root status report is ready for inspection, retains
  the fake/full gate roots, and does not recommend `write_rewrite_status`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_status.py /tmp/posetestbot_mixed_status_smoke --write --gate-run-root rewrite_fake_end_to_end.v1=/tmp/posetestbot_gate_full_smoke --gate-run-root rewrite_full_capture.v1=/tmp/posetestbot_real_full_capture_config_smoke`:
  wrote a mixed-root status report that kept fake evidence in
  `/tmp/posetestbot_gate_full_smoke`, evaluated full-capture blockers against
  `/tmp/posetestbot_real_full_capture_config_smoke`, and pointed the next action
  at the real validation sequence plan for that real-config run.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "...collect_run_artifacts('/tmp/posetestbot_gate_full_smoke')..."`:
  confirmed the actual fake-smoke `rewrite_status_report.json` summary exposes
  `next_gate_id=rewrite_full_capture.v1` and first next action
  `uv run python scripts/create_run_config.py /tmp/posetestbot_gate_full_smoke --robot-mode real --sequence real_full_capture_validation --print-sequence-plan`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_real_full_capture_config_smoke --robot-mode real --sequence real_full_capture_validation --print-sequence-plan`:
  passed; wrote a real-mode `run_config.json` whose saved sequence is
  `real_full_capture_validation` with plan-only true and explicit
  full-capture safety gates in the printed sequence plan.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_pipeline_sequence.py /tmp/posetestbot_gate_full_smoke --sequence real_full_capture_validation --plan-only`:
  wrote the earlier five-step plan-only sequence with capture plan, real
  preflight, full execution plan, full supervised capture, and
  `rewrite_full_capture.v1` audit commands; it did not start hardware. This has
  since been tightened to the seven-step validation sequence above.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_pipeline_stages.py`:
  118 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_status.py /tmp/posetestbot_gate_full_smoke --write`:
  reported blocked rewrite status with 1/4 gates ready, 10/17 checks ready, and
  next action `uv run python scripts/create_run_config.py /tmp/posetestbot_gate_full_smoke --robot-mode real --sequence real_full_capture_validation --print-sequence-plan`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  55 passed, including missing, stale, and current `rewrite_status_report.json`
  recommendation coverage.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_stages.py tests/test_artifact_browser.py`:
  86 passed, including aggregate `rewrite_status_report.json` generation, typed
  `rewrite_status` stage command building, and artifact-browser summary labels.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py posetestbot/pipeline/stages.py posetestbot/io/artifact_browser.py posetestbot/io/artifacts.py scripts/run_rewrite_status.py tests/test_rewrite_gate.py tests/test_pipeline_stages.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  52 passed, including calibration validation gate recommendation coverage for
  an unpromoted validation report.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_stages.py`:
  45 passed, including `rewrite_calibration_validation.v1` ready/unpromoted/
  needs-validation blocker coverage and typed-stage gate choices.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py scripts/run_rewrite_gate.py posetestbot/pipeline/stages.py tests/test_rewrite_gate.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  51 passed, including FoundationPose runtime gate recommendation coverage for
  dry-run BOP evaluation artifacts.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_stages.py`:
  41 passed, including `rewrite_foundationpose_runtime.v1` ready/dry-run blocker
  coverage and typed-stage gate choices.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py scripts/run_rewrite_gate.py posetestbot/pipeline/stages.py tests/test_rewrite_gate.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py`:
  12 passed, including `rewrite_full_capture.v1` blocking mismatched RGB/depth
  raw sensor frame counts.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py tests/test_rewrite_gate.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py`:
  36 passed, including `rewrite_gate_report.json` artifact summary coverage.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_stages.py tests/test_pipeline_recommendations.py`:
  75 passed, including the typed `rewrite_gate` stage and real-mode full-capture
  recommendation coverage.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/stages.py posetestbot/pipeline/recommendations.py tests/test_pipeline_stages.py tests/test_pipeline_recommendations.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/robot_status.py`:
  selected fake mode with command target `127.0.0.1:30300` and receiver bind
  `127.0.0.1:8080`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py`:
  read-only discovery completed but reported 0/3 RealSense, 0/1 OAK-D Pro, and
  0/1 ZED 2i visible in this managed workspace; RealSense discovery hit a udev
  monitor error and DepthAI reported USB protocol unavailable.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/runtime_status.py`:
  reported BlenderProc, FoundationPose, MegaPose, SAM6D, BOP Toolkit, and ZED
  SDK Python as missing.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_fake_e2e_smoke.py /tmp/posetestbot_gate_wrapper_smoke --overwrite`:
  passed with local UDP socket permission; wrote `rewrite_gate_report.json` with
  `rewrite_fake_end_to_end.v1: ready (9/9 ready)`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py`:
  11 passed, including `rewrite_full_capture.v1` ready/blocker coverage.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_synthetic_rgbd.py tests/test_synthetic_bop_results.py tests/test_bop_evaluation_stage.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py`:
  72 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall scripts/run_rewrite_fake_e2e_smoke.py posetestbot/pipeline/rewrite_gate.py posetestbot/pipeline/synthetic_rgbd.py posetestbot/evaluation/synthetic_bop_results.py tests/test_rewrite_gate.py tests/test_synthetic_rgbd.py tests/test_synthetic_bop_results.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_metric_report_export.py`:
  88 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_metric_report_export.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_metric_report_export.py`:
  52 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_metric_report_export.py`:
  39 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  79 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_evaluation_stage.py tests/test_bop_export_stage.py`:
  14 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/evaluation/bop_toolkit.py tests/test_bop_evaluation_stage.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_evaluation_stage.py tests/test_pipeline_recommendations.py`:
  50 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/evaluation/bop_toolkit.py tests/test_bop_evaluation_stage.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_bop_result_export_stage.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_bop_evaluation_stage.py`:
  47 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_bop_export_stage.py`:
  40 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_bop_evaluation_stage.py`:
  34 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_calibration_candidates.py tests/test_calibration_solver.py tests/test_calibration_validation.py`:
  46 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py posetestbot/calibration/profiles.py tests/test_artifact_browser.py tests/test_calibration_candidates.py tests/test_calibration_solver.py tests/test_calibration_validation.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_hardware_status.py`:
  31 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py posetestbot/pipeline/hardware_status.py tests/test_artifact_browser.py tests/test_hardware_status.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_capture_rehearsal.py`:
  30 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py tests/test_capture_rehearsal.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_run_config.py`:
  79 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py posetestbot/pipeline/run_config.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_run_config.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_pipeline_sequences.py`:
  92 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py posetestbot/pipeline/sequences.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_pipeline_sequences.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_plan.py tests/test_capture_plan_preflight.py`:
  74 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_plan.py tests/test_capture_plan_preflight.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_foundationpose_stage.py`:
  65 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_foundationpose_stage.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_execution.py`:
  67 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_execution.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_plan_preflight.py`:
  65 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_capture_plan_preflight.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_calibration_preflight.py`:
  63 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_calibration_preflight.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_aruco_coverage.py`:
  59 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  35 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_aruco_coverage.py`:
  58 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_aruco_coverage.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  52 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_calibration_validation.py`:
  5 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_calibration_validation.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_metric_report_export.py`:
  55 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  50 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_metric_report_export.py`:
  5 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_metric_report_export.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  47 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  45 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  45 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  43 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  40 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  37 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  35 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  32 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  29 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  19 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  28 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py posetestbot/pipeline/recommendations.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  27 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py`:
  10 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  26 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  18 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  63 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_web_interface.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py`:
  91 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/preflight.py posetestbot/pipeline/recommendations.py posetestbot/io/artifact_browser.py web_interface.py tests/test_preflight.py tests/test_web_interface.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  61 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_preflight.py tests/test_web_interface.py`:
  79 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py posetestbot/pipeline/preflight.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_web_interface.py tests/test_run_config.py`:
  74 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/preflight.py web_interface.py tests/test_preflight.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_preflight.py tests/test_run_config.py`:
  71 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall web_interface.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_preflight.py tests/test_run_config.py`:
  71 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall web_interface.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_preflight.py tests/test_run_config.py`:
  71 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall web_interface.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_preflight.py tests/test_run_config.py`:
  70 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall web_interface.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_preflight.py tests/test_run_config.py`:
  68 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall web_interface.py tests/test_web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  64 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  63 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/preflight.py posetestbot/io/artifact_browser.py scripts/run_preflight.py web_interface.py tests/test_preflight.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_run_preflight_write_smoke --sensor realsense:123 --sequence sync_aruco`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_preflight.py /tmp/posetestbot_run_preflight_write_smoke --no-sensors --no-runtimes --write --json`:
  passed, wrote `run_preflight_report.json`, and reported `overall_status:
  ok`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import json, pathlib; root=pathlib.Path('/tmp/posetestbot_run_preflight_write_smoke'); report=json.loads((root/'run_preflight_report.json').read_text()); manifest=json.loads((root/'dataset_manifest.json').read_text()); stage=next(s for s in manifest['stages'] if s['name']=='run_preflight'); print(report['overall_status'], stage['status'], stage['artifacts']['run_preflight_report.json'])"`:
  passed and printed `ok succeeded run_preflight_report.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  55 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_run_config.py tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  86 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/preflight.py posetestbot/pipeline/run_config.py tests/test_preflight.py tests/test_run_config.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  69 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/sequences.py tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  54 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_run_config.py tests/test_web_interface.py`:
  59 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/preflight.py tests/test_preflight.py`:
  passed.
- `git diff --check -- posetestbot/pipeline/preflight.py tests/test_preflight.py README.md docs/REWRITE_PROGRESS.md`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_calibration_observations.py tests/test_pipeline_stages.py tests/test_artifact_browser.py tests/test_web_interface.py tests/test_manifest.py`:
  89 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifacts.py posetestbot/calibration/observations.py scripts/run_calibration_observations.py posetestbot/pipeline/stages.py tests/test_calibration_observations.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_observations.py --help`:
  passed and describes synchronized target-pose outputs.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  passed.
- `git diff --check -- posetestbot/io/artifact_browser.py tests/test_artifact_browser.py tests/test_web_interface.py README.md docs/REWRITE_PROGRESS.md`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_export_stage.py tests/test_pipeline_stages.py tests/test_artifact_browser.py`:
  40 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifacts.py posetestbot/bop/writer.py scripts/run_bop_export_stage.py posetestbot/pipeline/stages.py posetestbot/io/artifact_browser.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_bop_export_stage.py --help`:
  passed and listed `--write-coco-annotations`.
- `git diff --check -- posetestbot/io/artifacts.py posetestbot/bop/writer.py scripts/run_bop_export_stage.py posetestbot/pipeline/stages.py posetestbot/io/artifact_browser.py tests/test_bop_export_stage.py tests/test_pipeline_stages.py tests/test_artifact_browser.py README.md docs/REWRITE_PROGRESS.md`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/io/artifact_browser.py`:
  passed.
- `git diff --check -- posetestbot/io/artifact_browser.py tests/test_artifact_browser.py tests/test_web_interface.py README.md docs/REWRITE_PROGRESS.md`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_export_stage.py tests/test_pipeline_stages.py tests/test_artifact_browser.py`:
  38 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/bop/writer.py scripts/run_bop_export_stage.py posetestbot/io/artifact_browser.py posetestbot/pipeline/stages.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_bop_export_stage.py --help`:
  passed and listed `--write-multiview-targets`.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_calibration_observations.py tests/test_pipeline_stages.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  80 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/calibration/targets.py posetestbot/calibration/observations.py scripts/run_calibration_observations.py web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_observations.py --help`:
  passed and listed calibration target metadata options.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_execution.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  59 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/capture_execution.py posetestbot/io/artifact_browser.py`:
  passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  68 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from pathlib import Path; from posetestbot.pipeline.sequences import build_sequence_job; job=build_sequence_job(sequence_id='capture_to_bop_foundationpose_dry_run', run_root=Path('/tmp/posetestbot_resource_smoke'), plan_only=True); print(job.resources); print(job.parameters['locked_resources']); print(job.parameters['planned_resources'])"`:
  printed `['disk_io']`, `['disk_io']`, and
  `['cpu', 'disk_io', 'estimator', 'render']`.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_run_config.py tests/test_web_interface.py`:
  76 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  68 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_run_config.py tests/test_web_interface.py`:
  57 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py`:
  4 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_run_config.py tests/test_web_interface.py`:
  75 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py`:
  22 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_run_config.py tests/test_web_interface.py`:
  53 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_run_config.py`:
  8 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  65 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py`:
  45 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  50 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  6 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  49 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  5 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  64 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  62 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_metric_report_export.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  57 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 234 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_web_interface.py`:
  51 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 221 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_artifact_browser.py`:
  50 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 220 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_hardware_status.py tests/test_artifact_browser.py tests/test_pipeline_stages.py tests/test_web_interface.py`:
  73 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 220 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_pipeline_stages.py`:
  62 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 216 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_plan_preflight.py tests/test_capture_plan.py tests/test_web_interface.py`:
  50 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 215 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_execution.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  54 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 212 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_job_runner.py`:
  46 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 210 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_run_config.py tests/test_web_interface.py`:
  43 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 208 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_aruco_coverage.py tests/test_manifest.py tests/test_pipeline_stages.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  39 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 207 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_metric_report_export.py tests/test_manifest.py tests/test_pipeline_stages.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  39 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 203 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  40 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 199 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py`:
  7 passed, including estimator plan artifact discovery and summaries.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 195 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_legacy_estimator_stages.py tests/test_pipeline_stages.py`:
  23 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_pipeline_stages.py`:
  36 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runtime_status.py tests/test_web_interface.py`:
  38 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_pipeline_sequences.py`:
  52 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 195 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/runtime_status.py --json`:
  passed and reported `megapose`/`sam6d` wrapper checks as missing because
  `scripts/megapose_wrapper.py` and `scripts/sam6d_wrapper.py` are not present.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 195 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/runtime_status.py --help`:
  passed and listed BlenderProc, FoundationPose, MegaPose, SAM6D, BOP Toolkit,
  and ZED SDK readiness.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_pipeline_sequence.py --help`:
  passed and listed the FoundationPose, ArUco, MegaPose, and SAM6D BOP eval
  dry-run sequence IDs.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 193 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_megapose_stage.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_sam6d_stage.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_evaluation_stage.py tests/test_artifact_browser.py`:
  9 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 189 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_bop_evaluation_stage.py --help`:
  passed and listed the BOP Toolkit plan/report stage options.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_bop_result_export_stage.py tests/test_pipeline_stages.py`:
  27 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 189 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_bop_result_export_stage.py --help`:
  passed and listed `foundationpose`, `aruco`, `megapose`, and `sam6d` sources.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_execution.py tests/test_artifact_browser.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  81 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 186 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_execution_stage.py --help`:
  passed and listed full-mode camera gating plus supervision timeout flags.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_calibration_validation.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_artifact_browser.py tests/test_manifest.py tests/test_web_interface.py`:
  84 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_validation.py --help`:
  passed and listed validation/promotion flags.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; from posetestbot.pipeline.sequences import build_sequence_plan; routes=str(web_interface.app.url_map); stages=web_interface.app.test_client().get('/pipeline/stages').get_json()['stages']; stage=next(s for s in stages if s['id']=='calibration_validation'); plan=build_sequence_plan(sequence_id='sync_aruco_calibration_validation', run_root='/tmp/posetestbot_validation_smoke', options={'calibration_validation': {'min_inliers': 4}}); print('/calibration/validation' in routes, [p['name'] for p in stage['parameters']], [step.id for step in plan.steps], plan.steps[-1].command[-2:])"`:
  passed and printed
  `True ['candidates', 'profiles', 'min_inliers', 'max_mean_translation_residual_mm', 'max_mean_rotation_residual_deg', 'max_outlier_ratio', 'promote', 'output_profiles', 'operator', 'json'] ['sync_run', 'sync_quality', 'aruco', 'calibration_observations', 'calibration_candidates', 'calibration_validation'] ['--max-outlier-ratio', '0.25']`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_candidates.py --help`:
  passed and listed residual-threshold flags.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; from posetestbot.pipeline.sequences import build_sequence_plan; routes=str(web_interface.app.url_map); stages=web_interface.app.test_client().get('/pipeline/stages').get_json()['stages']; stage=next(s for s in stages if s['id']=='calibration_candidates'); plan=build_sequence_plan(sequence_id='sync_aruco_calibration_candidates', run_root='/tmp/posetestbot_candidate_smoke', options={'calibration_candidates': {'max_translation_residual_mm': 25, 'max_rotation_residual_deg': 8}}); print('/calibration/candidates' in routes, [p['name'] for p in stage['parameters']], plan.steps[-1].command[-4:])"`:
  passed and printed
  `True ['observations', 'min_observations', 'target_to_reference', 'max_translation_residual_mm', 'max_rotation_residual_deg', 'no_residual_thresholds', 'json'] ['--max-translation-residual-mm', '25.0', '--max-rotation-residual-deg', '8.0']`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_observations.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 163 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_sync_quality.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; from posetestbot.pipeline.sequences import build_sequence_plan; routes=str(web_interface.app.url_map); stages=web_interface.app.test_client().get('/pipeline/stages').get_json()['stages']; plan=build_sequence_plan(sequence_id='sync_aruco', run_root='/tmp/posetestbot_sync_quality_smoke'); print('/sync/quality' in routes, any(stage['id']=='sync_quality' for stage in stages), [step.id for step in plan.steps])"`:
  passed and printed `True True ['sync_run', 'sync_quality', 'aruco']`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync_quality.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_artifact_browser.py tests/test_manifest.py tests/test_web_interface.py tests/test_run_config.py`:
  81 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall scripts/run_sync_quality.py posetestbot/sync/quality.py posetestbot/pipeline/stages.py posetestbot/pipeline/sequences.py posetestbot/io/artifact_browser.py web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 156 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_calibration_preflight.py tests/test_calibration_profiles.py tests/test_pipeline_stages.py tests/test_artifact_browser.py tests/test_web_interface.py tests/test_manifest.py`:
  69 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_calibration_preflight_smoke --sensor realsense:123 --sequence sync_aruco`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_calibration_preflight.py /tmp/posetestbot_calibration_preflight_smoke --json`:
  passed, wrote `calibration_preflight_report.json`, and reported warning-only
  status because no profile collection was configured.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; client=web_interface.app.test_client(); payload=client.post('/calibration/preflight', json={'run_root': '/tmp/posetestbot_calibration_preflight_smoke'}).get_json(); print(payload['report']['overall_status'], payload['report']['matched_sensor_count'], payload['report']['sensor_count'])"`:
  passed and printed `warning 0 1`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 148 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_registry.py tests/test_sensor_status.py tests/test_capture_plan.py tests/test_web_interface.py`:
  41 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_adapters.py --json`:
  passed and listed OAK-D Pro, RealSense D435, and ZED 2i adapter capabilities.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; payload=web_interface.app.test_client().get('/sensors/adapters').get_json(); print(len(payload['adapters']), [a['sensor_type'] for a in payload['adapters']])"`:
  passed and printed `3 ['oak_d_pro', 'realsense_d435', 'zed_2i']`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 142 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_execution.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_artifact_browser.py tests/test_web_interface.py tests/test_manifest.py`:
  71 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_capture_execution_supervisor_smoke --sensor realsense:123 --sequence fake_capture_execution`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python` one-liner writing a short fake
  `capture_plan.json` for `/tmp/posetestbot_capture_execution_supervisor_smoke`
  on localhost UDP ports `31411/31412`: passed.
- Unsandboxed `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_execution_stage.py /tmp/posetestbot_capture_execution_supervisor_smoke --mode pose_only_fake --timeout-s 8 --startup-wait 0.2 --terminate-timeout-s 2`:
  passed, wrote `capture_execution_report.json`, and captured 3 fake robot pose
  packets.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python` report inspection for
  `/tmp/posetestbot_capture_execution_supervisor_smoke`: passed and confirmed
  `succeeded 3`, successful `robot_controller` and `robot_pose_receiver`
  records, plus `00_fake_iiwa_controller.log` and
  `01_pose_receiver_udp_json.log`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest`: 139 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_execution.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_artifact_browser.py tests/test_web_interface.py tests/test_manifest.py`:
  68 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_execution_plan.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_capture_execution_smoke --sensor realsense:123 --sequence fake_capture_rehearsal`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_execution_plan.py /tmp/posetestbot_capture_execution_smoke --mode pose_only_fake`:
  passed, wrote `capture_execution_plan.json`, selected 2 fake pose commands,
  and skipped 1 camera command.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_plan_preflight.py tests/test_pipeline_stages.py tests/test_pipeline_sequences.py tests/test_web_interface.py tests/test_artifact_browser.py`:
  60 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_preflight.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_capture_plan_preflight_smoke --sensor realsense:123 --sequence fake_capture_rehearsal`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_preflight.py /tmp/posetestbot_capture_plan_preflight_smoke --no-sensors --json`:
  passed, wrote `capture_plan_preflight_report.json`, created/validated
  `capture_plan.json`, and reported warning-only status because sensor discovery
  was intentionally skipped.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python` smoke against Flask
  `POST /capture-plan/preflight`: passed, returning `201 warning`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_rehearsal.py tests/test_capture_plan.py tests/test_pipeline_sequences.py tests/test_pipeline_stages.py tests/test_web_interface.py tests/test_artifact_browser.py`:
  59 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.pipeline.sequences import build_sequence_plan; plan = build_sequence_plan(sequence_id='fake_capture_rehearsal', run_root='/tmp/posetestbot_fake_sequence_smoke', options={'capture_rehearsal': {'duration_s': 0.1, 'sample_ms': 20}}); print(plan.sequence_id, [step.id for step in plan.steps], plan.resources)"`:
  passed before `capture_plan_preflight` and `capture_execution_plan` were
  added to the `fake_capture_rehearsal` sequence.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python` report inspection for
  `/tmp/posetestbot_capture_rehearsal_smoke/capture_rehearsal_report.json`:
  passed and confirmed `succeeded 3` plus embedded `capture_plan.v1` command
  roles `robot_controller`, `sensor_capture`, and `robot_pose_receiver`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_rehearsal.py tests/test_pipeline_stages.py tests/test_web_interface.py tests/test_artifact_browser.py`:
  47 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_rehearsal.py tests/test_capture_plan.py tests/test_pipeline_stages.py tests/test_web_interface.py tests/test_artifact_browser.py`:
  49 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_rehearsal_stage.py --help`:
  passed.
- Unsandboxed `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_rehearsal_stage.py /tmp/posetestbot_capture_rehearsal_smoke --duration 0.06 --sample-ms 30 --startup-delay 0 --robot-port 31301 --receiver-port 31302 --timeout-s 8 --controller-startup-wait 0.2`:
  passed, wrote `capture_rehearsal_report.json`, and captured 3 fake robot pose
  packets without camera hardware.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/pipeline/stages' in str(web_interface.app.url_map)); stages = web_interface.app.test_client().get('/pipeline/stages').get_json()['stages']; print(any(stage['id'] == 'capture_rehearsal' for stage in stages)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('queueCaptureRehearsal()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_capture_plan.py tests/test_web_interface.py tests/test_artifact_browser.py tests/test_pipeline_stages.py`:
  46 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/capture-plan' in str(web_interface.app.url_map)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('capturePlanPanel' in html, 'writeCapturePlan()' in html, 'loadCapturePlan()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python` smoke against Flask
  `POST /capture-plan` and `GET /capture-plan`: passed, returning
  `201 realsense_123` and `200 capture_plan.v1`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_stage.py --help`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_capture_plan_smoke --sensor realsense:123 --sequence sync_aruco`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_capture_plan_stage.py /tmp/posetestbot_capture_plan_smoke --max-frames 1`:
  passed and wrote `capture_plan.json` with fake iiwa, RealSense capture, and
  pose receiver commands.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/pipeline/stages' in str(web_interface.app.url_map)); stages = web_interface.app.test_client().get('/pipeline/stages').get_json()['stages']; print(any(stage['id'] == 'capture_plan' for stage in stages))"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.pipeline.capture_plan import build_capture_plan; print('capture plan import ok')"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_preflight.py tests/test_run_config.py tests/test_web_interface.py`:
  34 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_robot_status.py tests/test_robot_contracts.py tests/test_web_interface.py`:
  32 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runtime_status.py tests/test_web_interface.py`:
  25 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_web_interface.py`:
  25 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_sequences.py tests/test_web_interface.py`:
  29 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_artifact_browser.py`:
  27 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_job_runner.py tests/test_artifact_browser.py`:
  34 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_run_config.py tests/test_web_interface.py`:
  25 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_run_config.py tests/test_manifest.py tests/test_artifact_browser.py`:
  16 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_foundationpose_stage.py tests/test_pipeline_stages.py`:
  10 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_frame_writer.py tests/test_non_destructive_sync.py tests/test_manifest.py`:
  12 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot scripts iiwa start_iiwa.py stop_iiwa.py main.py web_interface.py tests/test_preflight.py tests/test_web_interface.py`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv lock --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_foundationpose_stage.py --help`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_run_config_smoke --sensor realsense:123 --sequence sync_aruco --print-sequence-plan`:
  passed and wrote `run_config.json` plus a derived `sync_aruco` plan.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_run_config_ui_smoke --sensor realsense:123 --sequence sync_aruco --print-sequence-plan`:
  passed and wrote `run_config.json` plus a derived `sync_aruco` plan.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py --help`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.pipeline.run_config import build_sequence_job_from_run_config, load_run_config_for_run_root; print('run config queue helpers import ok')"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/run-config' in str(web_interface.app.url_map), '/pipeline/run-config' in str(web_interface.app.url_map))"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; html = web_interface.app.test_client().get('/').get_data(as_text=True); print('jobsPanel' in html, 'artifactsPanel' in html, 'refreshJobs()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; html = web_interface.app.test_client().get('/').get_data(as_text=True); print('bopInspectorPanel' in html, 'loadBopFrame()' in html, 'loadBopResult()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; html = web_interface.app.test_client().get('/').get_data(as_text=True); print('metricsPanel' in html, 'renderMetrics(' in html, 'bopInspectorPanel' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/sensors/status' in str(web_interface.app.url_map)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('sensorStatusPanel' in html, 'loadSensorStatus()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/runtime/status' in str(web_interface.app.url_map)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('runtimeStatusPanel' in html, 'loadRuntimeStatus()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/robot/status' in str(web_interface.app.url_map)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('robotStatusPanel' in html, 'loadRobotStatus()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import web_interface; print('/pipeline/preflight' in str(web_interface.app.url_map)); html = web_interface.app.test_client().get('/').get_data(as_text=True); print('preflightPanel' in html, 'preflightRunConfig()' in html)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/create_run_config.py /tmp/posetestbot_preflight_smoke --sequence sync_aruco`:
  passed and wrote `run_config.json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_preflight.py /tmp/posetestbot_preflight_smoke --no-sensors --no-runtimes --json`:
  passed and reported an OK fake-mode dry-run preflight for the generated
  `sync_aruco` run config.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/robot_status.py --json`:
  passed and reported fake mode selected with the real iiwa profile at
  `172.31.1.147` and receiver IP `172.31.1.169`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/runtime_status.py --json`:
  passed and reported BlenderProc, Docker/FoundationPose checkout, BOP Toolkit
  checkout, and `pyzed.sl` currently unavailable on this machine.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.pipeline.sequences import build_sequence_plan; plan = build_sequence_plan(sequence_id='capture_to_bop_foundationpose_dry_run', run_root='/tmp/posetestbot_capture_to_bop_smoke', options={'sync_run': {'timestamp_source': 'sensor'}}); print(plan.sequence_id, len(plan.steps), plan.resources)"`:
  passed and printed the five-step capture-to-BOP/FoundationPose plan with
  `cpu`, `disk_io`, `estimator`, and `render` resources.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.io.artifacts import FOUNDATIONPOSE_PLAN; print(FOUNDATIONPOSE_PLAN)"`:
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from posetestbot.sensors.frame_writer import write_legacy_camera_sidecars; print('camera sidecar writer import ok')"`:
  passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_calibration_profiles.py /tmp/posetestbot_calibration_profiles.json --json`: passed.
- Unsandboxed `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json` saw 3 RealSense devices:
  `825412070181`, `033422071805`, and `923322072633`.
- The same status probe reported 0 OAK-D Pro devices after a DepthAI udev
  permission warning, and reported `pyzed.sl` unavailable for ZED 2i.
- Local fake iiwa smoke test on localhost-only UDP ports wrote both
  `raw_robot_ee_poses.json` and `dataset_manifest.json`.
- Non-destructive sync fixture test wrote derived synchronized RGB/depth frames,
  `match_robot_ee_poses.json`, and `sync_report.json` while raw files remained
  present.
- Derived sync fixture also copies camera sidecars, and BlenderProc prep helper
  tests cover zero-distortion fallback plus sensor-type calibration lookup.
- Run-level sync and manifest-tracked ArUco stage fixtures passed.
- Manifest-tracked BlenderProc prep fixture passed and produced expected `.npy`
  artifacts plus manifest stage output.
- Manifest-tracked BlenderProc render dry-run fixture passed and produced
  `blenderproc_render_plan.json` plus manifest stage output.
- Manifest-tracked BOP export fixture passed and produced a BOP-shaped scene,
  `bop_export_manifest.json`, and manifest stage output.
- Calibration-aware BOP export fixture passed and recorded profile intrinsics,
  depth scale, extrinsics, and profile IDs in scene/export metadata.
- BOP export fixture passed importing BlenderProc `scene_gt.json`,
  `scene_gt_info.json`, and `masks/` into the BOP scene.
- BOP export fixture passed writing `models/`, `models_info.json`, normalized
  numeric object IDs, and `test_targets_bop19.json`.
- BOP export fixture passed writing geometry-derived model diameter and
  bounding-box metadata.
- Calibration profile fixtures passed and legacy default transforms migrated to
  valid `calibration.v1` profile records.
- BlenderProc prep fixture passed using `--calibration-profiles` and produced a
  derived camera transform map plus camera poses from profile extrinsics.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  57 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/recommendations.py tests/test_pipeline_recommendations.py`:
  passed.
- `git diff --check`: passed.
- Live recommendation smoke on `/tmp/posetestbot_rewrite_status` now reports
  `follow_rewrite_status_next_action` as the first and only recommendation,
  with command `uv run python scripts/sensor_status.py --json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_hardware_status.py tests/test_robot_status.py tests/test_rewrite_gate.py`:
  38 passed.
- Live hardware-status smoke on
  `/tmp/posetestbot_gate_full_smoke_real_full_capture` now records
  `selected_profile.mode=real` from `run_config.json`; the report remains
  `overall_status=error` because RealSense discovery still fails in this
  workspace.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_hardware_status.py tests/test_robot_status.py tests/test_rewrite_gate.py tests/test_pipeline_stages.py tests/test_web_interface.py`:
  119 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/robot/status.py posetestbot/pipeline/hardware_status.py tests/test_hardware_status.py`:
  passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_hardware_status.py tests/test_pipeline_recommendations.py`:
  93 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py`:
  128 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py scripts/run_rewrite_status.py tests/test_rewrite_gate.py`:
  passed.
- `git diff --check`: passed.
- Live `run_rewrite_status.py` text output for the current mixed-root status now
  prints both sensor-blocker next actions: inspect sensor status, then refresh
  the full-capture hardware snapshot after camera visibility is fixed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py`:
  129 passed.
- Live artifact summary smoke for `/tmp/posetestbot_rewrite_status` now exposes
  `next_action_count=2` with both sensor-blocker commands.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_artifact_browser.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py`:
  183 passed.
- Transition index HTML smoke confirms the artifact renderer includes
  `appendArtifactNextActions`, `next_action_labels`, and `next_action_commands`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py`:
  130 passed.
- Live aggregate rewrite status now prints
  `capture_execution_plan.json is blocked by capture_plan_preflight:
  Capture-plan preflight status is error.` with the failing safety gate listed
  as a blocked check.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  184 passed, confirming rewrite gates, artifact summaries, recommendations,
  and the transition UI agree on the multi-action sensor blocker and the
  capture-plan-preflight safety gate blocker.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall posetestbot/pipeline/rewrite_gate.py posetestbot/io/artifact_browser.py posetestbot/pipeline/recommendations.py web_interface.py tests/test_rewrite_gate.py tests/test_artifact_browser.py tests/test_pipeline_recommendations.py tests/test_web_interface.py`:
  passed.
- `git diff --check`: passed.
- Live artifact summary smoke on
  `/tmp/posetestbot_gate_full_smoke_real_full_capture` now shows
  `capture_execution_plan.json` blocked by the nested
  `capture_plan_preflight` check with message
  `Capture-plan preflight status is error.`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py`:
  5 passed, including a regression where fd-level vendor SDK stdout noise is
  moved to stderr for `sensor_status.py --json`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json >/tmp/posetestbot_sensor_stdout_final.json 2>/tmp/posetestbot_sensor_stderr_final.txt`:
  exited 0; `/tmp/posetestbot_sensor_stdout_final.json` parsed as
  `sensor_status.v1` JSON with `total_connected=0`, while the DepthAI USB
  warning was written to stderr.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  196 passed, confirming the sensor diagnostic CLI, capture-plan preflight,
  rewrite gates, recommendations, artifact summaries, and transition UI still
  agree on the current hardware-visibility blocker.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  58 passed, including multi-action rewrite-status recommendations that surface
  both `sensor_status.py --json` and the follow-up
  `run_hardware_status_stage.py <run>` command.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after preserving the full rewrite-status next-action sequence in
  pipeline recommendations.
- Live recommendation smoke for `/tmp/posetestbot_rewrite_status` now returns
  `follow_rewrite_status_next_action` for `sensor_status.py --json` followed by
  `follow_rewrite_status_next_action_2` for
  `run_hardware_status_stage.py /tmp/posetestbot_gate_full_smoke_real_full_capture`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py`:
  58 passed after preserving action-specific rewrite-status reasons in
  multi-action recommendations.
- Live recommendation smoke now keeps the first action reason focused on
  inspecting camera SDK/device visibility and the second action reason focused
  on refreshing the run-scoped hardware snapshot after the sensor fix, while
  both still include the shared `hardware_status` blocker context.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after the action-specific recommendation reason update.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json --check-expected >/tmp/posetestbot_sensor_check.json 2>/tmp/posetestbot_sensor_check.err`:
  exited 2 in this workspace while still writing parseable `sensor_status.v1`
  JSON, correctly reflecting that expected lab cameras are not visible.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_sensor_status.py`:
  98 passed after making the rewrite-status sensor diagnostic command
  automation-strict with `--check-expected`.
- Rewrite-status sensor-blocker next actions now merge warning/error sensor
  diagnostics into `blocks_on`, not just `status=error` checks. For the current
  managed-workspace full-capture blocker, both actions now explicitly block on
  `sensor:realsense_d435`, `sensor:oak_d_pro`, and `sensor:zed_2i`, matching the
  stricter `sensor_status.py --json --check-expected` command.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after the strict sensor diagnostic next-action update.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after expanding sensor-blocker `blocks_on` coverage from hardware
  status diagnostics.
- Live `run_rewrite_status.py` output for `/tmp/posetestbot_rewrite_status`
  now prints `uv run python scripts/sensor_status.py --json --check-expected`
  as the first next action.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_artifact_browser.py::test_rewrite_status_report_summary_labels_blocked_status tests/test_artifact_browser.py::test_rewrite_status_report_summary_lists_multiple_next_actions`:
  2 passed after exposing `next_action_blocks_on` in rewrite-status artifact
  summaries.
- Live artifact summary smoke for `/tmp/posetestbot_rewrite_status` now reports
  both next actions blocking on `sensor:realsense_d435`, `sensor:oak_d_pro`,
  and `sensor:zed_2i`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after exposing next-action blocker lists through the artifact
  summary contract.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_web_interface.py tests/test_artifact_browser.py::test_rewrite_status_report_summary_lists_multiple_next_actions`:
  55 passed after rendering rewrite-status `next_action_blocks_on` in the
  transition artifact list.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after the transition UI blocker rendering update.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_pipeline_recommendations.py::test_recommendations_surface_all_rewrite_status_next_actions tests/test_web_interface.py::test_index_contains_run_config_controls tests/test_web_interface.py::test_pipeline_recommendations_endpoint_reports_next_steps`:
  3 passed after exposing recommendation `blocks_on` in the API and transition
  recommendation panel.
- Live recommendation smoke for `/tmp/posetestbot_rewrite_status` now returns
  `blocks_on=["sensor:realsense_d435", "sensor:oak_d_pro", "sensor:zed_2i"]`
  on both rewrite-status follow-up recommendations.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sensor_status.py tests/test_capture_plan_preflight.py tests/test_rewrite_gate.py tests/test_pipeline_recommendations.py tests/test_artifact_browser.py tests/test_web_interface.py`:
  197 passed after the recommendation `blocks_on` API/UI update.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rewrite_status.py /tmp/posetestbot_rewrite_status --write --gate-run-root rewrite_fake_end_to_end.v1=/tmp/posetestbot_gate_full_smoke --gate-run-root rewrite_full_capture.v1=/tmp/posetestbot_gate_full_smoke_real_full_capture`:
  on 2026-06-22 still reports blocked rewrite status with 1/4 gates ready and
  12/26 checks ready. The current next actions remain the strict sensor
  diagnostic followed by full-capture hardware snapshot refresh after the
  sensor fix.
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/sensor_status.py --json --check-expected >/tmp/posetestbot_sensor_check_latest.json 2>/tmp/posetestbot_sensor_check_latest.err`:
  on 2026-06-22 exited 2 while writing parseable `sensor_status.v1` JSON.
  The status reported 0/3 RealSense, 0/1 OAK-D Pro, and 0/1 ZED 2i visible;
  RealSense discovery failed with `RuntimeError: could not initialize udev
  monitor`, DepthAI emitted the container USB access warning on stderr, and
  `pyzed.sl` remains unavailable.
