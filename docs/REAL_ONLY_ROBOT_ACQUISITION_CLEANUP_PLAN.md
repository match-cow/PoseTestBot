# Real-Only Robot Acquisition Cleanup

Implementation status: code and automated validation completed on 2026-07-10;
fresh operator-triggered physical capture acceptance remains pending. This
document is retained as the cleanup acceptance checklist.

## Summary

- Make the lab KUKA iiwa the sole robot profile and remove the fake acquisition
  stack entirely.
- Keep hardware-free unit tests, SDK stubs, and numerical fixtures; physical
  regression testing becomes an explicit operator-triggered real capture
  sequence.
- Preserve existing real `run_config.v1` and capture artifacts. Reject fake
  configs and IDs without compatibility aliases.

## Implementation Changes

- Collapse robot configuration to `robot_profile()` returning the real lab
  profile. Remove localhost constants, `POSETESTBOT_ROBOT_MODE`, mode selection,
  and fake-profile status fields while retaining IP/port/velocity overrides and
  the constant artifact value `"mode": "real"`.
- Remove fake-only code and artifacts: `fake_iiwa_controller`, capture rehearsal,
  synthetic RGB-D generation, the fake smoke script, their stages/sequences,
  artifact constants/browser entries, and
  `rewrite_fake_acquisition_to_bop.v1`.
- Simplify capture planning to emit only enabled sensor commands followed by the
  robot pose receiver. Remove `robot_controller` handling and fake-controller
  tuning.
- Collapse execution to full real capture: remove the execution `mode` input and
  all `plan_only`/`pose_only_fake` branches, while retaining explicit
  `allow_real_robot` and `allow_cameras` safety gates. Continue emitting
  `"mode": "full"` in existing v1 execution artifacts.
- Make `real_full_capture_validation` the default new-run sequence. Keep
  sequence planning non-executing by default.
- Decouple the RealSense-only smoke workflow from robot mode so it remains a
  camera-only check and never commands the robot.
- Update web forms, JavaScript, APIs, job classification, recommendations, and
  command construction for the real-only interface.
- Reduce rewrite status to the three real-data gates: full capture, calibration
  validation, and BOP export readiness. Remove fake-to-real sibling-root logic.
- Rewrite `AGENTS.md`, `README.md`, `INSTALL.md`, installer notes, the system
  overview, and `docs/REWRITE_PROGRESS.md` around the real-only workflow. Mark
  completed design documents as historical where they mention retired
  validation paths.

## Public Interface Changes

- Remove `robot_mode` parameters and `--robot-mode`/`--robot_mode` options from
  run creation, pose reception, and IIWA start/stop commands. Replace the
  receiver's diagnostic `--test` flag with `--verbose`.
- Reject API payloads containing retired robot/execution mode selectors.
- Require loaded `run_config.v1` files to contain
  `robot_profile.mode == "real"`; existing fake configs fail validation.
- Introduce `robot_status.v2` containing the fixed `selected_profile`,
  normal-network IP, environment overrides, protocols, and notes; remove
  `profiles`, `fake_first`, and duplicated fake/real selection data.
- Remove the fake stage, sequence, gate, and artifact IDs immediately. Existing
  files under `working_data/` are not deleted or rewritten.

## Test Plan

- Delete fake-controller, rehearsal, synthetic RGB-D, and fake-gate tests.
- Rewrite unit tests around real configs with injected process/SDK stand-ins,
  covering:
  - real defaults and environment overrides;
  - rejection of fake configs, flags, APIs, stages, sequences, and gates;
  - capture plans containing sensors plus exactly one pose receiver;
  - blocking when either hardware permission is absent;
  - successful supervised execution, cancellation, process cleanup, and raw-data
    preservation;
  - camera-only RealSense smoke behavior;
  - real-only web UI and the three-gate rewrite status.
- Run `uv` pytest, Ruff, `git diff --check`, build/install checks, and Playwright
  on a host that permits localhost browser tests.
- On the lab host, create a fresh run using connected sensors and a conservative
  `0.05 m/s`, review `real_full_capture_validation` with `--plan-only`, then
  deliberately execute it and require `rewrite_full_capture.v1` to pass.
  Calibration and BOP gates subsequently use real captured data from that
  workflow.
- Never reuse a run root containing raw frames or robot poses, and never launch
  the physical sequence automatically from the normal pytest suite.

## Assumptions

- Unit-test doubles and synthetic mathematical/calibration fixtures remain
  because they do not expose a fake operator workflow.
- BlenderProc/BOP dry-run planning remains because it processes real captured
  data without emulating the robot.
- Physical validation requires operator readiness and execution outside the
  restricted USB/network sandbox used by coding agents.
