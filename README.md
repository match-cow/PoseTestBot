<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="posetestbot/web/static/cow_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="posetestbot/web/static/cow_light.png">
    <img src="posetestbot/web/static/cow_light.png" alt="PoseTestBot cow logo" width="96">
  </picture>
</p>

<h1 align="center">PoseTestBot</h1>

<p align="center">
  <strong>From supervised multi-camera capture to traceable BOP datasets.</strong>
</p>

PoseTestBot is an acquisition-first system for building 6D object-pose
datasets from robot-mounted and static RGB-D cameras. It brings camera
calibration, robot-aware capture, synchronization, optional ground-truth
generation, and BOP export into one operator-guided workflow.

The goal is not simply to collect frames. It is to produce a dataset whose
origin can be understood and reproduced: raw evidence stays untouched, every
dataset-derived result belongs to its run, and exports remain bound to the exact
calibration, timing, geometry, and object inputs that created them.

## The Workflow

The desktop console guides two outcomes:

1. **Camera calibration** — configure the camera rig, capture target
   observations, compare candidates, validate the result, and publish a
   reusable calibration profile.
2. **Object dataset** — snapshot a calibration and pose template into a run,
   supervise physical capture, synchronize and validate the data, then export
   a BOP dataset.

```text
reusable inputs → run snapshot → supervised capture → sync + quality gates
                                                    → base BOP export
                                                      ↳ optional GT + masks
                                                        → annotated BOP export
```

Calibration targets feed calibration runs. Workpieces feed immutable pose
templates. Dataset runs snapshot the selected calibration and pose template,
so later library edits cannot silently change an existing dataset.
Long-running work is queued, recoverable, and visible in **Jobs** after
navigation.

## What a Run Produces

- preserved RGB-D frames, timestamps, and robot poses;
- calibration, synchronization, and quality evidence;
- optional object-pose ground truth, masks, and visibility evidence; and
- a standard BOP dataset with camera parameters, object models, targets, and
  compact PoseTestBot provenance.

The base export is useful without annotations. Pose ground truth can be added
when object placement is known, while the full annotation mode adds masks and
visibility data for evaluation-ready datasets.

## Core Principles

- **Raw data is evidence.** Processing writes derived artifacts instead of
  rewriting or deleting the only copy of a capture.
- **A run is self-contained.** Configuration and reusable inputs are
  snapshotted and hash-bound to their outputs.
- **Hardware action is explicit.** Readiness checks never authorize motion or
  start physical capture.
- **Failures stay inspectable.** Partial captures, logs, reports, and job state
  are retained for diagnosis.
- **The output is estimator-agnostic.** Pose-estimation methods consume the
  exported dataset elsewhere.

## Repository Boundary

PoseTestBot ends at a validated BOP dataset. It does not run pose estimators or
convert estimator-specific results.

The **Inspect → BOP Evaluation** page is intentionally limited to dataset
validation. It can apply the pinned official BOP19 metrics to an already
compatible result CSV, or to a clearly labelled deterministic test result
derived from ground truth. Evaluation evidence remains run-scoped and is never
an acquisition-pipeline stage.

## Lab Context and Safety

The current cell uses a KUKA LBR iiwa with three Intel RealSense D435-class
cameras, one Luxonis OAK-D Pro, and one Stereolabs ZED 2i. Physical capture
requires an operator, explicit authorization, and both execution safety gates.

The iiwa UDP `STOP` command is not a safety stop: it cannot interrupt active
motion and exits the waiting calibration application. Do not use it between
calibration captures.

## Run the Console

PoseTestBot uses Python 3.12 and `uv`. Follow the
[installation guide](INSTALL.md) for SDK and optional-tool setup, then start
the operator console with:

```bash
uv run posetestbot-web
```

The console is unauthenticated and can expose deliberate robot controls. Run
it only on the trusted lab network, or bind it to localhost for local use.

## Documentation

- [Operator workflows](docs/OPERATOR_WORKFLOWS.md)
- [Installation and runtime requirements](INSTALL.md)
- [Workpiece Catalogue](docs/WORKPIECE_CATALOGUE.md)
- [Pose templates and object ground truth](docs/POSETEMPLATECREATOR_OBJECT_GT.md)
- [Contributor and safety rules](AGENTS.md)
