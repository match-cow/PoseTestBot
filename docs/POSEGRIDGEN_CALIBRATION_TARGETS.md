# PoseGridGen Calibration Targets

PoseTestBot uses the pinned PoseGridGen source checkout to generate printable
ArUco targets. This workflow creates acquisition/calibration inputs only; it
does not command the robot or open cameras.

## Enable generation

PoseGridGen generation requires Python 3.12 and the exact clean submodule at
`third_party/PoseGridGen`:

```bash
bash scripts/install.sh --with-posegridgen
bash scripts/install.sh --check-only --with-posegridgen
uv run posetestbot-web
```

The required revision is
`ad152e369e8d2746d0cf66cb1455f2371b0ec0f0`. A missing, dirty, mismatched, or
wheel-only checkout disables generation. Existing `calibration_target.v1` and
`calibration_target.v2` artifacts remain readable in that state. The status and
the direct `/calibration-targets` route explain the concrete failure. Navigation,
saved-bundle browsing, downloads, and run selection remain available even when
generation is disabled.

## Generate and select

Open **Calibration Targets** in the operator console:

1. Choose the ArUco dictionary, rows/columns, marker size, gap, paper,
   orientation, annotations, and independent X/Y print compensation.
2. Optionally attach a board-to-base pose and use **Fit to page** when needed.
3. Inspect the debounced PNG preview, enter a display name, and queue
   **Generate bundle**.
4. Download and inspect the source JSON, canonical target JSON, and printable
   PDF. Generation does not change the active run.
5. Choose a configured run, select the bundle, and declare one placement:
   `unknown`, `template_base_identity`, or `posegridgen_board_to_base`.

The simplified **Workflow → Calibration** screen can also select any saved
bundle directly for an immutable calculation attempt. Its two modes derive the
target mounting per attempt: eye-in-hand estimates a target stationary relative
to `template_base`, while eye-to-hand estimates a target attached to
`robot_flange`. This does not require PoseGridGen to be available and does not
initiate physical capture.

`posegridgen_board_to_base` is available only when the source records that
pose. Selection cross-checks PoseGridGen's matrix, translation, and quaternion,
converts metres to millimetres and XYZW to WXYZ, and explicitly treats the base
as `template_base`.

## Artifact contract

Each immutable library entry is stored at:

```text
working_data/calibration_targets/<opaque-uuid>/
  calibration_target_bundle.json
  posegridgen_source.json
  calibration_target.json
  calibration_target.pdf
```

`calibration_target_bundle.v1` records the UUID, display name, creation time,
pinned generator revision, configuration/geometry hashes, and fixed file paths,
media types, sizes, and SHA-256 values. Generation stages every file and
promotes the complete directory atomically. Confirmed deletion is allowed only
for an inactive library bundle.

Selection copies the unchanged bundle to
`<run>/calibration_targets/<target_id>/`, writes the placement-aware root
`<run>/calibration_target.json`, and adds `run_config.v1.calibration_target`
hash/provenance fields. The bundle, root target, run config, and dataset
manifest are promoted together with rollback on failure.

Intent-level calculation snapshots the bundle below
`<run>/processed/calibration/<attempt_id>/target_bundle/`. Prior attempts and
raw capture data are never replaced. Only explicit recommendation acceptance
copies the selected evidence and bundle into canonical run artifacts.

`calibration_target.v2` makes the compensated `corners_mm` for every marker
authoritative. The target frame is `aruco_grid`, its origin is the compensated
outer board top-left, +X points right, +Y down, and +Z into the page. Consumers
use a generic OpenCV `Board`; they do not reconstruct a regular grid or apply
print compensation again.

## Reselection and preflight

Selecting the same target and placement again is idempotent. Capture and
synchronization output alone do not block a change. A different target or
placement is rejected once detections, calibrated intrinsics, poses, coverage,
observations, candidates, solver/validation output, promoted profiles,
rectification, or BOP output exists. The API returns the concrete blocker paths;
create a new run rather than deleting calibration evidence.

Preflight verifies bundle containment, absence of symlinks, file hashes,
canonical target agreement, pinned generator compatibility, run-config hashes,
and placement when the solver uses `known_target` or `compare`. Target selection
changes `run_config.json`, so older run-preflight evidence becomes stale
automatically.

## Legacy import

The `calibration_target_import` stage first resolves the run-config selection.
Without one, it accepts PoseGridGen schema 2.0 source JSON through the exact
pinned checkout or the legacy ArUcoGridGen 1.0 format:

```bash
uv run python scripts/run_calibration_target_import.py working_data/example_run \
  --source working_data/example_run/aruco_grid_config.json \
  --aligned-to-template-base
```

All new imports write v2. Legacy v1 target specs remain loadable and expand to
explicit marker corners in memory.

## API and jobs

The scoped API surface is:

- `GET /calibration-targets/status`
- `GET /calibration-targets/capabilities`
- `POST /calibration-targets/fit`
- `POST /calibration-targets/preview`
- `GET /calibration-targets/bundles`
- `POST /calibration-targets/generate`
- `DELETE /calibration-targets/bundles/<target_id>`
- `POST /calibration-targets/bundles/<target_id>/select`
- `GET /calibration-targets/bundles/<target_id>/download/<source|target|pdf>`

The intent-level calculation façade consumes those saved bundles through:

- `GET /calibration/setup?run_root=...`
- `GET /calibration/attempts?run_root=...`
- `POST /calibration/attempts`
- `GET /calibration/attempts/<attempt_id>?run_root=...`
- `POST /calibration/attempts/<attempt_id>/promote`

Attempt creation records stable sensor keys and queues one `cpu`/`disk_io`
parent job. Promotion is a separate queued transaction and requires passing
recommendations or explicit passing candidate IDs.

Request bodies are capped at 256 KiB. Generation queues `cpu` and `disk_io`;
selection queues `disk_io`. Commands use fixed argument arrays and appear in
the existing Jobs page. Deletion requires `confirm: true`, atomically removes
the library bundle, and rejects the target active for the selected run. No
generic filesystem download endpoint is provided.
