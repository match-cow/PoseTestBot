# PoseTemplateCreator Object Ground Truth

PoseTestBot uses the pinned PoseTemplateCreator backend to turn managed CAD
models into printable, immutable object-pose templates. This workflow creates
ground-truth inputs and a validated BOP dataset; it does not run pose estimators
or evaluation. Test-object upload and lifecycle are owned by the separate
**Workpiece Catalogue** page; see
[WORKPIECE_CATALOGUE.md](WORKPIECE_CATALOGUE.md) for its persistence and API
contract.

## Install and verify

Initialize both printable-source checkouts and validate the local environment:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
bash scripts/install.sh --check-only --with-posetemplatecreator
```

PoseTemplateCreator must be clean and exactly at
`97ddb9b7b756912deb8c2d2d6dde186b461e5d9d`. If it is missing, dirty, or at a
different revision, existing catalogs, bundles, and run selections remain
browsable, and existing workpiece metadata remains editable, but new CAD
inspection/conversion, exact slicing, and generation are disabled.
Non-dry-run pose-template rendering requires BlenderProc 2.8.0.

## Coordinate contract

The nominal blue `pose_template` origin is 15 mm from the lower-left page
corner: +X points right, +Y points up, and +Z points out of the page. Printer
compensation moves and scales that dot with all other meaningful PDF content
about page centre; the footprint cards show its compensated physical position.
CAD and translations use millimetres. For each catalogue mesh,
PoseTemplateCreator ranks physically stable, grounded orientations and returns
an exact `source_to_placed` rigid transform. The editor then exposes only the
meaningful template-plane placement: X, Y, and rotation about +Z.

For a selected stable orientation, the authoritative transform is:

```text
pose_template_from_object =
    translate_xy * rotate_z * source_to_placed
```

Legacy bundles and six-DoF drafts remain readable. New templates do not ask an
operator to reproduce arbitrary roll, pitch, or Z values by eye.

The operator-confirmed placement maps the blue dot into `template_base`:

```text
template_base_from_object =
    template_base_from_pose_template * pose_template_from_object
```

X/Y print scaling follows the pinned upstream renderer and corrects printable
template content about the physical page centre. It does not scale CAD, rigid
transforms, the PDF page boundary, or GT.

## Operator workflow

1. Open **Workpiece Catalogue**. Upload one PLY/STL/OBJ and optionally one PNG.
   Inspection and conversion run as local CPU/disk jobs; the page reports
   queued/running status and refreshes the catalogue at completion. Add or edit
   the name, alias, description, tags, groups, and custom attributes. Use the
   single orbitable bounded 3D preview and compact isometric cards to identify
   the object without loading its full canonical PLY. Archive is reversible;
   permanent deletion is available only after archiving and explicit
   confirmation, and only when no pose-template bundle references the
   workpiece. If a unitless CAD file was interpreted at the
   wrong scale, archive it, inspect the before/after dimensions, and create an
   audited metre-to-millimetre (×1000) or millimetre-to-metre (÷1000) geometry
   revision. Restore it only after checking the corrected preview.
2. Open **Pose Templates** and filter the active workpieces by name, alias, tag,
   or group. Duplicate physical instances are allowed. Choose a ranked stable
   orientation using the same-scale isometric view and exact base footprint,
   then add it to the page. Drag or use arrow keys to position it, use the
   rotation handle, or enter exact X/Y/rotation values. PoseTestBot retains the
   upstream orientation ID, probability, grounded transform, slice height, and
   contours instead of accepting uploaded geometry in this workflow.
3. Select ISO paper/orientation and X/Y print compensation in percent. The
   upstream page contract fixes the nominal printable margin and blue origin at
   15 mm. Generate an
   immutable version only after the debounced server preview for the current
   editor state passes exact fit and geometry validation. Download the PDF and
   manifest; clone to make another immutable version. If a referenced
   workpiece now has a different geometry revision, cloning fails clearly and
   the operator must create a new template and review its stable orientation.
4. Create or update the run in pose-template mode:

   ```bash
   uv run python scripts/create_run_config.py working_data/my_run \
     --dataset-mode pose_template
   ```

5. Open **Workflow → Ground Truth**, select an active immutable version from
   its bounded footprint-preview card, and inspect the immutable objects in the
   single full interactive 3D scene. A **Simplified** badge reports card-only
   contour/point reduction; it never changes the printable or GT geometry.
   Enter the measured full
   template-to-`template_base` placement, identify the operator, and explicitly
   confirm it. Changing the version or any placement value clears that
   confirmation; identity defaults are not implicitly trusted.
6. Run BlenderProc preparation/rendering and BOP export through the existing
   workflow. No camera or robot operation is initiated by catalog, template,
   selection, preparation, or export actions.

## Artifacts and immutability

- Global Workpiece Catalogue:
  `working_data/object_catalog/object_catalog.json`, numbered revisions, and
  UUID-addressed retained source/canonical/texture assets. Canonical geometry
  revisions live below each object's `derived/` directory; a current
  `pose_template_orientation_analysis.json` cache and its separately bounded
  `pose_template_orientation_thumbnail.json` card cache sit beside the
  canonical PLY. Both are reproducible, hash/revision-bound derivatives rather
  than immutable catalogue assets. Editable metadata lives beside stable
  catalogue UUID and BOP `obj_id` identity.
- Global library: `working_data/pose_templates/<template_uuid>/`, containing
  `pose_template_bundle.json`, exact preview data, a hash-verified bounded
  `pose_template_thumbnail.json`, asset snapshots, and PDF. The thumbnail keeps
  every instance's largest compensated contour, then admits secondary contours
  round-robin, with hard limits of 400 contours, 4096 total points, and 48
  points per contour. Its approximation record reports every source/included
  count. Pre-thumbnail bundles remain readable: the endpoint derives the same
  bounded representation from their verified preview in memory and never
  modifies the historical bundle. `GET /pose-templates/library` returns
  metadata-only summaries; exact contours and preview meshes are available only
  from the explicitly requested detail/full-preview endpoints. New manifests
  omit the duplicate raw `nominal_contours` and `compensated_contours` arrays
  from instance records; authoritative exact contours remain hash-verified in
  `pose_template_preview.json`, so this reduces synchronous metadata cost
  without changing the PDF, placement, or GT.
- Run selection: `pose_template_selection.json` plus the copied bundle at
  `processed/pose_template_selection/`. A hidden
  `.pose_template_selection.transaction.json` exists only while a replacement
  transaction or its cleanup is recoverable.
- Prepared identity: `object_instances.json` and per-sensor BlenderProc
  `objects.json`/`posetestbot_render_instances.json`.
- BOP provenance: `bop/posetestbot_pose_template.json` and
  `bop/posetestbot_instance_map.json` beside standards-compatible BOP files.

Catalogue and library archives are reversible. Catalogue JSON export/import is
metadata-only: JSON never embeds CAD or texture bytes, and import skips records
whose matching local UUID assets are absent. A generated immutable bundle
snapshots its selected workpieces and assets, and a selected run owns a complete
copy of that bundle, so later catalogue metadata or archive actions do not
change either snapshot. Selection replacement is blocked after dependent
object-instance, render, mask, or BOP artifacts exist; the UI/API reports the
exact blocking paths.

Ordinary library cards/details read the bounded, self-hashed manifest and do
not hash every immutable PDF, preview, mesh, and texture. A thumbnail, full
preview, PDF, or individual instance asset request verifies the manifest plus
only the requested declared artifact. Oversized pre-bounded legacy manifests
use strict whole-bundle validation as a compatibility fallback. Authoritative
operations remain stricter: run selection, catalogue-reference checks before
permanent deletion, and explicit whole-bundle validation reject missing,
modified, undeclared, or symlinked tree entries.

Template slicing, preview construction, PDF rendering, and asset copying run
outside the catalogue mutation lock. Publication takes that lock only long
enough to re-check every selected canonical geometry/texture identity and
atomically expose the staged bundle. A unit correction, archive, or deletion
that wins the race therefore causes stale publication to fail and discard its
stage, without blocking ordinary catalogue work for the full render.

## Validation and recovery

Run the acquisition-only readiness gate after export:

```bash
uv run python scripts/run_rewrite_gate.py working_data/my_run \
  --gate rewrite_bop_export_readiness.v1 --write
```

For pose-template mode the gate cross-checks selection, prepared geometry,
calibration, renderer version/identity, every BOP GT index, model hashes, and
both provenance sidecars. Preserve raw capture data; retry preparation or
export into derived artifacts after correcting a blocker.

Selection creation and replacement hold the template-library lock while they
strictly validate and snapshot the chosen active bundle. The run-local reader
then cross-checks the selection record against that verified copy, including
UUIDs, catalogue/BOP identities, assets, transforms, print compensation,
configuration hash, frame semantics, timestamp, confirmation type, and
operator provenance. It fails closed on path traversal, symlinks, partial
trees, or inconsistent fields.

Promotion of the copied bundle, `pose_template_selection.json`, and any updated
`run_config.json` is one staged transaction. A durable prepared journal is
written before live paths move. On the next selection-locked access, an
interrupted prepared transaction restores the exact prior artifacts; a
committed transaction finishes the remaining stage/backup cleanup. The journal
accepts only the three managed run-local targets and rejects unsafe or
incomplete recovery metadata.
