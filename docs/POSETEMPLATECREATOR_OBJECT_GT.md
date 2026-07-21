# PoseTemplateCreator Object Ground Truth

PoseTestBot uses the pinned PoseTemplateCreator backend to turn managed CAD
models into printable, immutable object-pose templates. This workflow creates
ground-truth inputs and a validated BOP dataset; it does not run pose estimators
or evaluation.

## Install and verify

Initialize both printable-source checkouts and validate the local environment:

```bash
bash scripts/install.sh --with-posegridgen --with-posetemplatecreator
bash scripts/install.sh --check-only --with-posetemplatecreator
```

PoseTemplateCreator must be clean and exactly at
`450747bfee0e50b76f72ab38e1d0d04643124e02`. If it is missing, dirty, or at a
different revision, existing catalogs, bundles, and run selections remain
browsable, but CAD inspection, exact slicing, and generation are disabled.
Non-dry-run pose-template rendering requires BlenderProc 2.8.0.

## Coordinate contract

The printed blue dot is the `pose_template` origin: +X points right, +Y points
up, and +Z points out of the page. CAD and translations use millimetres. The
editor stores roll, pitch, and yaw in degrees with
`R = Rz(yaw) * Ry(pitch) * Rx(roll)`, plus a canonical 4×4 matrix and WXYZ
quaternion.

The operator-confirmed placement maps the blue dot into `template_base`:

```text
template_base_from_object =
    template_base_from_pose_template * pose_template_from_object
```

X/Y print scaling corrects printer output about the blue origin only. It does
not scale CAD, rigid transforms, the page border, axes, title, or GT.

## Operator workflow

1. Open **Pose Templates**. Upload one PLY/STL/OBJ and optionally one PNG.
   Inspection and conversion run as local CPU/disk jobs; the page reports
   queued/running status and refreshes the catalog automatically at completion.
   Archive replaces deletion.
2. Add up to 20 physical instances. Duplicate catalog objects are allowed.
   Enter each instance's X/Y/Z and roll/pitch/yaw. Z, roll, or pitch changes
   trigger a debounced exact posed-mesh intersection with template Z=0. An
   invalid or open intersection cannot be generated.
3. Select ISO paper/orientation and X/Y print compensation in percent. The upstream page
   contract fixes the printable margin and blue origin at 15 mm. Generate an
   immutable version only after fit and geometry validation pass. Download the
   PDF and manifest; clone to make edits.
4. Create or update the run in pose-template mode:

   ```bash
   uv run python scripts/create_run_config.py working_data/my_run \
     --dataset-mode pose_template
   ```

5. Open **Workflow → Ground Truth**, select an active immutable version, enter
   the measured full template-to-`template_base` placement, identify the
   operator, and explicitly confirm it. Identity defaults are not implicitly
   trusted.
6. Run BlenderProc preparation/rendering and BOP export through the existing
   workflow. No camera or robot operation is initiated by catalog, template,
   selection, preparation, or export actions.

## Artifacts and immutability

- Global catalog: `working_data/object_catalog/object_catalog.json` and
  UUID-addressed retained source/canonical/texture assets.
- Global library: `working_data/pose_templates/<template_uuid>/`, containing
  `pose_template_bundle.json`, exact preview data, asset snapshots, and PDF.
- Run selection: `pose_template_selection.json` plus the copied bundle at
  `processed/pose_template_selection/`.
- Prepared identity: `object_instances.json` and per-sensor BlenderProc
  `objects.json`/`posetestbot_render_instances.json`.
- BOP provenance: `bop/posetestbot_pose_template.json` and
  `bop/posetestbot_instance_map.json` beside standards-compatible BOP files.

Catalog and library archives are reversible. A selected run owns a complete
snapshot, so later archive actions do not change it. Selection replacement is
blocked after dependent object-instance, render, mask, or BOP artifacts exist;
the UI/API reports the exact blocking paths.

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
