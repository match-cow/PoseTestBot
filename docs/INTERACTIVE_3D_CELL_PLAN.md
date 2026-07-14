# Interactive 3D Cell and Dataset Contents

## Summary

- Add a dedicated read-only **Cell** page to the React operator console while
  keeping `web_interface.py` as the compatibility launcher.
- Use pytransform3d as the authoritative backend frame graph and React Three
  Fiber/Three.js for browser rendering.
- Show the `template_base` frame, physical robot base, moving flange/TCP,
  calibrated sensors, HRI template, calibration target, selected object meshes,
  and recorded trajectories.
- Before capture, show every resolvable configured component. After
  synchronization, provide exact frame playback with sensor-timeline selection,
  play/pause, and scrubbing.
- Do not depict an articulated iiwa arm in v1: datasets contain flange poses but
  no joint states or robot geometry. Render accurate base/flange/TCP proxies and
  clearly identify unresolved components.

## Interfaces and Data Contracts

- Extend `run_config.v1` with `selected_objects: string[]`.
  - New CLI and web configurations always snapshot explicit object names.
  - New runs initially select all valid registry objects.
  - `[]` intentionally denotes an objectless RGB-D run.
  - Legacy configs without the field load all current registry objects with a
    compatibility warning; rewriting the config materializes the snapshot.
  - Changing `object_folder` preserves matching selections, leaves new objects
    unchecked, and reports missing prior names for operator resolution.
- Introduce one shared object-registry service used by the viewer, BlenderProc
  preparation, and BOP export. It will validate safe names, PLY/texture
  existence, finite rigid transforms, and expose the existing inverted
  object-to-`template_base` transform convention.
- Assign BOP `obj_id` values from the complete alphabetically sorted registry,
  then filter by selection without renumbering; subsets may therefore have ID
  gaps.
- Add versioned, read-only APIs:
  - `GET /ui/object-registry?run_root=&object_folder=` returns registry entries,
    IDs, validation state, and current selection.
  - `GET /ui/cell-scene?run_root=` returns `cell_scene.v1`: millimetre units,
    right-handed Z-up coordinates, `template_base` reference frame, typed
    entities, local transforms, provenance, warnings, timeline metadata, and a
    bounded trajectory preview.
  - `GET /ui/cell-scene/timeline?run_root=&timeline_id=&offset=&limit=` returns
    exact ordered flange poses in `cell_timeline.v1`; limit is capped at 2,000
    and the frontend prefetches adjacent pages.
  - Allowlisted mesh and texture routes serve only registered objects selected
    by that run, using contained-path validation and conditional caching.
- Represent transforms as `translation_mm` plus
  `rotation_quaternion_wxyz`, with explicit `entity_to_parent` semantics. The
  browser only applies the resolved scene hierarchy; it does not reproduce
  calibration or robot-pose composition.
- Add repeatable `--object-name` and mutually exclusive `--objectless` options
  to configuration and object-dependent stage CLIs. Configured sequences inject
  the saved selection unless an explicit stage override is present.

## Implementation Changes

- Build the scene through pytransform3d's frame manager, reusing existing KUKA
  pose conversion and calibration profile selection:
  - Resolve static cameras to `template_base` and eye-in-hand cameras relative
    to `robot_flange`.
  - Resolve fixed physical-base and TCP frames from
    `frames.fixed_transforms`.
  - Derive camera frustums from calibrated intrinsics and resolution.
  - Include unresolved entities with a reason instead of inventing transforms.
  - Use synchronized `match_robot_ee_poses.json` timelines when available,
    defaulting to the first configured sensor; expose each synchronized sensor
    as a selectable timeline and fall back to `raw_robot_ee_poses.json`.
  - Never interpolate recorded poses. Trajectory previews may be uniformly
    sampled, but scrubbed frames remain exact.
- Add the Cell route and lazy-loaded frontend feature using React Three Fiber,
  Three.js `PLYLoader`, orbit controls, and a demand-driven canvas:
  - Orbit, pan, zoom, reset, top/front/perspective presets, selection details,
    visibility layers, trajectory toggle, timeline selector, play/pause, and
    frame scrubber.
  - Render selected PLY meshes with vertex colors or UV textures where
    available, otherwise a neutral material.
  - Render camera frustums, coordinate axes, base/flange/TCP primitives,
    calibration-target geometry, and status styling for
    planned/recorded/unresolved entities.
  - Keep world coordinates numerically Z-up in millimetres; configure the Three
    camera and grid accordingly.
  - Gracefully fall back to the component/provenance list if WebGL is
    unavailable or an individual mesh fails.
- Package the existing 420 x 297 mm HRI SVG as a self-contained web asset. Its
  centre maps to `template_base`, SVG right is +X, SVG down is +Y, and it lies on
  the template XY plane. Render `calibration_target.json` separately when its
  placement is known.
- Propagate object selection through derived artifacts:
  - BlenderProc preparation transactionally writes only selected models and
    clears stale prepared output.
  - Objectless preparation writes camera inputs and explicit empty object
    metadata.
  - Objectless render stages emit a successful skipped plan with
    `skip_reason="objectless_run"`, never invoke BlenderProc, and do not require
    its runtime.
  - BOP subset export includes only selected models, GT, masks, targets, and
    COCO categories; stale or unselected GT is rejected.
  - Objectless BOP export writes RGB, depth, camera metadata, empty GT/GT-info
    arrays per frame, an explicit empty `test_targets_bop19.json`, no models or
    masks, and optional COCO images with empty categories/annotations.
  - Add selection, objectless state, stable ID mapping, registry provenance, and
    validation counts to `bop_export_manifest.v2`.
- Update BOP readiness gates and recommendations so explicit objectless exports
  are valid only when models, targets, masks, and scene object references are
  consistently empty. Valid camera calibration and RGB-D scene integrity remain
  required.
- Add `three`, `@react-three/fiber`, and `@react-three/drei` through Bun, with
  `@types/three` as a dev dependency; update the lockfile and bundled Flask UI.
  Pytransform3d is already a Python dependency.
- Ensure `frontend/src/lib` is tracked despite the generic `lib/` ignore rule,
  package the template asset, and update README/INSTALL guidance plus
  `docs/REWRITE_PROGRESS.md`. The existing
  `scripts/install.sh --with-web-build` workflow remains the installation path
  and will be revalidated.

## Test Plan

- Unit-test registry validation, transform inversion, stable subset IDs,
  explicit empty selection, legacy config fallback, and sequence option
  injection.
- Test pytransform3d scene composition for static and eye-in-hand sensors,
  base/TCP frames, objects, template/target placement, synchronized and raw
  timelines, pagination, and unresolved-frame behavior.
- Test scene and asset APIs for schema stability, invalid artifacts, unknown
  objects, traversal and symlink containment, texture fallback, and objectless
  runs.
- Test BlenderProc/BOP subset and objectless flows, including stale artifact
  removal, skipped execution, empty targets/GT, readiness-gate acceptance, and
  rejection of nonselected GT.
- Add Playwright coverage for Cell navigation, WebGL canvas creation, layer
  toggles, entity inspection, timeline seeking, missing-calibration warnings,
  objectless state, and WebGL fallback without touching hardware.
- Run frontend typecheck/lint/build, targeted pytest suites, the full pytest
  suite, `git diff --check`, package/wheel asset checks, the prescribed
  Playwright test, and installer validation. Do not execute physical capture or
  install browser binaries without explicit authorization.

## Assumptions

- The Cell page is entirely read-only; object inclusion is edited in Workflow
  Setup, while visibility controls are session-local.
- Accurate proxies plus actual dataset object meshes are the v1 fidelity target;
  an articulated iiwa requires future joint-state recording and an approved
  URDF/CAD asset.
- Missing transforms remain visibly unresolved rather than being placed at
  identity.
- "Everything in the dataset" follows the run's saved object selection and
  current acquisition artifacts, with exported-manifest mismatch shown as stale
  provenance.
