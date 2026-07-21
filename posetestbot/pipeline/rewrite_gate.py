"""Milestone gates that keep the rewrite focused on proved workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    CALIBRATION_PROFILES,
    CALIBRATION_VALIDATION_REPORT,
    CAPTURE_EXECUTION_PLAN,
    CAPTURE_EXECUTION_REPORT,
    CAPTURE_PLAN,
    CAPTURE_PLAN_PREFLIGHT_REPORT,
    DEPTH_DIR,
    FRAME_METADATA_JSONL,
    HARDWARE_STATUS_REPORT,
    MODELS_DIR,
    OBJECT_INSTANCES,
    PIPELINE_SEQUENCE_PLAN,
    POSE_TEMPLATE_SELECTION,
    PROCESSED_DIR,
    REWRITE_GATE_REPORT,
    REWRITE_STATUS_REPORT,
    RGB_DIR,
    RUN_CONFIG,
    RUN_PREFLIGHT_REPORT,
    BOP_TARGETS_BOP19,
)


SCHEMA_VERSION = "rewrite_gate_report.v1"
STATUS_SCHEMA_VERSION = "rewrite_status_report.v1"
FULL_CAPTURE_GATE_ID = "rewrite_full_capture.v1"
CALIBRATION_VALIDATION_GATE_ID = "rewrite_calibration_validation.v1"
BOP_EXPORT_READINESS_GATE_ID = "rewrite_bop_export_readiness.v1"
GATE_IDS = (
    FULL_CAPTURE_GATE_ID,
    CALIBRATION_VALIDATION_GATE_ID,
    BOP_EXPORT_READINESS_GATE_ID,
)


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(loaded, dict):
        return None, "json_root_not_object"
    return loaded, None


def _check(
    *,
    name: str,
    path: Path,
    ok: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "ready" if ok else "blocked",
        "artifact": path.as_posix(),
        "message": message,
        "details": details or {},
    }


def _json_file_check(name: str, path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    data, error = _load_json_object(path)
    if error:
        return (
            _check(
                name=name,
                path=path,
                ok=False,
                message=f"{path.name} is {error}.",
            ),
            None,
        )
    return (
        _check(
            name=name,
            path=path,
            ok=True,
            message=f"{path.name} exists and is valid JSON.",
        ),
        data,
    )


def _status_value(data: dict[str, Any]) -> str | None:
    value = data.get("overall_status", data.get("status"))
    return str(value) if value is not None else None


def _report_problem_checks(
    data: Mapping[str, Any],
    *,
    blocking_statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    blocking_statuses = blocking_statuses or {"error", "blocked", "failed"}
    checks = data.get("checks")
    if not isinstance(checks, list):
        checks = data.get("gates")
    if not isinstance(checks, list):
        return []
    problem_checks: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("status") or "")
        if status not in blocking_statuses:
            continue
        problem_checks.append(
            {
                "name": check.get("name"),
                "status": status,
                "message": check.get("message"),
                "details": (
                    dict(check.get("details"))
                    if isinstance(check.get("details"), Mapping)
                    else {}
                ),
            }
        )
    return problem_checks


def _sensor_diagnostics_from_status(
    data: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(data, Mapping):
        return []
    families = data.get("families")
    if not isinstance(families, list):
        return []
    diagnostics: list[dict[str, Any]] = []
    for family in families:
        if not isinstance(family, Mapping):
            continue
        family_diagnostics = family.get("diagnostics")
        if not isinstance(family_diagnostics, list):
            continue
        for diagnostic in family_diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            diagnostics.append(
                {
                    "sensor_type": family.get("sensor_type"),
                    "display_name": family.get("display_name"),
                    **dict(diagnostic),
                }
            )
    return diagnostics


def _sensor_diagnostics_from_report(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    sensor_status = data.get("sensor_status")
    return _sensor_diagnostics_from_status(
        sensor_status if isinstance(sensor_status, Mapping) else None
    )


def _sensor_blockers_from_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for diagnostic in diagnostics:
        severity = str(diagnostic.get("severity") or "")
        sensor_type = diagnostic.get("sensor_type")
        if severity not in {"error", "warning"} or not sensor_type:
            continue
        blocker = f"sensor:{sensor_type}"
        if blocker not in blockers:
            blockers.append(blocker)
    return blockers


def format_blocker_detail_lines(
    blocker: Mapping[str, Any],
    *,
    indent: str = "  ",
    max_checks: int = 3,
    max_diagnostics: int = 3,
    max_hints: int = 2,
) -> list[str]:
    """Return compact human-readable detail lines for a gate blocker."""

    details = blocker.get("details")
    if not isinstance(details, Mapping):
        return []

    lines: list[str] = []
    error_checks = details.get("error_checks")
    if isinstance(error_checks, list):
        shown_checks = 0
        for check in error_checks:
            if not isinstance(check, Mapping):
                continue
            name = check.get("name")
            message = check.get("message")
            if name or message:
                text = f"blocked check: {name}" if name else "blocked check"
                if message:
                    text = f"{text} - {message}"
                lines.append(f"{indent}{text}")
                shown_checks += 1
            if shown_checks >= max_checks:
                break

    sensor_diagnostics = details.get("sensor_diagnostics")
    if isinstance(sensor_diagnostics, list):
        shown_diagnostics = 0
        for diagnostic in sensor_diagnostics:
            if not isinstance(diagnostic, Mapping):
                continue
            message = diagnostic.get("message")
            if message:
                lines.append(f"{indent}diagnostic: {message}")
                shown_diagnostics += 1
            hints = diagnostic.get("hints")
            if isinstance(hints, list):
                for hint in hints[:max_hints]:
                    lines.append(f"{indent}  hint: {hint}")
            if shown_diagnostics >= max_diagnostics:
                break

    return lines


def _gate_report(
    *,
    gate_id: str,
    run_root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers = [check for check in checks if check["status"] != "ready"]
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": gate_id,
        "run_root": run_root.as_posix(),
        "overall_status": "ready" if not blockers else "blocked",
        "summary": {
            "ready_count": len(checks) - len(blockers),
            "blocked_count": len(blockers),
            "check_count": len(checks),
        },
        "checks": checks,
        "next_blockers": [
            {
                "name": check["name"],
                "artifact": check["artifact"],
                "message": check["message"],
                "details": check.get("details", {}),
            }
            for check in blockers
        ],
    }


def _capture_selected_roles(capture: dict[str, Any]) -> list[str]:
    selected_roles = capture.get("selected_roles")
    if isinstance(selected_roles, list):
        return [str(role) for role in selected_roles]

    plan = capture.get("capture_execution_plan")
    if isinstance(plan, dict) and isinstance(plan.get("selected_roles"), list):
        return [str(role) for role in plan["selected_roles"]]

    processes = capture.get("processes")
    if isinstance(processes, list):
        return [
            str(process["role"])
            for process in processes
            if isinstance(process, dict) and process.get("role")
        ]
    return []


def _enabled_run_config_sensors(run_config: dict[str, Any]) -> list[dict[str, Any]]:
    capture = run_config.get("capture")
    if not isinstance(capture, dict):
        return []
    sensors = capture.get("sensors")
    if not isinstance(sensors, list):
        return []
    return [
        sensor
        for sensor in sensors
        if isinstance(sensor, dict) and sensor.get("enabled", True)
    ]


def _sensor_folder_name(sensor: dict[str, Any]) -> str:
    sensor_type = str(sensor.get("sensor_type") or "sensor")
    device_id = str(sensor.get("device_id") or "auto")
    if sensor_type in {"realsense", "realsense_d435"}:
        return f"realsense_{device_id}"
    if sensor_type in {"luxonis", "oak", "oak_d_pro"}:
        return f"luxonis_{device_id}"
    if sensor_type == "zed_2i":
        return f"zed_2i_{device_id}"
    return f"{sensor_type}_{device_id}"


def _png_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return len(list(path.glob("*.png")))


def _numeric_pose_file_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    count = 0
    for child in path.iterdir():
        if not child.is_file():
            continue
        try:
            int(child.stem)
        except ValueError:
            continue
        count += 1
    return count


def _artifact_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _bop_export_readiness_checks(
    root: Path,
    *,
    require_targets: bool = True,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    bop_root = root / BOP_DIR
    bop_manifest_path = bop_root / BOP_EXPORT_MANIFEST
    check, bop_export = _json_file_check("bop_export", bop_manifest_path)
    exports: list[Mapping[str, Any]] = []
    scene_images: dict[int, set[int]] = {}
    scene_objects: dict[tuple[int, int], set[int]] = {}
    scene_annotation_ids: dict[tuple[int, int], list[int]] = {}
    scene_sensors: dict[int, str] = {}
    seen_scene_ids: set[int] = set()
    objectless = False
    template_selection: dict[str, Any] | None = None
    template_object_instances: dict[str, Any] | None = None
    template_instance_map: dict[str, Any] | None = None
    if bop_export is not None:
        objectless = bop_export.get("objectless") is True
        raw_exports = bop_export.get("exports")
        exports = [
            export for export in raw_exports if isinstance(export, Mapping)
        ] if isinstance(raw_exports, list) else []
        validation = bop_export.get("validation")
        validation_ok = (
            isinstance(validation, Mapping) and validation.get("status") == "ok"
        )
        export_schema = bop_export.get("schema_version")
        ok = (
            export_schema in {"bop_export_manifest.v2", "bop_export_manifest.v3"}
            and bop_export.get("format") == "bop-scenewise"
            and len(exports) > 0
            and validation_ok
        )
        check = _check(
            name="bop_export",
            path=bop_manifest_path,
            ok=ok,
            message=(
                f"bop_export_manifest.json is a validated BOP-scenewise {export_schema} export."
                if ok
                else (
                    "bop_export_manifest.json must be v2/v3, BOP-scenewise, contain "
                    "scenes, and record successful validation."
                )
            ),
            details={
                "schema_version": bop_export.get("schema_version"),
                "format": bop_export.get("format"),
                "export_count": len(exports),
                "validation_ok": validation_ok,
            },
        )
    checks.append(check)

    if bop_export is not None and bop_export.get("dataset_mode") == "pose_template":
        pose_template_path = bop_root / "posetestbot_pose_template.json"
        instance_map_path = bop_root / "posetestbot_instance_map.json"
        pose_template, pose_error = _load_json_object(pose_template_path)
        instance_map, instance_error = _load_json_object(instance_map_path)
        template_selection, selection_error = _load_json_object(
            root / POSE_TEMPLATE_SELECTION
        )
        template_object_instances, objects_error = _load_json_object(
            root / OBJECT_INSTANCES
        )
        template_instance_map = instance_map
        template_summary = bop_export.get("pose_template")
        template_ok = (
            pose_template is not None
            and pose_template.get("schema_version") == "posetestbot_pose_template.v1"
            and isinstance(template_summary, Mapping)
            and pose_template.get("template_uuid") == template_summary.get("template_uuid")
            and pose_template.get("bundle_sha256") == template_summary.get("bundle_sha256")
            and template_selection is not None
            and template_selection.get("schema_version") == "pose_template_selection.v1"
            and template_selection.get("placement_confirmed") is True
            and pose_template.get("template_uuid") == template_selection.get("template_uuid")
            and pose_template.get("bundle_sha256") == template_selection.get("bundle_sha256")
            and pose_template.get("configuration_sha256")
            == template_selection.get("configuration_sha256")
            and pose_template.get("template_base_from_pose_template")
            == template_selection.get("template_base_from_pose_template")
            and template_object_instances is not None
            and template_object_instances.get("schema_version") == "object_instances.v1"
            and template_object_instances.get("template_uuid")
            == template_selection.get("template_uuid")
            and template_object_instances.get("bundle_sha256")
            == template_selection.get("bundle_sha256")
        )
        instances = instance_map.get("instances") if instance_map is not None else None
        instance_ok = (
            instance_map is not None
            and instance_map.get("schema_version") == "posetestbot_bop_instance_map.v1"
            and isinstance(instances, list)
        )
        checks.extend(
            [
                _check(
                    name="bop_pose_template_provenance",
                    path=pose_template_path,
                    ok=template_ok,
                    message=(
                        "Pose-template selection and BOP manifest provenance agree."
                        if template_ok
                        else (
                            "Pose-template provenance is missing or inconsistent: "
                            f"bop={pose_error}, selection={selection_error}, objects={objects_error}."
                        )
                    ),
                ),
                _check(
                    name="bop_instance_map",
                    path=instance_map_path,
                    ok=instance_ok,
                    message=(
                        "BOP GT instance-map sidecar is present."
                        if instance_ok
                        else f"BOP GT instance-map sidecar is invalid: {instance_error}."
                    ),
                ),
            ]
        )

    for index, export in enumerate(exports):
        scene_value = export.get("scene_folder")
        scene_folder = (
            (bop_root / str(scene_value)).resolve()
            if isinstance(scene_value, str) and scene_value
            else None
        )
        scene_label = str(export.get("sensor_name") or f"scene_{index}")
        if scene_folder is None:
            checks.append(
                _check(
                    name=f"bop_scene:{scene_label}",
                    path=root / BOP_DIR,
                    ok=False,
                    message="BOP export scene entry is missing scene_folder.",
                    details={"export_index": index},
                )
            )
            continue
        try:
            scene_folder.relative_to(bop_root.resolve())
        except ValueError:
            checks.append(
                _check(
                    name=f"bop_scene:{scene_label}",
                    path=bop_root,
                    ok=False,
                    message="BOP export scene_folder escapes the BOP dataset root.",
                    details={"export_index": index, "scene_folder": scene_value},
                )
            )
            continue
        rgb_count = _png_count(scene_folder / RGB_DIR)
        depth_count = _png_count(scene_folder / DEPTH_DIR)
        rgb_names = {path.name for path in (scene_folder / RGB_DIR).glob("*.png")}
        depth_names = {path.name for path in (scene_folder / DEPTH_DIR).glob("*.png")}
        scene_camera, camera_error = _load_json_object(scene_folder / "scene_camera.json")
        scene_gt, gt_error = _load_json_object(scene_folder / "scene_gt.json")
        scene_gt_info, info_error = _load_json_object(scene_folder / "scene_gt_info.json")
        expected_keys = {str(int(Path(name).stem)) for name in rgb_names}
        json_keys_ok = all(
            data is not None and set(data) == expected_keys
            for data in (scene_camera, scene_gt, scene_gt_info)
        )
        ok = (
            scene_folder.is_dir()
            and rgb_count > 0
            and rgb_names == depth_names
            and json_keys_ok
        )
        try:
            scene_id = int(export.get("scene_id"))
        except (TypeError, ValueError):
            scene_id = -1
            ok = False
        split = export.get("split")
        expected_scene_folder = (
            bop_root / str(split) / f"{scene_id:06d}"
            if isinstance(split, str) and split
            else None
        )
        if (
            scene_id < 0
            or scene_id in seen_scene_ids
            or expected_scene_folder is None
            or scene_folder != expected_scene_folder.resolve()
        ):
            ok = False
        seen_scene_ids.add(scene_id)
        image_ids = {int(key) for key in expected_keys}
        scene_images[scene_id] = image_ids
        scene_sensors[scene_id] = scene_label
        if scene_gt is not None:
            for image_id, annotations in scene_gt.items():
                object_ids = set()
                annotation_ids: list[int] = []
                if isinstance(annotations, list):
                    for annotation in annotations:
                        if not isinstance(annotation, Mapping):
                            ok = False
                            continue
                        try:
                            annotation_obj_id = int(annotation["obj_id"])
                            object_ids.add(annotation_obj_id)
                            annotation_ids.append(annotation_obj_id)
                        except (KeyError, TypeError, ValueError):
                            ok = False
                else:
                    ok = False
                scene_objects[(scene_id, int(image_id))] = object_ids
                scene_annotation_ids[(scene_id, int(image_id))] = annotation_ids
        if objectless and (
            any(scene_objects.get((scene_id, image_id), set()) for image_id in image_ids)
            or (scene_folder / "mask").exists()
            or (scene_folder / "mask_visib").exists()
        ):
            ok = False
        checks.append(
            _check(
                name=f"bop_scene:{scene_label}",
                path=scene_folder,
                ok=ok,
                message=(
                    f"{scene_label} has aligned RGB-D and scene metadata keys."
                    if ok
                    else (
                        f"{scene_label} must include matching RGB/depth names and "
                        "camera/GT/GT-info keys for every image."
                    )
                ),
                details={
                    "export_index": index,
                    "rgb_count": rgb_count,
                    "depth_count": depth_count,
                    "camera_error": camera_error,
                    "gt_error": gt_error,
                    "gt_info_error": info_error,
                    "json_keys_ok": json_keys_ok,
                    "standard_scene_path": (
                        expected_scene_folder.as_posix()
                        if expected_scene_folder is not None
                        else None
                    ),
                },
            )
        )

    dataset_info_path = bop_root / "dataset_info.json"
    dataset_info, dataset_info_error = _load_json_object(dataset_info_path)
    dataset_info_ok = (
        dataset_info is not None
        and dataset_info.get("schema_version") == "posetestbot_bop_dataset_info.v1"
        and dataset_info.get("bop_format") == "scenewise"
        and dataset_info.get("scene_count") == len(scene_images)
        and set(dataset_info.get("sensors", [])) == set(scene_sensors.values())
    )
    checks.append(
        _check(
            name="bop_dataset_info",
            path=dataset_info_path,
            ok=dataset_info_ok,
            message=(
                "dataset_info.json matches the exported BOP scenes."
                if dataset_info_ok
                else f"dataset_info.json is missing or inconsistent: {dataset_info_error}."
            ),
        )
    )

    frame_map_path = bop_root / "posetestbot_bop_frame_map.json"
    frame_map, frame_map_error = _load_json_object(frame_map_path)
    mapped_scenes = frame_map.get("scenes") if frame_map is not None else None
    frame_map_ok = (
        frame_map is not None
        and frame_map.get("schema_version") == "posetestbot_bop_frame_map.v2"
        and isinstance(mapped_scenes, Mapping)
        and set(mapped_scenes) == {str(scene_id) for scene_id in scene_images}
    )
    if frame_map_ok and isinstance(mapped_scenes, Mapping):
        for scene_id, image_ids in scene_images.items():
            entry = mapped_scenes.get(str(scene_id))
            frames = entry.get("frames") if isinstance(entry, Mapping) else None
            if (
                not isinstance(entry, Mapping)
                or entry.get("sensor_name") != scene_sensors[scene_id]
                or not isinstance(frames, Mapping)
                or set(frames) != {str(image_id) for image_id in image_ids}
            ):
                frame_map_ok = False
                break
    checks.append(
        _check(
            name="bop_posetestbot_bop_frame_map",
            path=frame_map_path,
            ok=frame_map_ok,
            message=(
                "posetestbot_bop_frame_map.json covers every exported scene and image."
                if frame_map_ok
                else (
                    "posetestbot_bop_frame_map.json is missing or inconsistent: "
                    f"{frame_map_error}."
                )
            ),
        )
    )

    targets_path = bop_root / BOP_TARGETS_BOP19
    check, targets = _json_file_check("bop_targets", targets_path)
    if targets is not None:
        check = _check(
            name="bop_targets",
            path=targets_path,
            ok=False,
            message="test_targets_bop19.json must contain a JSON list.",
            details={"target_count": 0},
        )
    elif targets_path.is_file():
        try:
            target_rows = json.loads(targets_path.read_text())
        except json.JSONDecodeError as exc:
            check = _check(
                name="bop_targets",
                path=targets_path,
                ok=False,
                message=f"test_targets_bop19.json is invalid_json: {exc.msg}.",
                details={"target_count": 0},
            )
        else:
            target_count = len(target_rows) if isinstance(target_rows, list) else 0
            references_ok = isinstance(target_rows, list)
            if isinstance(target_rows, list):
                for target in target_rows:
                    if not isinstance(target, Mapping):
                        references_ok = False
                        continue
                    try:
                        scene_id = int(target["scene_id"])
                        image_id = int(target["im_id"])
                        obj_id = int(target["obj_id"])
                    except (KeyError, TypeError, ValueError):
                        references_ok = False
                        continue
                    if image_id not in scene_images.get(scene_id, set()) or obj_id not in scene_objects.get((scene_id, image_id), set()):
                        references_ok = False
            ok = (
                isinstance(target_rows, list)
                and references_ok
                and (target_count > 0 or not require_targets or objectless)
            )
            check = _check(
                name="bop_targets",
                path=targets_path,
                ok=ok,
                message=(
                    "test_targets_bop19.json contains target rows."
                    if ok and target_count > 0
                    else "test_targets_bop19.json exists as an explicit empty target list."
                    if ok
                    else "test_targets_bop19.json must contain at least one target row."
                ),
                details={
                    "target_count": target_count,
                    "references_ok": references_ok,
                },
            )
    checks.append(check)

    models_info_path = bop_root / MODELS_DIR / "models_info.json"
    check, models_info = _json_file_check("bop_models_info", models_info_path)
    if models_info is not None:
        model_count = len(models_info)
        geometry_ok = all(
            isinstance(value, Mapping)
            and isinstance(value.get("diameter"), (int, float))
            and float(value["diameter"]) > 0
            for value in models_info.values()
        )
        ok = model_count > 0 and geometry_ok
        check = _check(
            name="bop_models_info",
            path=models_info_path,
            ok=ok,
            message=(
                "models_info.json contains exported model metadata."
                if ok
                else "models_info.json must contain at least one exported model."
            ),
            details={"model_count": model_count, "geometry_ok": geometry_ok},
        )
    elif objectless:
        models_absent = not (bop_root / MODELS_DIR).exists()
        check = _check(
            name="bop_models_info",
            path=models_info_path,
            ok=models_absent,
            message=(
                "Explicit objectless export contains no model artifacts."
                if models_absent
                else "Objectless export must not contain a models directory."
            ),
            details={"objectless": True, "model_count": 0},
        )
    checks.append(check)

    if template_selection is not None and template_object_instances is not None:
        selection_path = root / POSE_TEMPLATE_SELECTION
        object_rows = template_object_instances.get("instances")
        map_rows = template_instance_map.get("instances") if template_instance_map else None
        selection_digest_ok = (
            selection_path.is_file()
            and template_object_instances.get("selection_sha256")
            == hashlib.sha256(selection_path.read_bytes()).hexdigest()
        )
        object_by_uuid = {
            str(item.get("instance_uuid")): item
            for item in object_rows or []
            if isinstance(item, Mapping)
        } if isinstance(object_rows, list) else {}
        expected_gt_keys = {
            (scene_id, image_id, gt_id)
            for (scene_id, image_id), ids in scene_annotation_ids.items()
            for gt_id in range(len(ids))
        }
        mapped_gt_keys: set[tuple[int, int, int]] = set()
        mapping_ok = isinstance(map_rows, list) and bool(object_by_uuid)
        if isinstance(map_rows, list):
            for row in map_rows:
                if not isinstance(row, Mapping):
                    mapping_ok = False
                    continue
                try:
                    key = (int(row["scene_id"]), int(row["im_id"]), int(row["gt_id"]))
                    obj_id = int(row["obj_id"])
                    instance = object_by_uuid[str(row["instance_uuid"])]
                    annotation_obj_id = scene_annotation_ids[(key[0], key[1])][key[2]]
                except (KeyError, IndexError, TypeError, ValueError):
                    mapping_ok = False
                    continue
                mapped_gt_keys.add(key)
                if (
                    obj_id != annotation_obj_id
                    or int(instance.get("obj_id", -1)) != obj_id
                    or instance.get("catalog_uuid") != row.get("catalog_uuid")
                ):
                    mapping_ok = False
            mapping_ok = mapping_ok and mapped_gt_keys == expected_gt_keys

        render_ok = True
        render_details: dict[str, str] = {}
        expected_render_instances: set[tuple[str, str, int]] = set()
        for item in object_by_uuid.values():
            try:
                expected_render_instances.add(
                    (str(item["instance_uuid"]), str(item["catalog_uuid"]), int(item["obj_id"]))
                )
            except (KeyError, TypeError, ValueError):
                render_ok = False
        for sensor_name in scene_sensors.values():
            candidates = [
                root / PROCESSED_DIR / "rectified" / sensor_name / "blenderproc" / "output"
                / "posetestbot_render_instances.json",
                root / PROCESSED_DIR / "synchronized" / sensor_name / "blenderproc" / "output"
                / "posetestbot_render_instances.json",
            ]
            sidecar_path = next((path for path in candidates if path.is_file()), candidates[-1])
            sidecar, sidecar_error = _load_json_object(sidecar_path)
            rows = sidecar.get("instances") if sidecar else None
            actual: set[tuple[str, str, int]] = set()
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, Mapping):
                        render_ok = False
                        continue
                    try:
                        actual.add(
                            (
                                str(item["instance_uuid"]),
                                str(item["catalog_uuid"]),
                                int(item["obj_id"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        render_ok = False
            sensor_ok = (
                sidecar is not None
                and sidecar.get("schema_version") == "posetestbot_render_instances.v1"
                and sidecar.get("blenderproc_version") == "2.8.0"
                and sidecar.get("identity_contract")
                == "bop_gt_index_matches_loaded_instance_order.v1"
                and actual == expected_render_instances
            )
            render_ok = render_ok and sensor_ok
            render_details[sensor_name] = "ok" if sensor_ok else str(sidecar_error or "mismatch")

        geometry_ok = isinstance(models_info, Mapping)
        if isinstance(models_info, Mapping):
            geometry_by_obj: dict[int, str] = {}
            for item in object_by_uuid.values():
                try:
                    obj_id = int(item["obj_id"])
                except (KeyError, TypeError, ValueError):
                    geometry_ok = False
                    continue
                digest = str(item.get("canonical_ply_sha256", ""))
                if obj_id in geometry_by_obj and geometry_by_obj[obj_id] != digest:
                    geometry_ok = False
                geometry_by_obj[obj_id] = digest
            for obj_id, digest in geometry_by_obj.items():
                model = models_info.get(str(obj_id))
                geometry = model.get("posetestbot_geometry") if isinstance(model, Mapping) else None
                if not isinstance(geometry, Mapping) or geometry.get("source_sha256") != digest:
                    geometry_ok = False
        evidence_ok = selection_digest_ok and mapping_ok and render_ok and geometry_ok
        checks.append(
            _check(
                name="bop_pose_template_evidence_agreement",
                path=root / OBJECT_INSTANCES,
                ok=evidence_ok,
                message=(
                    "Selection, geometry, rendered identities, BOP GT indices, and instance provenance agree."
                    if evidence_ok
                    else "Pose-template selection, geometry, rendering, and BOP instance evidence disagree."
                ),
                details={
                    "selection_digest_ok": selection_digest_ok,
                    "geometry_ok": geometry_ok,
                    "mapping_ok": mapping_ok,
                    "render_ok": render_ok,
                    "render_sensors": render_details,
                    "expected_gt_count": len(expected_gt_keys),
                    "mapped_gt_count": len(mapped_gt_keys),
                },
            )
        )

    if require_targets and bop_export is not None:
        profile_statuses = {
            str(profile.get("profile_id")): profile.get("status")
            for profile in bop_export.get("calibration_profiles", [])
            if isinstance(profile, Mapping)
        }
        scene_profile_ids = [export.get("calibration_profile_id") for export in exports]
        calibration_ok = bool(scene_profile_ids) and all(
            isinstance(profile_id, str)
            and profile_statuses.get(profile_id) == "valid"
            for profile_id in scene_profile_ids
        )
        checks.append(
            _check(
                name="bop_calibration_provenance",
                path=bop_manifest_path,
                ok=calibration_ok,
                message=(
                    "Every BOP scene references a valid calibration profile."
                    if calibration_ok
                    else "BOP readiness requires valid calibration provenance for every scene."
                ),
                details={"scene_profile_ids": scene_profile_ids},
            )
        )

    return checks


def build_full_capture_gate_report(run_root: str | Path) -> dict[str, Any]:
    """Report whether real/full camera capture has actually been validated."""

    root = Path(run_root)
    checks: list[dict[str, Any]] = []

    check, run_config = _json_file_check("run_config", root / RUN_CONFIG)
    if run_config is not None:
        robot_mode = run_config.get("robot_profile", {}).get("mode")
        sensors = _enabled_run_config_sensors(run_config)
        ok = robot_mode == "real" and len(sensors) > 0
        check = _check(
            name="run_config",
            path=root / RUN_CONFIG,
            ok=ok,
            message=(
                "run_config.json targets real capture with enabled sensors."
                if ok
                else (
                    "run_config.json must target robot_profile.mode=real and "
                    "include at least one enabled sensor."
                )
            ),
            details={
                "robot_mode": robot_mode,
                "enabled_sensor_count": len(sensors),
            },
        )
    else:
        sensors = []
    checks.append(check)

    check, preflight = _json_file_check("run_preflight", root / RUN_PREFLIGHT_REPORT)
    if preflight is not None:
        status = _status_value(preflight)
        ok = status in {"ok", "warning", "ready", "succeeded"}
        check = _check(
            name="run_preflight",
            path=root / RUN_PREFLIGHT_REPORT,
            ok=ok,
            message=(
                "run_preflight_report.json has acceptable status."
                if ok
                else "run_preflight_report.json is missing an acceptable status."
            ),
            details={"status": status},
        )
    checks.append(check)

    check, hardware = _json_file_check(
        "hardware_status",
        root / HARDWARE_STATUS_REPORT,
    )
    if hardware is not None:
        status = _status_value(hardware)
        robot_status = hardware.get("robot_status")
        selected_profile = (
            robot_status.get("selected_profile")
            if isinstance(robot_status, dict)
            else None
        )
        selected_robot_mode = (
            selected_profile.get("mode")
            if isinstance(selected_profile, dict)
            else None
        )
        status_ok = status in {"ok", "warning", "ready", "succeeded"}
        robot_mode_ok = selected_robot_mode == "real"
        ok = status_ok and robot_mode_ok
        if ok:
            message = (
                "hardware_status_report.json has acceptable status and real robot mode."
            )
        elif not robot_mode_ok:
            message = (
                "hardware_status_report.json must select the real robot profile."
            )
        else:
            message = "hardware_status_report.json must be ok or warning."
        check = _check(
            name="hardware_status",
            path=root / HARDWARE_STATUS_REPORT,
            ok=ok,
            message=message,
            details={
                "status": status,
                "selected_robot_mode": selected_robot_mode,
                "status_ok": status_ok,
                "robot_mode_ok": robot_mode_ok,
                "error_checks": _report_problem_checks(hardware),
                "sensor_diagnostics": _sensor_diagnostics_from_report(hardware),
            },
        )
    checks.append(check)

    check, capture_plan = _json_file_check("capture_plan", root / CAPTURE_PLAN)
    if capture_plan is not None:
        commands = capture_plan.get("commands")
        command_count = len(commands) if isinstance(commands, list) else 0
        ok = command_count > 0
        check = _check(
            name="capture_plan",
            path=root / CAPTURE_PLAN,
            ok=ok,
            message=(
                "capture_plan.json records planned capture commands."
                if ok
                else "capture_plan.json must record planned capture commands."
            ),
            details={"command_count": command_count},
        )
    checks.append(check)

    check, capture_plan_preflight = _json_file_check(
        "capture_plan_preflight",
        root / CAPTURE_PLAN_PREFLIGHT_REPORT,
    )
    if capture_plan_preflight is not None:
        status = _status_value(capture_plan_preflight)
        ok = status in {"ok", "warning", "ready", "succeeded"}
        check = _check(
            name="capture_plan_preflight",
            path=root / CAPTURE_PLAN_PREFLIGHT_REPORT,
            ok=ok,
            message=(
                "capture_plan_preflight_report.json has acceptable status."
                if ok
                else "capture_plan_preflight_report.json must be ok or warning."
            ),
            details={
                "status": status,
                "error_checks": _report_problem_checks(capture_plan_preflight),
                "sensor_diagnostics": _sensor_diagnostics_from_report(
                    capture_plan_preflight
                ),
            },
        )
    checks.append(check)

    check, execution_plan = _json_file_check(
        "capture_execution_plan",
        root / CAPTURE_EXECUTION_PLAN,
    )
    if execution_plan is not None:
        status = str(execution_plan.get("status") or "")
        ready_to_execute = bool(execution_plan.get("ready_to_execute"))
        problem_checks = _report_problem_checks(execution_plan)
        selected_roles = [
            str(role)
            for role in execution_plan.get("selected_roles", [])
            if isinstance(role, str)
        ]
        ok = (
            status == "ok"
            and ready_to_execute
            and "sensor_capture" in selected_roles
            and "robot_pose_receiver" in selected_roles
        )
        check = _check(
            name="capture_execution_plan",
            path=root / CAPTURE_EXECUTION_PLAN,
            ok=ok,
            message=(
                "capture_execution_plan.json is ready for full capture."
                if ok
                else (
                    f"capture_execution_plan.json is blocked by "
                    f"{problem_checks[0]['name']}: {problem_checks[0]['message']}"
                    if problem_checks
                    and problem_checks[0].get("name")
                    and problem_checks[0].get("message")
                    else (
                        "capture_execution_plan.json must be ready_to_execute "
                        "with sensor_capture and robot_pose_receiver selected."
                    )
                )
            ),
            details={
                "status": status,
                "ready_to_execute": ready_to_execute,
                "selected_roles": selected_roles,
                "error_checks": problem_checks,
            },
        )
    checks.append(check)

    check, capture = _json_file_check(
        "capture_execution",
        root / CAPTURE_EXECUTION_REPORT,
    )
    if capture is not None:
        status = capture.get("status")
        mode = capture.get("mode")
        allow_cameras = bool(capture.get("allow_cameras"))
        raw_pose_count = int(capture.get("raw_pose_count") or 0)
        selected_roles = _capture_selected_roles(capture)
        processes = capture.get("processes")
        sensor_processes = [
            process
            for process in processes
            if isinstance(process, dict) and process.get("role") == "sensor_capture"
        ] if isinstance(processes, list) else []
        sensor_process_ready = bool(sensor_processes) and all(
            process.get("status") in {"succeeded", "stopped"}
            and process.get("started_at")
            and process.get("ended_at")
            for process in sensor_processes
        )
        ok = (
            status == "succeeded"
            and mode == "full"
            and allow_cameras
            and raw_pose_count > 0
            and "sensor_capture" in selected_roles
            and "robot_pose_receiver" in selected_roles
            and sensor_process_ready
        )
        check = _check(
            name="capture_execution",
            path=root / CAPTURE_EXECUTION_REPORT,
            ok=ok,
            message=(
                "capture_execution_report.json proves supervised full capture."
                if ok
                else (
                    "capture_execution_report.json must be succeeded full mode "
                    "with camera commands, robot poses, and completed sensor processes."
                )
            ),
            details={
                "status": status,
                "mode": mode,
                "allow_cameras": allow_cameras,
                "raw_pose_count": raw_pose_count,
                "selected_roles": selected_roles,
                "sensor_process_count": len(sensor_processes),
            },
        )
    checks.append(check)

    for sensor in sensors:
        folder_name = _sensor_folder_name(sensor)
        sensor_path = root / folder_name
        rgb_count = _png_count(sensor_path / RGB_DIR)
        depth_count = _png_count(sensor_path / DEPTH_DIR)
        metadata_path = sensor_path / FRAME_METADATA_JSONL
        ok = (
            rgb_count > 0
            and depth_count > 0
            and rgb_count == depth_count
            and metadata_path.is_file()
        )
        checks.append(
            _check(
                name=f"sensor_frames:{folder_name}",
                path=sensor_path,
                ok=ok,
                message=(
                    f"{folder_name} contains raw RGB-D frames and metadata."
                    if ok
                    else (
                        f"{folder_name} must contain matching rgb/*.png and "
                        f"depth/*.png frame counts plus {FRAME_METADATA_JSONL}."
                    )
                ),
                details={
                    "rgb_count": rgb_count,
                    "depth_count": depth_count,
                    "frame_count_match": rgb_count == depth_count,
                    "has_frame_metadata": metadata_path.is_file(),
                },
            )
        )

    return _gate_report(gate_id=FULL_CAPTURE_GATE_ID, run_root=root, checks=checks)


def build_calibration_validation_gate_report(run_root: str | Path) -> dict[str, Any]:
    """Report whether production calibration profiles were validated and promoted."""

    root = Path(run_root)
    checks: list[dict[str, Any]] = []

    check, validation = _json_file_check(
        "calibration_validation",
        root / CALIBRATION_VALIDATION_REPORT,
    )
    promoted_profile_count = 0
    promoted_profile_ids: set[str] = set()
    if validation is not None:
        overall_status = validation.get("overall_status")
        promotion = validation.get("promotion")
        profile_count = int(validation.get("profile_count") or 0)
        promotable_profile_count = int(
            validation.get("promotable_profile_count") or 0
        )
        promotion_requested = (
            bool(promotion.get("requested"))
            if isinstance(promotion, dict)
            else False
        )
        promotion_promoted = (
            bool(promotion.get("promoted"))
            if isinstance(promotion, dict)
            else False
        )
        promoted_profile_count = (
            int(promotion.get("profile_count") or 0)
            if isinstance(promotion, dict)
            else 0
        )
        raw_promoted_profile_ids = (
            promotion.get("promoted_profile_ids")
            if isinstance(promotion, dict)
            else None
        )
        if isinstance(raw_promoted_profile_ids, list) and all(
            isinstance(profile_id, str) and profile_id
            for profile_id in raw_promoted_profile_ids
        ):
            promoted_profile_ids = set(raw_promoted_profile_ids)
        promotion_path = (
            str(promotion.get("path"))
            if isinstance(promotion, dict) and promotion.get("path")
            else None
        )
        ok = (
            overall_status == "ok"
            and promotion_requested
            and promotion_promoted
            and profile_count > 0
            and promotable_profile_count == profile_count
            and len(promoted_profile_ids) == profile_count
            and promoted_profile_count >= profile_count
            and promotion_path is not None
        )
        check = _check(
            name="calibration_validation",
            path=root / CALIBRATION_VALIDATION_REPORT,
            ok=ok,
            message=(
                "calibration_validation_report.json records promoted valid profiles."
                if ok
                else (
                    "calibration_validation_report.json must be ok, explicitly "
                    "promoted, and promote every validated profile."
                )
            ),
            details={
                "overall_status": overall_status,
                "profile_count": profile_count,
                "promotable_profile_count": promotable_profile_count,
                "promotion_requested": promotion_requested,
                "promotion_promoted": promotion_promoted,
                "promoted_profile_count": promoted_profile_count,
                "promoted_profile_ids": sorted(promoted_profile_ids),
                "promotion_path": promotion_path,
            },
        )
    checks.append(check)

    check, profile_collection = _json_file_check(
        "calibration_profiles",
        root / CALIBRATION_PROFILES,
    )
    if profile_collection is not None:
        profiles = profile_collection.get("profiles")
        profile_summaries = []
        if isinstance(profiles, list):
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                quality = profile.get("quality")
                quality = quality if isinstance(quality, dict) else {}
                profile_summaries.append(
                    {
                        "profile_id": profile.get("profile_id"),
                        "sensor_id": profile.get("sensor_id"),
                        "sensor_type": profile.get("sensor_type"),
                        "status": profile.get("status"),
                        "num_inliers": quality.get("num_inliers"),
                        "residual_translation_mm": quality.get(
                            "residual_translation_mm"
                        ),
                        "residual_rotation_deg": quality.get(
                            "residual_rotation_deg"
                        ),
                    }
                )
        all_profiles_valid = bool(profile_summaries) and all(
            profile["status"] == "valid"
            and isinstance(profile["num_inliers"], int)
            and profile["num_inliers"] > 0
            and profile["residual_translation_mm"] is not None
            and profile["residual_rotation_deg"] is not None
            for profile in profile_summaries
        )
        collection_profile_ids = {
            profile["profile_id"]
            for profile in profile_summaries
            if isinstance(profile["profile_id"], str)
        }
        promoted_profiles_present = bool(promoted_profile_ids) and (
            promoted_profile_ids <= collection_profile_ids
        )
        collection_count_matches_promotion = (
            promoted_profile_count == len(profile_summaries)
        )
        ok = (
            all_profiles_valid
            and promoted_profiles_present
            and collection_count_matches_promotion
        )
        check = _check(
            name="calibration_profiles",
            path=root / CALIBRATION_PROFILES,
            ok=ok,
            message=(
                "calibration_profiles.json contains promoted valid profiles."
                if ok
                else (
                    "calibration_profiles.json must contain valid profiles "
                    "with inlier counts and residual quality fields."
                )
            ),
            details={
                "profile_count": len(profile_summaries),
                "promoted_profile_count": promoted_profile_count,
                "promoted_profiles_present": promoted_profiles_present,
                "collection_count_matches_promotion": collection_count_matches_promotion,
                "profiles": profile_summaries,
            },
        )
    checks.append(check)

    return _gate_report(
        gate_id=CALIBRATION_VALIDATION_GATE_ID,
        run_root=root,
        checks=checks,
    )


def build_bop_export_readiness_gate_report(run_root: str | Path) -> dict[str, Any]:
    """Report whether a run folder contains a structurally usable BOP dataset."""

    root = Path(run_root)
    checks = _bop_export_readiness_checks(root)
    return _gate_report(
        gate_id=BOP_EXPORT_READINESS_GATE_ID,
        run_root=root,
        checks=checks,
    )


def build_gate_report(run_root: str | Path, *, gate_id: str) -> dict[str, Any]:
    if gate_id == FULL_CAPTURE_GATE_ID:
        return build_full_capture_gate_report(run_root)
    if gate_id == CALIBRATION_VALIDATION_GATE_ID:
        return build_calibration_validation_gate_report(run_root)
    if gate_id == BOP_EXPORT_READINESS_GATE_ID:
        return build_bop_export_readiness_gate_report(run_root)
    raise ValueError(f"Unknown rewrite gate: {gate_id}")


def write_gate_report(run_root: str | Path, *, gate_id: str) -> tuple[Path, dict[str, Any]]:
    root = Path(run_root)
    report = build_gate_report(root, gate_id=gate_id)
    path = root / REWRITE_GATE_REPORT
    atomic_write_json(path, report)
    return path, report


def _action(
    *,
    gate_id: str,
    label: str,
    command: list[str],
    reason: str,
    blocks_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "label": label,
        "command": command,
        "reason": reason,
        "blocks_on": blocks_on or [],
    }


def _existing_hardware_status_blockers(gate_run_root: Path) -> list[str]:
    data, error = _load_json_object(gate_run_root / HARDWARE_STATUS_REPORT)
    if error or data is None:
        return []
    checks = data.get("checks")
    if not isinstance(checks, list):
        return []
    blockers: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("status") == "error" and check.get("name"):
            blockers.append(str(check["name"]))
    for blocker in _sensor_blockers_from_diagnostics(
        _sensor_diagnostics_from_report(data)
    ):
        if blocker not in blockers:
            blockers.append(blocker)
    return blockers


def _existing_hardware_selected_robot_mode(gate_run_root: Path) -> str | None:
    data, error = _load_json_object(gate_run_root / HARDWARE_STATUS_REPORT)
    if error or data is None:
        return None
    robot_status = data.get("robot_status")
    if not isinstance(robot_status, dict):
        return None
    selected_profile = robot_status.get("selected_profile")
    if not isinstance(selected_profile, dict):
        return None
    mode = selected_profile.get("mode")
    return str(mode) if mode is not None else None


def _rewrite_status_next_actions(
    root: Path,
    blocked_gates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not blocked_gates:
        return []

    gate = blocked_gates[0]
    gate_id = str(gate["gate_id"])
    gate_run_root = Path(str(gate.get("run_root") or root.as_posix()))
    blocker_names = [
        str(blocker["name"])
        for blocker in gate.get("next_blockers", [])
        if isinstance(blocker, dict) and blocker.get("name")
    ]
    run_root = gate_run_root.as_posix()
    actions: list[dict[str, Any]] = []

    if gate_id == FULL_CAPTURE_GATE_ID:
        has_sequence_plan = (gate_run_root / PIPELINE_SEQUENCE_PLAN).is_file()
        if "run_config" in blocker_names:
            actions.append(
                _action(
                    gate_id=gate_id,
                    label="Create real lab run config",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/create_run_config.py",
                        run_root,
                        "--sequence",
                        "real_full_capture_validation",
                        "--print-sequence-plan",
                    ],
                    reason=(
                        "The full-capture gate requires an intentional real "
                        "robot profile with enabled lab sensors and the saved "
                        "real full-capture validation sequence."
                    ),
                    blocks_on=["run_config"],
                )
            )
            if not has_sequence_plan:
                actions.append(
                    _action(
                        gate_id=gate_id,
                        label="Plan real full-capture validation sequence",
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/run_pipeline_sequence.py",
                            run_root,
                            "--sequence",
                            "real_full_capture_validation",
                            "--plan-only",
                        ],
                        reason=(
                            "Preview the real robot plus camera validation workflow "
                            "without starting hardware."
                        ),
                        blocks_on=["run_config"],
                    )
                )
            return actions
        if not has_sequence_plan:
            actions.append(
                _action(
                    gate_id=gate_id,
                    label="Plan real full-capture validation sequence",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_pipeline_sequence.py",
                        run_root,
                        "--sequence",
                        "real_full_capture_validation",
                        "--plan-only",
                    ],
                    reason=(
                        "Preview the real robot plus camera validation workflow "
                        "without starting hardware."
                    ),
                    blocks_on=blocker_names,
                )
            )
            return actions
        if "run_preflight" in blocker_names:
            return [
                _action(
                    gate_id=gate_id,
                    label="Write real run preflight",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_preflight.py",
                        run_root,
                        "--check",
                        "--write",
                    ],
                    reason=(
                        "Confirm the real run config, sequence, robot, sensor, "
                        "and runtime readiness before planning capture execution."
                    ),
                    blocks_on=["run_preflight"],
                )
            ]
        if "hardware_status" in blocker_names:
            selected_robot_mode = _existing_hardware_selected_robot_mode(
                gate_run_root
            )
            if selected_robot_mode is not None and selected_robot_mode != "real":
                return [
                    _action(
                        gate_id=gate_id,
                        label="Refresh hardware status from run config",
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/run_hardware_status_stage.py",
                            run_root,
                        ],
                        reason=(
                            "The latest hardware snapshot did not select the real "
                            "robot profile; refresh it so the run-scoped snapshot "
                            "uses the profile saved in "
                            "run_config.json before validating real full capture."
                        ),
                        blocks_on=["hardware_status"],
                    )
                ]
            hardware_blockers = _existing_hardware_status_blockers(gate_run_root)
            if any(name.startswith("sensor:") for name in hardware_blockers):
                return [
                    _action(
                        gate_id=gate_id,
                        label="Inspect sensor status",
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/sensor_status.py",
                            "--json",
                            "--check-expected",
                        ],
                        reason=(
                            "The latest hardware snapshot has sensor discovery "
                            "errors; inspect camera SDK/device visibility before "
                            "refreshing the full-capture gate."
                        ),
                        blocks_on=hardware_blockers,
                    ),
                    _action(
                        gate_id=gate_id,
                        label="Refresh hardware status after sensor fix",
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/run_hardware_status_stage.py",
                            run_root,
                        ],
                        reason=(
                            "After the lab host can see the expected cameras and "
                            "camera SDKs, refresh the run-scoped hardware snapshot "
                            "so the full-capture gate can advance to capture-plan "
                            "preflight."
                        ),
                        blocks_on=hardware_blockers,
                    ),
                ]
            return [
                _action(
                    gate_id=gate_id,
                    label="Write hardware status snapshot",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_hardware_status_stage.py",
                        run_root,
                    ],
                    reason=(
                        "Record read-only robot, sensor, and runtime readiness; "
                        "resolve any error status before starting full capture."
                    ),
                    blocks_on=["hardware_status"],
                )
            ]
        if "capture_plan" in blocker_names:
            return [
                _action(
                    gate_id=gate_id,
                    label="Write capture plan",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_capture_plan_stage.py",
                        run_root,
                    ],
                    reason=(
                        "Full capture execution starts from the saved "
                        "run_config.json command plan."
                    ),
                    blocks_on=["capture_plan"],
                )
            ]
        if "capture_plan_preflight" in blocker_names:
            return [
                _action(
                    gate_id=gate_id,
                    label="Preflight real capture plan",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_capture_plan_preflight.py",
                        run_root,
                        "--allow-real-robot",
                    ],
                    reason=(
                        "Check command shape, real-robot safety, scripts, and "
                        "sensor readiness before selecting commands for execution."
                    ),
                    blocks_on=["capture_plan_preflight"],
                )
            ]
        if "capture_execution_plan" in blocker_names:
            return [
                _action(
                    gate_id=gate_id,
                    label="Write full capture execution plan",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_capture_execution_plan.py",
                        run_root,
                        "--mode",
                        "full",
                        "--allow-cameras",
                        "--allow-real-robot",
                        "--include-sensors",
                    ],
                    reason=(
                        "Select real robot and camera commands explicitly "
                        "before process supervision."
                    ),
                    blocks_on=["capture_execution_plan"],
                )
            ]
        actions.append(
            _action(
                gate_id=gate_id,
                label="Run real full-capture validation sequence",
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/run_pipeline_sequence.py",
                    run_root,
                    "--sequence",
                    "real_full_capture_validation",
                ],
                reason=(
                    "Execute capture planning, real preflight, full supervised "
                    "capture, and the rewrite_full_capture.v1 audit in order."
                ),
                blocks_on=blocker_names,
            )
        )
        if "capture_execution" in blocker_names or any(
            name.startswith("sensor_frames:") for name in blocker_names
        ):
            actions.extend(
                [
                    _action(
                        gate_id=gate_id,
                        label="Run full supervised capture",
                        command=[
                            "uv",
                            "run",
                            "python",
                            "scripts/run_capture_execution_stage.py",
                            run_root,
                            "--mode",
                            "full",
                            "--allow-cameras",
                            "--allow-real-robot",
                            "--include-sensors",
                        ],
                        reason=(
                            "Produce the capture_execution_report.json, raw "
                            "robot poses, and raw RGB-D sensor folders required "
                            "by the full-capture gate."
                        ),
                        blocks_on=["capture_execution"],
                    ),
                ]
            )
        actions.append(
            _action(
                gate_id=gate_id,
                label="Audit full capture gate",
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/run_rewrite_gate.py",
                    run_root,
                    "--gate",
                    FULL_CAPTURE_GATE_ID,
                    "--write",
                ],
                reason="Confirm the real capture evidence satisfies the rewrite gate.",
            )
        )
        return actions

    if gate_id == CALIBRATION_VALIDATION_GATE_ID:
        if "calibration_validation" in blocker_names or "calibration_profiles" in blocker_names:
            actions.append(
                _action(
                    gate_id=gate_id,
                    label="Validate and promote calibration profiles",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_calibration_validation.py",
                        run_root,
                        "--promote",
                    ],
                    reason=(
                        "The calibration gate requires an ok validation report "
                        "and promoted valid calibration_profiles.json entries."
                    ),
                    blocks_on=blocker_names,
                )
            )
        actions.append(
            _action(
                gate_id=gate_id,
                label="Audit calibration validation gate",
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/run_rewrite_gate.py",
                    run_root,
                    "--gate",
                    CALIBRATION_VALIDATION_GATE_ID,
                    "--write",
                ],
                reason="Confirm promoted calibration profile evidence satisfies the gate.",
            )
        )
        return actions

    if gate_id == BOP_EXPORT_READINESS_GATE_ID:
        if "bop_export" in blocker_names or any(
            name.startswith("bop_scene:") for name in blocker_names
        ):
            actions.append(
                _action(
                    gate_id=gate_id,
                    label="Export BOP dataset",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_bop_export_stage.py",
                        run_root,
                        "--overwrite",
                    ],
                    reason=(
                        "The BOP readiness gate requires a structural BOP "
                        "dataset export with scene RGB/depth, camera, and GT files."
                    ),
                    blocks_on=blocker_names,
                )
            )
        if "bop_targets" in blocker_names or "bop_models_info" in blocker_names:
            actions.append(
                _action(
                    gate_id=gate_id,
                    label="Re-export BOP dataset with models",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_bop_export_stage.py",
                        run_root,
                        "--overwrite",
                    ],
                    reason=(
                        "Rebuild BOP targets and model metadata from the object "
                        "registry used by acquisition exports."
                    ),
                    blocks_on=blocker_names,
                )
            )
        actions.append(
            _action(
                gate_id=gate_id,
                label="Audit BOP export readiness gate",
                command=[
                    "uv",
                    "run",
                    "python",
                    "scripts/run_rewrite_gate.py",
                    run_root,
                    "--gate",
                    BOP_EXPORT_READINESS_GATE_ID,
                    "--write",
                ],
                reason="Confirm the exported BOP dataset satisfies the acquisition gate.",
            )
        )
        return actions

    return [
        _action(
            gate_id=gate_id,
            label="Audit rewrite gate",
            command=[
                "uv",
                "run",
                "python",
                "scripts/run_rewrite_gate.py",
                run_root,
                "--gate",
                gate_id,
                "--write",
            ],
            reason="Write the gate report and inspect its blockers.",
            blocks_on=blocker_names,
        )
    ]


def build_rewrite_status_report(
    run_root: str | Path,
    *,
    gate_ids: tuple[str, ...] = GATE_IDS,
    gate_run_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    explicit_gate_roots = {
        gate_id: Path(gate_run_roots[gate_id])
        for gate_id in gate_ids
        if gate_run_roots is not None and gate_id in gate_run_roots
    }
    gate_roots: dict[str, Path] = {}
    gate_reports: list[dict[str, Any]] = []
    for gate_id in gate_ids:
        gate_root = explicit_gate_roots.get(gate_id, root)
        gate_roots[gate_id] = gate_root
        report = build_gate_report(gate_root, gate_id=gate_id)
        gate_reports.append(report)
    ready_gates = [
        report for report in gate_reports if report["overall_status"] == "ready"
    ]
    blocked_gates = [
        report for report in gate_reports if report["overall_status"] != "ready"
    ]
    total_check_count = sum(
        int(report["summary"]["check_count"]) for report in gate_reports
    )
    total_ready_check_count = sum(
        int(report["summary"]["ready_count"]) for report in gate_reports
    )
    total_blocked_check_count = sum(
        int(report["summary"]["blocked_count"]) for report in gate_reports
    )
    next_gate = blocked_gates[0] if blocked_gates else None
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_root": root.as_posix(),
        "gate_run_roots": {
            gate_id: gate_roots.get(gate_id, root).as_posix()
            for gate_id in gate_ids
        },
        "overall_status": "ready" if not blocked_gates else "blocked",
        "summary": {
            "gate_count": len(gate_reports),
            "ready_gate_count": len(ready_gates),
            "blocked_gate_count": len(blocked_gates),
            "check_count": total_check_count,
            "ready_check_count": total_ready_check_count,
            "blocked_check_count": total_blocked_check_count,
        },
        "gates": [
            {
                "gate_id": report["gate_id"],
                "run_root": report["run_root"],
                "overall_status": report["overall_status"],
                "summary": report["summary"],
                "next_blockers": report["next_blockers"],
            }
            for report in gate_reports
        ],
        "next_gate": {
            "gate_id": next_gate["gate_id"],
            "run_root": next_gate["run_root"],
            "overall_status": next_gate["overall_status"],
            "summary": next_gate["summary"],
        }
        if next_gate is not None
        else None,
        "next_blockers": [
            {
                "gate_id": report["gate_id"],
                "name": blocker["name"],
                "artifact": blocker["artifact"],
                "message": blocker["message"],
                "details": blocker.get("details", {}),
            }
            for report in blocked_gates
            for blocker in report["next_blockers"][:3]
        ],
        "next_actions": _rewrite_status_next_actions(root, blocked_gates),
    }


def write_rewrite_status_report(
    run_root: str | Path,
    *,
    gate_ids: tuple[str, ...] = GATE_IDS,
    gate_run_roots: Mapping[str, str | Path] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(run_root)
    report = build_rewrite_status_report(
        root,
        gate_ids=gate_ids,
        gate_run_roots=gate_run_roots,
    )
    path = root / REWRITE_STATUS_REPORT
    atomic_write_json(path, report)
    return path, report
