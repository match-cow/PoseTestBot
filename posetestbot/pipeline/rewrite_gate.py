"""Milestone gates that keep the rewrite focused on proved workflows."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_FRAME_SETS,
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
    MULTIVIEW_FRAME_GROUPS,
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
    except UnicodeDecodeError as exc:
        return None, f"invalid_utf8: {exc.reason}"
    except OSError as exc:
        return None, f"unreadable: {exc}"
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


def _json_file_check(
    name: str, path: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
        if isinstance(sensor, dict) and sensor.get("enabled", True) is True
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


def _hardware_sync_execution_binding_error(value: Any) -> str | None:
    """Return why a capture-time hardware-sync binding is not exact."""

    if not isinstance(value, Mapping):
        return "binding is not an object"
    expected_fields = {
        "configuration_sha256",
        "qualification_artifact_sha256",
        "revalidated_immediately_before_receiver_spawn",
    }
    if set(value) != expected_fields:
        return "binding contains missing or unknown fields"
    if value.get("revalidated_immediately_before_receiver_spawn") is not True:
        return "binding does not prove immediate pre-receiver revalidation"
    for field in (
        "configuration_sha256",
        "qualification_artifact_sha256",
    ):
        digest = value.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return f"{field} is not a lowercase SHA-256 digest"
    return None


def _hardware_sync_bop_checks(
    root: Path,
    *,
    bop_export: Mapping[str, Any] | None,
    scene_images: Mapping[int, set[int]],
    scene_sensors: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Require complete-set provenance when the run claims hardware triggering."""

    run_config, _ = _load_json_object(root / RUN_CONFIG)
    capture = run_config.get("capture") if run_config is not None else None
    synchronization = (
        capture.get("synchronization")
        if isinstance(capture, Mapping)
        else None
    )
    if (
        not isinstance(synchronization, Mapping)
        or synchronization.get("mode") != "hardware_trigger"
    ):
        return []
    raw_configured_skew_ms = synchronization.get(
        "max_depth_timestamp_skew_ms"
    )
    expected_skew_ns: int | None = None
    if (
        not isinstance(raw_configured_skew_ms, bool)
        and isinstance(raw_configured_skew_ms, (int, float))
        and math.isfinite(float(raw_configured_skew_ms))
        and float(raw_configured_skew_ms) > 0
    ):
        expected_skew_ns = int(
            round(float(raw_configured_skew_ms) * 1_000_000)
        )

    source_path = (
        root / PROCESSED_DIR / "synchronized" / MULTIVIEW_FRAME_GROUPS
    )
    frame_sets_path = root / BOP_DIR / BOP_FRAME_SETS
    frame_map_path = root / BOP_DIR / BOP_FRAME_MAP_JSON
    source_groups, source_error = _load_json_object(source_path)
    frame_sets, frame_sets_error = _load_json_object(frame_sets_path)
    frame_map, frame_map_error = _load_json_object(frame_map_path)
    capture_report, capture_report_error = _load_json_object(
        root / CAPTURE_EXECUTION_REPORT
    )
    mapped_scenes = (
        frame_map.get("scenes")
        if isinstance(frame_map, Mapping)
        else None
    )

    source_rows = (
        source_groups.get("groups")
        if isinstance(source_groups, Mapping)
        else None
    )
    source_validation_error: str | None = None
    source_valid = False
    if isinstance(source_groups, Mapping):
        try:
            from posetestbot.sync.hardware import (
                validate_hardware_sync_frame_groups,
            )

            validate_hardware_sync_frame_groups(
                source_groups,
                run_root=root,
            )
            source_valid = True
        except (KeyError, OSError, TypeError, ValueError) as exc:
            source_validation_error = str(exc)
    source_row_items = source_rows if isinstance(source_rows, list) else []
    source_by_id = {
        str(group.get("frame_group_id")): group
        for group in source_row_items
        if isinstance(group, Mapping) and group.get("frame_group_id")
    }
    frame_set_rows = (
        frame_sets.get("frame_sets")
        if isinstance(frame_sets, Mapping)
        else None
    )
    source_ids = {
        str(group.get("frame_group_id"))
        for group in source_row_items
        if isinstance(group, Mapping) and group.get("frame_group_id")
    }
    frame_set_ids: set[str] = set()
    references_ok = isinstance(frame_set_rows, list)
    sensor_order = (
        source_groups.get("sensor_order")
        if isinstance(source_groups, Mapping)
        else None
    )
    expected_sensor_keys = (
        set(sensor_order)
        if isinstance(sensor_order, list)
        and all(isinstance(item, str) for item in sensor_order)
        else set()
    )
    expected_sensor_order = (
        list(sensor_order)
        if isinstance(sensor_order, list)
        and all(isinstance(item, str) for item in sensor_order)
        else []
    )
    projection_truth_errors: list[str] = []
    projection_truth_by_scene: dict[int, dict[str, str]] = {}
    exports_by_scene: dict[int, Mapping[str, Any]] = {}
    raw_exports = (
        bop_export.get("exports")
        if isinstance(bop_export, Mapping)
        else None
    )
    if isinstance(raw_exports, list):
        for export in raw_exports:
            if not isinstance(export, Mapping):
                references_ok = False
                continue
            try:
                raw_scene_id = export["scene_id"]
                if isinstance(raw_scene_id, bool) or not isinstance(
                    raw_scene_id, int
                ):
                    raise TypeError
                scene_id = raw_scene_id
            except (KeyError, TypeError, ValueError):
                references_ok = False
                continue
            if scene_id in exports_by_scene:
                references_ok = False
                continue
            exports_by_scene[scene_id] = export

    def projection_truth(
        *,
        scene_id: int,
        export: Mapping[str, Any] | None,
        authoritative_sensor_folder: str,
    ) -> dict[str, str] | None:
        """Recompute the exact native/rectified source represented by a scene."""

        cached = projection_truth_by_scene.get(scene_id)
        if cached is not None:
            return cached
        if not source_valid or not isinstance(export, Mapping):
            return None
        projection = export.get("projection")
        if projection not in {"native", "rectified"}:
            projection_truth_errors.append(
                f"scene {scene_id} has unsupported projection {projection!r}"
            )
            return None
        sensor_name = Path(authoritative_sensor_folder).name
        expected_authoritative = (
            Path(PROCESSED_DIR) / "synchronized" / sensor_name
        ).as_posix()
        if authoritative_sensor_folder != expected_authoritative:
            projection_truth_errors.append(
                f"scene {scene_id} authoritative sensor folder is not canonical"
            )
            return None
        expected_input = (
            expected_authoritative
            if projection == "native"
            else (Path(PROCESSED_DIR) / "rectified" / sensor_name).as_posix()
        )
        authoritative_path = root / expected_authoritative
        input_path = root / expected_input
        try:
            from posetestbot.calibration.rectification import (
                rgbd_camera_artifact_fingerprint,
                validate_rectification_provenance,
            )

            if projection == "rectified":
                rectification = validate_rectification_provenance(
                    authoritative_path,
                    input_path,
                )
                authoritative_fingerprint = rectification[
                    "source_fingerprint"
                ]
                input_fingerprint = rectification["output_fingerprint"]
            else:
                authoritative_fingerprint = rgbd_camera_artifact_fingerprint(
                    authoritative_path
                )
                input_fingerprint = authoritative_fingerprint
            authoritative_digest = authoritative_fingerprint.get("digest")
            input_digest = input_fingerprint.get("digest")
            if (
                not isinstance(authoritative_digest, str)
                or len(authoritative_digest) != 64
                or not isinstance(input_digest, str)
                or len(input_digest) != 64
            ):
                raise ValueError("source fingerprint digest is invalid")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            projection_truth_errors.append(
                f"scene {scene_id} source truth is invalid: {exc}"
            )
            return None
        truth = {
            "projection": projection,
            "input_sensor_folder": expected_input,
            "authoritative_source_sensor_folder": expected_authoritative,
            "input_fingerprint_sha256": input_digest,
            "authoritative_source_fingerprint_sha256": (
                authoritative_digest
            ),
        }
        projection_truth_by_scene[scene_id] = truth
        return truth

    if isinstance(frame_set_rows, list):
        for expected_index, frame_set in enumerate(frame_set_rows):
            if not isinstance(frame_set, Mapping):
                references_ok = False
                continue
            frame_set_id = str(frame_set.get("frame_set_id") or "")
            if (
                not frame_set_id
                or frame_set_id in frame_set_ids
                or frame_set.get("frame_set_index") != expected_index
            ):
                references_ok = False
            frame_set_ids.add(frame_set_id)
            source_group = source_by_id.get(frame_set_id)
            if not isinstance(source_group, Mapping):
                references_ok = False
            else:
                expected_group_values = {
                    "capture_group_id": source_group.get("capture_group_id"),
                    "master_sensor_key": source_group.get("master_sensor_key"),
                    "depth_sensor_timestamp_ns": source_group.get(
                        "depth_sensor_timestamp_ns"
                    ),
                    "max_abs_depth_timestamp_skew_ns": source_group.get(
                        "max_abs_depth_timestamp_skew_ns"
                    ),
                    "depth_timestamp_span_ns": source_group.get(
                        "depth_timestamp_span_ns"
                    ),
                    "matched_robot_pose": source_group.get(
                        "matched_robot_pose"
                    ),
                }
                if any(
                    frame_set.get(field) != expected
                    for field, expected in expected_group_values.items()
                ):
                    references_ok = False
            views = frame_set.get("views")
            if not isinstance(views, list) or len(views) != len(expected_sensor_keys):
                references_ok = False
                continue
            view_sensor_keys: set[str] = set()
            ordered_view_sensor_keys: list[str] = []
            for view in views:
                if not isinstance(view, Mapping):
                    references_ok = False
                    continue
                sensor_key = str(view.get("sensor_key") or "")
                try:
                    raw_scene_id = view["scene_id"]
                    raw_im_id = view["im_id"]
                    if (
                        isinstance(raw_scene_id, bool)
                        or not isinstance(raw_scene_id, int)
                        or isinstance(raw_im_id, bool)
                        or not isinstance(raw_im_id, int)
                    ):
                        raise TypeError
                    scene_id = raw_scene_id
                    im_id = raw_im_id
                except (KeyError, TypeError, ValueError):
                    references_ok = False
                    continue
                view_sensor_keys.add(sensor_key)
                ordered_view_sensor_keys.append(sensor_key)
                source_frames = (
                    source_group.get("frames")
                    if isinstance(source_group, Mapping)
                    else None
                )
                source_frame = (
                    source_frames.get(sensor_key)
                    if isinstance(source_frames, Mapping)
                    else None
                )
                if not isinstance(source_frame, Mapping):
                    references_ok = False
                else:
                    provenance_fields = (
                        "sensor_folder",
                        "mounting_mode",
                        "hardware_sync_role",
                        "source_frame_index",
                        "source_frame_id",
                        "source_sensor_folder",
                        "source_rgb_path",
                        "source_depth_path",
                        "synchronized_frame_index",
                        "synchronized_frame_id",
                        "synchronized_rgb_path",
                        "synchronized_depth_path",
                        "depth_sensor_timestamp_ns",
                        "depth_frame_number",
                        "depth_timestamp_domain",
                        "depth_timestamp_skew_ns",
                        "abs_depth_timestamp_skew_ns",
                        "matched_robot_pose",
                    )
                    if any(
                        view.get(field) != source_frame.get(field)
                        for field in provenance_fields
                    ):
                        references_ok = False
                if (
                    im_id not in scene_images.get(scene_id, set())
                    or view.get("sensor_name") != scene_sensors.get(scene_id)
                ):
                    references_ok = False
                expected_sensor_name = (
                    Path(str(source_frame.get("sensor_folder") or "")).name
                    if isinstance(source_frame, Mapping)
                    else ""
                )
                scene_map = (
                    mapped_scenes.get(str(scene_id))
                    if isinstance(mapped_scenes, Mapping)
                    else None
                )
                mapped_frames = (
                    scene_map.get("frames")
                    if isinstance(scene_map, Mapping)
                    else None
                )
                mapped_frame = (
                    mapped_frames.get(str(im_id))
                    if isinstance(mapped_frames, Mapping)
                    else None
                )
                export = exports_by_scene.get(scene_id)
                expected_scene_folder = (
                    str(export.get("scene_folder") or "")
                    if isinstance(export, Mapping)
                    else ""
                )
                expected_bop_rgb = (
                    (
                        Path(expected_scene_folder)
                        / RGB_DIR
                        / f"{im_id:06d}.png"
                    ).as_posix()
                    if expected_scene_folder
                    else ""
                )
                expected_bop_depth = (
                    (
                        Path(expected_scene_folder)
                        / DEPTH_DIR
                        / f"{im_id:06d}.png"
                    ).as_posix()
                    if expected_scene_folder
                    else ""
                )
                truth = (
                    projection_truth(
                        scene_id=scene_id,
                        export=export,
                        authoritative_sensor_folder=str(
                            source_frame.get("sensor_folder") or ""
                        ),
                    )
                    if isinstance(source_frame, Mapping)
                    else None
                )
                truth_fields_match = (
                    truth is not None
                    and isinstance(export, Mapping)
                    and isinstance(scene_map, Mapping)
                    and all(
                        export.get(field) == expected
                        and scene_map.get(field) == expected
                        for field, expected in truth.items()
                    )
                    and view.get("projection") == truth["projection"]
                    and view.get("bop_input_sensor_folder")
                    == truth["input_sensor_folder"]
                    and view.get("authoritative_source_sensor_folder")
                    == truth["authoritative_source_sensor_folder"]
                    and view.get("bop_input_fingerprint_sha256")
                    == truth["input_fingerprint_sha256"]
                    and view.get(
                        "authoritative_source_fingerprint_sha256"
                    )
                    == truth[
                        "authoritative_source_fingerprint_sha256"
                    ]
                )
                if (
                    not isinstance(source_frame, Mapping)
                    or not isinstance(scene_map, Mapping)
                    or not isinstance(mapped_frame, Mapping)
                    or not expected_sensor_name
                    or view.get("sensor_name") != expected_sensor_name
                    or scene_map.get("sensor_name") != expected_sensor_name
                    or scene_map.get("scene_folder") != expected_scene_folder
                    or not truth_fields_match
                    or mapped_frame.get("sensor_name") != expected_sensor_name
                    or mapped_frame.get("scene_id") != scene_id
                    or mapped_frame.get("projection")
                    != (truth or {}).get("projection")
                    or mapped_frame.get("input_sensor_folder")
                    != (truth or {}).get("input_sensor_folder")
                    or mapped_frame.get(
                        "authoritative_source_sensor_folder"
                    )
                    != (truth or {}).get(
                        "authoritative_source_sensor_folder"
                    )
                    or mapped_frame.get("source_rgb")
                    != view.get("bop_input_rgb_path")
                    or mapped_frame.get("source_depth")
                    != view.get("bop_input_depth_path")
                    or mapped_frame.get("authoritative_source_rgb")
                    != source_frame.get("synchronized_rgb_path")
                    or mapped_frame.get("authoritative_source_depth")
                    != source_frame.get("synchronized_depth_path")
                    or view.get("authoritative_source_rgb_path")
                    != source_frame.get("synchronized_rgb_path")
                    or view.get("authoritative_source_depth_path")
                    != source_frame.get("synchronized_depth_path")
                    or mapped_frame.get("bop_rgb")
                    != f"{RGB_DIR}/{im_id:06d}.png"
                    or mapped_frame.get("bop_depth")
                    != f"{DEPTH_DIR}/{im_id:06d}.png"
                    or view.get("bop_rgb") != expected_bop_rgb
                    or view.get("bop_depth") != expected_bop_depth
                ):
                    references_ok = False
            if view_sensor_keys != expected_sensor_keys:
                references_ok = False
            if ordered_view_sensor_keys != expected_sensor_order:
                references_ok = False

    claims = (
        frame_sets.get("synchronization_claims")
        if isinstance(frame_sets, Mapping)
        else None
    )
    configuration_matches = (
        source_groups is not None
        and frame_sets is not None
        and all(
            source_groups.get(key) == synchronization.get(key)
            and frame_sets.get(key) == synchronization.get(key)
            for key in (
                "group_id",
                "implementation",
                "scope",
                "master_sensor_key",
            )
        )
        and source_groups.get("max_depth_timestamp_skew_ns")
        == expected_skew_ns
        and frame_sets.get("max_depth_timestamp_skew_ns")
        == source_groups.get("max_depth_timestamp_skew_ns")
    )
    source_sensor_inventory = (
        source_groups.get("sensors")
        if isinstance(source_groups, Mapping)
        else None
    )
    expected_config_sensors: list[dict[str, Any]] = []
    if isinstance(capture, Mapping):
        raw_config_sensors = capture.get("sensors")
        if isinstance(raw_config_sensors, list):
            for sensor in raw_config_sensors:
                if (
                    not isinstance(sensor, Mapping)
                    or sensor.get("enabled", True) is not True
                ):
                    continue
                sensor_type = str(sensor.get("sensor_type") or "")
                device_id = str(sensor.get("device_id") or "")
                sensor_key = f"{sensor_type}:{device_id}"
                expected_config_sensors.append(
                    {
                        "sensor_key": sensor_key,
                        "sensor_type": sensor_type,
                        "device_id": device_id,
                        "sensor_folder": (
                            Path(PROCESSED_DIR)
                            / "synchronized"
                            / _sensor_folder_name(dict(sensor))
                        ).as_posix(),
                        "mounting_mode": sensor.get("mounting_mode"),
                        "hardware_sync_role": (
                            "master"
                            if sensor_key
                            == synchronization.get("master_sensor_key")
                            else "subordinate"
                        ),
                    }
                )
    expected_config_sensors.sort(
        key=lambda sensor: (
            0 if sensor["hardware_sync_role"] == "master" else 1,
            next(
                (
                    index
                    for index, configured in enumerate(
                        capture.get("sensors", [])
                        if isinstance(capture, Mapping)
                        and isinstance(capture.get("sensors"), list)
                        else []
                    )
                    if isinstance(configured, Mapping)
                    and configured.get("sensor_type") == sensor["sensor_type"]
                    and configured.get("device_id") == sensor["device_id"]
                ),
                0,
            ),
        )
    )
    inventory_matches = (
        isinstance(source_sensor_inventory, list)
        and len(source_sensor_inventory) == len(expected_config_sensors)
        and [
            {
                key: sensor.get(key)
                for key in (
                    "sensor_key",
                    "sensor_type",
                    "device_id",
                    "sensor_folder",
                    "mounting_mode",
                    "hardware_sync_role",
                )
            }
            for sensor in source_sensor_inventory
            if isinstance(sensor, Mapping)
        ]
        == expected_config_sensors
        and expected_sensor_order
        == [sensor["sensor_key"] for sensor in expected_config_sensors]
    )
    frame_set_top_level_matches = (
        isinstance(frame_sets, Mapping)
        and isinstance(source_groups, Mapping)
        and frame_sets.get("source_schema_version")
        == source_groups.get("schema_version")
        and frame_sets.get("sensor_order") == expected_sensor_order
        and frame_sets.get("frame_set_count")
        == (len(frame_set_rows) if isinstance(frame_set_rows, list) else -1)
        and frame_sets.get("frame_set_count")
        == (len(source_rows) if isinstance(source_rows, list) else -2)
    )
    frame_map_matches = (
        isinstance(frame_map, Mapping)
        and frame_map.get("schema_version") == "posetestbot_bop_frame_map.v2"
        and isinstance(mapped_scenes, Mapping)
        and set(mapped_scenes)
        == {str(scene_id) for scene_id in scene_images}
    )
    qualification_matches = False
    qualification_error: str | None = None
    current_qualification: Mapping[str, Any] | None = None
    if isinstance(run_config, Mapping):
        try:
            from posetestbot.sensors.hardware_sync_qualification import (
                validate_hardware_sync_qualification,
            )

            current_qualification = validate_hardware_sync_qualification(
                root,
                run_config=run_config,
            )
            qualification_matches = (
                isinstance(source_groups, Mapping)
                and isinstance(frame_sets, Mapping)
                and source_groups.get("hardware_sync_qualification")
                == current_qualification
                and frame_sets.get("hardware_sync_qualification")
                == current_qualification
            )
            if not qualification_matches:
                qualification_error = (
                    "qualification provenance differs between the current "
                    "run, hardware groups, and BOP frame sets"
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            qualification_error = str(exc)
    source_execution_binding = (
        source_groups.get("hardware_sync_execution_binding")
        if isinstance(source_groups, Mapping)
        else None
    )
    frame_set_execution_binding = (
        frame_sets.get("hardware_sync_execution_binding")
        if isinstance(frame_sets, Mapping)
        else None
    )
    capture_execution_binding = (
        capture_report.get("hardware_sync_execution_binding")
        if isinstance(capture_report, Mapping)
        else None
    )
    execution_binding_errors = {
        name: error
        for name, error in (
            (
                "source_groups",
                _hardware_sync_execution_binding_error(
                    source_execution_binding
                ),
            ),
            (
                "frame_sets",
                _hardware_sync_execution_binding_error(
                    frame_set_execution_binding
                ),
            ),
            (
                "capture_execution_report",
                _hardware_sync_execution_binding_error(
                    capture_execution_binding
                ),
            ),
        )
        if error is not None
    }
    execution_binding_matches = (
        not execution_binding_errors
        and capture_report_error is None
        and source_execution_binding == frame_set_execution_binding
        and source_execution_binding == capture_execution_binding
    )
    validation = (
        bop_export.get("validation")
        if isinstance(bop_export, Mapping)
        else None
    )
    ok = (
        isinstance(source_groups, Mapping)
        and source_valid
        and source_groups.get("schema_version")
        == "hardware_sync_frame_groups.v1"
        and isinstance(source_rows, list)
        and bool(source_rows)
        and isinstance(frame_sets, Mapping)
        and frame_sets.get("schema_version") == "posetestbot_frame_sets.v1"
        and isinstance(frame_set_rows, list)
        and bool(frame_set_rows)
        and source_ids == frame_set_ids
        and references_ok
        and configuration_matches
        and inventory_matches
        and frame_set_top_level_matches
        and frame_map_matches
        and qualification_matches
        and execution_binding_matches
        and not projection_truth_errors
        and claims
        == {
            "depth_exposure_hardware_synchronized": True,
            "rgb_exposure_hardware_synchronized": False,
            "rgb_association": "same_device_frameset_timestamp_association",
            "synthetic_robot_occlusion_modeled": False,
        }
        and isinstance(bop_export, Mapping)
        and bop_export.get("schema_version") == "bop_export_manifest.v4"
        and bop_export.get("frame_map_path") == BOP_FRAME_MAP_JSON
        and bop_export.get("frame_sets_path") == BOP_FRAME_SETS
        and isinstance(validation, Mapping)
        and validation.get("frame_set_count") == len(frame_set_rows)
        and validation.get("hardware_sync_scope")
        == synchronization.get("scope")
    )
    return [
        _check(
            name="bop_hardware_sync_frame_sets",
            path=frame_sets_path,
            ok=ok,
            message=(
                "BOP frame sets exactly map every authoritative complete "
                "hardware-sync group to exported scenes and images."
                if ok
                else (
                    "Hardware-triggered runs require matching non-empty "
                    f"{MULTIVIEW_FRAME_GROUPS} and {BOP_FRAME_SETS} provenance; "
                    f"source={source_error}, frame_sets={frame_sets_error}."
                )
            ),
            details={
                "source_group_count": (
                    len(source_rows) if isinstance(source_rows, list) else 0
                ),
                "frame_set_count": (
                    len(frame_set_rows)
                    if isinstance(frame_set_rows, list)
                    else 0
                ),
                "references_ok": references_ok,
                "configuration_matches": configuration_matches,
                "inventory_matches": inventory_matches,
                "frame_set_top_level_matches": frame_set_top_level_matches,
                "frame_map_matches": frame_map_matches,
                "frame_map_error": frame_map_error,
                "source_validation_error": source_validation_error,
                "qualification_matches": qualification_matches,
                "qualification_error": qualification_error,
                "execution_binding_matches": execution_binding_matches,
                "execution_binding_errors": execution_binding_errors,
                "capture_execution_report_error": capture_report_error,
                "projection_truth_errors": projection_truth_errors,
                "source_group_ids_match_frame_set_ids": source_ids
                == frame_set_ids,
            },
        )
    ]


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
        exports = (
            [export for export in raw_exports if isinstance(export, Mapping)]
            if isinstance(raw_exports, list)
            else []
        )
        validation = bop_export.get("validation")
        validation_ok = (
            isinstance(validation, Mapping) and validation.get("status") == "ok"
        )
        export_schema = bop_export.get("schema_version")
        ok = (
            export_schema
            in {
                "bop_export_manifest.v2",
                "bop_export_manifest.v3",
                "bop_export_manifest.v4",
            }
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
                    "bop_export_manifest.json must be v2/v3/v4, BOP-scenewise, contain "
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
            and pose_template.get("template_uuid")
            == template_summary.get("template_uuid")
            and pose_template.get("bundle_sha256")
            == template_summary.get("bundle_sha256")
            and template_selection is not None
            and template_selection.get("schema_version") == "pose_template_selection.v1"
            and template_selection.get("placement_confirmed") is True
            and pose_template.get("template_uuid")
            == template_selection.get("template_uuid")
            and pose_template.get("bundle_sha256")
            == template_selection.get("bundle_sha256")
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
        scene_camera, camera_error = _load_json_object(
            scene_folder / "scene_camera.json"
        )
        scene_gt, gt_error = _load_json_object(scene_folder / "scene_gt.json")
        scene_gt_info, info_error = _load_json_object(
            scene_folder / "scene_gt_info.json"
        )
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
            any(
                scene_objects.get((scene_id, image_id), set()) for image_id in image_ids
            )
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
    checks.extend(
        _hardware_sync_bop_checks(
            root,
            bop_export=bop_export,
            scene_images=scene_images,
            scene_sensors=scene_sensors,
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
                    if image_id not in scene_images.get(
                        scene_id, set()
                    ) or obj_id not in scene_objects.get((scene_id, image_id), set()):
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
        map_rows = (
            template_instance_map.get("instances") if template_instance_map else None
        )
        selection_digest_ok = (
            selection_path.is_file()
            and template_object_instances.get("selection_sha256")
            == hashlib.sha256(selection_path.read_bytes()).hexdigest()
        )
        object_by_uuid = (
            {
                str(item.get("instance_uuid")): item
                for item in object_rows or []
                if isinstance(item, Mapping)
            }
            if isinstance(object_rows, list)
            else {}
        )
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
                    (
                        str(item["instance_uuid"]),
                        str(item["catalog_uuid"]),
                        int(item["obj_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                render_ok = False
        for sensor_name in scene_sensors.values():
            candidates = [
                root
                / PROCESSED_DIR
                / "rectified"
                / sensor_name
                / "blenderproc"
                / "output"
                / "posetestbot_render_instances.json",
                root
                / PROCESSED_DIR
                / "synchronized"
                / sensor_name
                / "blenderproc"
                / "output"
                / "posetestbot_render_instances.json",
            ]
            sidecar_path = next(
                (path for path in candidates if path.is_file()), candidates[-1]
            )
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
            render_details[sensor_name] = (
                "ok" if sensor_ok else str(sidecar_error or "mismatch")
            )

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
                geometry = (
                    model.get("posetestbot_geometry")
                    if isinstance(model, Mapping)
                    else None
                )
                if (
                    not isinstance(geometry, Mapping)
                    or geometry.get("source_sha256") != digest
                ):
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
            isinstance(profile_id, str) and profile_statuses.get(profile_id) == "valid"
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
            selected_profile.get("mode") if isinstance(selected_profile, dict) else None
        )
        status_ok = status in {"ok", "warning", "ready", "succeeded"}
        robot_mode_ok = selected_robot_mode == "real"
        ok = status_ok and robot_mode_ok
        if ok:
            message = (
                "hardware_status_report.json has acceptable status and real robot mode."
            )
        elif not robot_mode_ok:
            message = "hardware_status_report.json must select the real robot profile."
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

    capture_plan_preflight_path = root / CAPTURE_PLAN_PREFLIGHT_REPORT
    capture_plan_preflight, capture_plan_preflight_error = _load_json_object(
        capture_plan_preflight_path
    )
    capture_plan_preflight_source = CAPTURE_PLAN_PREFLIGHT_REPORT
    if capture_plan_preflight_error == "missing":
        embedded_plan, embedded_plan_error = _load_json_object(
            root / CAPTURE_EXECUTION_PLAN
        )
        embedded_preflight = (
            embedded_plan.get("preflight_report")
            if embedded_plan_error is None and embedded_plan is not None
            else None
        )
        if (
            isinstance(embedded_preflight, dict)
            and embedded_preflight.get("schema_version")
            == "capture_plan_preflight.v1"
            and embedded_plan.get("preflight_status")
            == _status_value(embedded_preflight)
        ):
            capture_plan_preflight = embedded_preflight
            capture_plan_preflight_error = None
            capture_plan_preflight_path = root / CAPTURE_EXECUTION_PLAN
            capture_plan_preflight_source = (
                f"{CAPTURE_EXECUTION_PLAN}:preflight_report"
            )
    if capture_plan_preflight_error is not None:
        check = _check(
            name="capture_plan_preflight",
            path=capture_plan_preflight_path,
            ok=False,
            message=(
                f"{CAPTURE_PLAN_PREFLIGHT_REPORT} is "
                f"{capture_plan_preflight_error}."
            ),
        )
    else:
        check = _check(
            name="capture_plan_preflight",
            path=capture_plan_preflight_path,
            ok=True,
            message="Capture-plan preflight evidence exists and is valid JSON.",
            details={"source": capture_plan_preflight_source},
        )
    if capture_plan_preflight is not None:
        status = _status_value(capture_plan_preflight)
        ok = status in {"ok", "warning", "ready", "succeeded"}
        check = _check(
            name="capture_plan_preflight",
            path=capture_plan_preflight_path,
            ok=ok,
            message=(
                "Capture-plan preflight evidence has acceptable status."
                if ok
                else "Capture-plan preflight evidence must be ok or warning."
            ),
            details={
                "status": status,
                "source": capture_plan_preflight_source,
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
        sensor_processes = (
            [
                process
                for process in processes
                if isinstance(process, dict) and process.get("role") == "sensor_capture"
            ]
            if isinstance(processes, list)
            else []
        )
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
    _run_config_check, run_config = _json_file_check("run_config", root / RUN_CONFIG)
    enabled_sensors = (
        _enabled_run_config_sensors(run_config) if run_config is not None else []
    )
    configured_sensors = (
        [
            sensor
            for sensor in run_config.get("capture", {}).get("sensors", [])
            if isinstance(sensor, dict)
        ]
        if run_config is not None and isinstance(run_config.get("capture"), dict)
        else []
    )
    enabled_identities = {
        (str(sensor.get("sensor_type") or ""), str(sensor.get("device_id") or ""))
        for sensor in enabled_sensors
    }
    disabled_only_identities = {
        (str(sensor.get("sensor_type") or ""), str(sensor.get("device_id") or ""))
        for sensor in configured_sensors
        if sensor.get("enabled", True) is not True
    } - enabled_identities

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
        promotable_profile_count = int(validation.get("promotable_profile_count") or 0)
        promotion_requested = (
            bool(promotion.get("requested")) if isinstance(promotion, dict) else False
        )
        promotion_promoted = (
            bool(promotion.get("promoted")) if isinstance(promotion, dict) else False
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
                        "mounting_mode": profile.get("mounting_mode"),
                        "status": profile.get("status"),
                        "num_inliers": quality.get("num_inliers"),
                        "residual_translation_mm": quality.get(
                            "residual_translation_mm"
                        ),
                        "residual_rotation_deg": quality.get("residual_rotation_deg"),
                    }
                )
        ignored_disabled_profiles = [
            profile
            for profile in profile_summaries
            if (str(profile["sensor_type"] or ""), str(profile["sensor_id"] or ""))
            in disabled_only_identities
        ]
        validated_profile_summaries = [
            profile
            for profile in profile_summaries
            if profile not in ignored_disabled_profiles
        ]
        all_profiles_valid = bool(validated_profile_summaries) and all(
            profile["status"] == "valid"
            and isinstance(profile["num_inliers"], int)
            and profile["num_inliers"] > 0
            and profile["residual_translation_mm"] is not None
            and profile["residual_rotation_deg"] is not None
            for profile in validated_profile_summaries
        )
        collection_profile_ids = {
            profile["profile_id"]
            for profile in profile_summaries
            if isinstance(profile["profile_id"], str)
        }
        promoted_profiles_present = bool(promoted_profile_ids) and (
            promoted_profile_ids <= collection_profile_ids
        )
        collection_count_matches_promotion = promoted_profile_count == len(
            profile_summaries
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
                "validated_profile_ids": [
                    profile["profile_id"] for profile in validated_profile_summaries
                ],
                "ignored_disabled_profile_ids": [
                    profile["profile_id"] for profile in ignored_disabled_profiles
                ],
            },
        )
    checks.append(check)

    coverage: list[dict[str, Any]] = []
    if profile_collection is not None and isinstance(
        profile_collection.get("profiles"), list
    ):
        raw_profiles = [
            profile
            for profile in profile_collection["profiles"]
            if isinstance(profile, dict)
        ]
        for sensor in enabled_sensors:
            sensor_type = str(sensor.get("sensor_type") or "")
            device_id = str(sensor.get("device_id") or "")
            mounting_mode = str(sensor.get("mounting_mode") or "")
            configured_profile_id = sensor.get("calibration_profile_id")
            matching_profile_ids = [
                str(profile.get("profile_id"))
                for profile in raw_profiles
                if profile.get("status") == "valid"
                and str(profile.get("sensor_type") or "") == sensor_type
                and str(profile.get("sensor_id") or "") == device_id
                and str(profile.get("mounting_mode") or "") == mounting_mode
                and (
                    not configured_profile_id
                    or str(profile.get("profile_id") or "")
                    == str(configured_profile_id)
                )
            ]
            coverage.append(
                {
                    "sensor_type": sensor_type,
                    "device_id": device_id,
                    "mounting_mode": mounting_mode,
                    "configured_profile_id": configured_profile_id,
                    "matching_profile_ids": matching_profile_ids,
                    "exact_match": len(matching_profile_ids) == 1,
                }
            )
    coverage_ok = (
        bool(enabled_sensors)
        and len(coverage) == len(enabled_sensors)
        and all(item["exact_match"] for item in coverage)
    )
    checks.append(
        _check(
            name="calibration_profile_sensor_coverage",
            path=root / CALIBRATION_PROFILES,
            ok=coverage_ok,
            message=(
                "Every enabled run-config sensor has exactly one valid identity- and mounting-matched profile."
                if coverage_ok
                else (
                    "Every enabled run-config sensor must have exactly one valid "
                    "profile matching sensor type, device ID, mounting mode, and "
                    "any configured profile ID."
                )
            ),
            details={
                "run_config_path": (root / RUN_CONFIG).as_posix(),
                "enabled_sensor_count": len(enabled_sensors),
                "covered_sensor_count": sum(
                    1 for item in coverage if item["exact_match"]
                ),
                "sensors": coverage,
            },
        )
    )

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


def write_gate_report(
    run_root: str | Path, *, gate_id: str
) -> tuple[Path, dict[str, Any]]:
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
            selected_robot_mode = _existing_hardware_selected_robot_mode(gate_run_root)
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
        if (
            "calibration_validation" in blocker_names
            or "calibration_profiles" in blocker_names
        ):
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
                    label="Re-export BOP targets and model metadata",
                    command=[
                        "uv",
                        "run",
                        "python",
                        "scripts/run_bop_export_stage.py",
                        run_root,
                        "--overwrite",
                    ],
                    reason=(
                        "Rebuild BOP targets and model metadata from the run's "
                        "immutable pose-template selection or objectless contract."
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
            gate_id: gate_roots.get(gate_id, root).as_posix() for gate_id in gate_ids
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
