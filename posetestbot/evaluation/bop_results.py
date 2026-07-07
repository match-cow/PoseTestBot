"""Export estimator outputs into BOP-compatible pose result CSV files."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import cv2 as cv

from posetestbot.evaluation.bop_toolkit import BOP19_RESULT_HEADER
from posetestbot.io.artifacts import (
    ARUCO_POSE_ESTIMATION,
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_RESULT_EXPORT_MANIFEST,
    MODELS_DIR,
    RESULTS_DIR,
)


SCHEMA_VERSION = "bop_result_export_manifest.v1"


@dataclass(frozen=True)
class BopExportIndex:
    bop_root: str
    sensor_scenes: dict[str, dict[str, object]]
    object_ids_by_name: dict[str, int]


@dataclass(frozen=True)
class FoundationPoseOutput:
    output_folder: str
    sensor_folder: str
    sensor_name: str
    method: str
    result_id: str | None
    object_index: int | None
    object_name: str
    obj_id: int


@dataclass(frozen=True)
class ArucoOutput:
    path: str
    sensor_folder: str
    sensor_name: str
    object_name: str
    obj_id: int


@dataclass(frozen=True)
class EstimatorOutput:
    output_folder: str
    sensor_folder: str
    sensor_name: str
    method: str
    result_id: str | None
    object_index: int | None
    object_name: str
    obj_id: int


@dataclass(frozen=True)
class BopResultRow:
    scene_id: int
    im_id: int
    obj_id: int
    score: float
    R: list[float]
    t: list[float]
    time: float
    source_pose_file: str


@dataclass(frozen=True)
class BopResultFile:
    path: str
    filename: str
    method: str
    dataset: str
    split: str
    result_id: str | None
    row_count: int
    source_outputs: list[str]


@dataclass(frozen=True)
class BopResultExportManifest:
    schema_version: str
    run_root: str
    bop_root: str
    input_folder: str
    output_folder: str
    dataset_name: str
    source_type: str
    translation_scale_to_mm: float
    results: list[BopResultFile]

    def to_dict(self) -> dict:
        return asdict(self)


def _load_json(path: Path) -> object:
    with open(path, "r") as f:
        return json.load(f)


def load_bop_export_index(bop_root: str | Path) -> BopExportIndex:
    bop_root = Path(bop_root)
    manifest_path = bop_root / BOP_EXPORT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing BOP export manifest: {manifest_path}")

    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"BOP export manifest must be a JSON object: {manifest_path}")

    sensor_scenes = {}
    for export in manifest.get("exports", []):
        if not isinstance(export, dict):
            continue
        sensor_name = export.get("sensor_name")
        if isinstance(sensor_name, str):
            sensor_scenes[sensor_name] = export
    if not sensor_scenes:
        raise ValueError(f"BOP export manifest has no scene exports: {manifest_path}")

    object_ids_by_name: dict[str, int] = {}
    for model in manifest.get("object_models", []):
        if not isinstance(model, dict):
            continue
        object_name = model.get("object_name")
        obj_id = model.get("obj_id")
        if isinstance(object_name, str) and obj_id is not None:
            object_ids_by_name[object_name] = int(obj_id)

    models_info_path = bop_root / MODELS_DIR / "models_info.json"
    if models_info_path.is_file():
        models_info = _load_json(models_info_path)
        if isinstance(models_info, dict):
            for obj_id, model_info in models_info.items():
                if not isinstance(model_info, dict):
                    continue
                source_name = model_info.get("source_name")
                if isinstance(source_name, str):
                    object_ids_by_name.setdefault(source_name, int(obj_id))

    if not object_ids_by_name:
        raise ValueError(
            "BOP export does not contain object model metadata. "
            "Run BOP export with model export before converting estimator results."
        )

    return BopExportIndex(
        bop_root=bop_root.as_posix(),
        sensor_scenes=sensor_scenes,
        object_ids_by_name=object_ids_by_name,
    )


def parse_foundationpose_output_name(
    folder_name: str,
) -> tuple[str, str | None, int | None] | None:
    return parse_estimator_output_name(
        folder_name,
        methods=("foundationposeNoTracking", "foundationpose"),
    )


def parse_estimator_output_name(
    folder_name: str,
    *,
    methods: tuple[str, ...],
) -> tuple[str, str | None, int | None] | None:
    if not folder_name.endswith("_output"):
        return None
    base_name = folder_name.removesuffix("_output")
    for method in methods:
        if not base_name.startswith(method):
            continue
        rest = base_name[len(method) :].lstrip("_")
        config_parts = []
        object_index = None
        for part in rest.split("_") if rest else []:
            match = re.fullmatch(r"obj(\d+)", part)
            if match:
                object_index = int(match.group(1))
            else:
                config_parts.append(part)
        result_id = "_".join(config_parts) if config_parts else None
        return method, result_id, object_index
    return None


def discover_foundationpose_output_folders(input_folder: str | Path) -> list[Path]:
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"FoundationPose input folder not found: {input_folder}")
    outputs = [
        path
        for sensor_folder in sorted(input_folder.iterdir())
        if sensor_folder.is_dir()
        for path in sorted(sensor_folder.glob("foundationpose*_output"))
        if (path / "ob_in_cam").is_dir()
    ]
    if not outputs:
        raise FileNotFoundError(f"No FoundationPose output folders in {input_folder}")
    return outputs


def discover_megapose_output_folders(input_folder: str | Path) -> list[Path]:
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"MegaPose input folder not found: {input_folder}")
    outputs = [
        path
        for sensor_folder in sorted(input_folder.iterdir())
        if sensor_folder.is_dir()
        for path in sorted(sensor_folder.glob("megapose*_output"))
        if (path / "megapose_poses.json").is_file()
    ]
    if not outputs:
        raise FileNotFoundError(f"No MegaPose output folders in {input_folder}")
    return outputs


def discover_sam6d_output_folders(input_folder: str | Path) -> list[Path]:
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"SAM6D input folder not found: {input_folder}")
    outputs = [
        path
        for sensor_folder in sorted(input_folder.iterdir())
        if sensor_folder.is_dir()
        for path in sorted(sensor_folder.glob("sam6d*_output"))
        if (path / "detections_pem").is_dir()
    ]
    if not outputs:
        raise FileNotFoundError(f"No SAM6D output folders in {input_folder}")
    return outputs


def sensor_object_names(sensor_folder: str | Path) -> list[str]:
    objects_json = Path(sensor_folder) / "blenderproc" / "objects.json"
    if not objects_json.is_file():
        raise FileNotFoundError(f"Missing BlenderProc objects file: {objects_json}")
    value = _load_json(objects_json)
    if not isinstance(value, dict):
        raise ValueError(f"BlenderProc objects file must be a JSON object: {objects_json}")
    return list(value.keys())


def object_name_for_index(sensor_folder: Path, object_index: int | None) -> str:
    object_names = sensor_object_names(sensor_folder)
    if object_index is None:
        if len(object_names) == 1:
            return object_names[0]
        raise ValueError(
            f"Cannot infer object for {sensor_folder}; output folder does not "
            "include objN and objects.json contains multiple objects."
        )
    try:
        return object_names[object_index]
    except IndexError as exc:
        raise ValueError(
            f"Object index {object_index} is not present in {sensor_folder / 'blenderproc' / 'objects.json'}"
        ) from exc


def foundationpose_output_metadata(
    output_folder: str | Path, bop_index: BopExportIndex
) -> FoundationPoseOutput:
    output_folder = Path(output_folder)
    parsed = parse_foundationpose_output_name(output_folder.name)
    if parsed is None:
        raise ValueError(f"Not a recognized FoundationPose output folder: {output_folder}")
    method, result_id, object_index = parsed
    sensor_folder = output_folder.parent
    sensor_name = sensor_folder.name
    if sensor_name not in bop_index.sensor_scenes:
        raise ValueError(
            f"Sensor {sensor_name!r} is not present in BOP export manifest."
        )
    object_name = object_name_for_index(sensor_folder, object_index)
    try:
        obj_id = bop_index.object_ids_by_name[object_name]
    except KeyError as exc:
        raise ValueError(
            f"Object {object_name!r} is not present in BOP model metadata."
        ) from exc

    return FoundationPoseOutput(
        output_folder=output_folder.as_posix(),
        sensor_folder=sensor_folder.as_posix(),
        sensor_name=sensor_name,
        method=method,
        result_id=result_id,
        object_index=object_index,
        object_name=object_name,
        obj_id=obj_id,
    )


def estimator_output_metadata(
    output_folder: str | Path,
    bop_index: BopExportIndex,
    *,
    methods: tuple[str, ...],
) -> EstimatorOutput:
    output_folder = Path(output_folder)
    parsed = parse_estimator_output_name(output_folder.name, methods=methods)
    if parsed is None:
        method_names = ", ".join(methods)
        raise ValueError(
            f"Not a recognized {method_names} output folder: {output_folder}"
        )
    method, result_id, object_index = parsed
    sensor_folder = output_folder.parent
    sensor_name = sensor_folder.name
    if sensor_name not in bop_index.sensor_scenes:
        raise ValueError(
            f"Sensor {sensor_name!r} is not present in BOP export manifest."
        )
    object_name = object_name_for_index(sensor_folder, object_index)
    try:
        obj_id = bop_index.object_ids_by_name[object_name]
    except KeyError as exc:
        raise ValueError(
            f"Object {object_name!r} is not present in BOP model metadata."
        ) from exc

    return EstimatorOutput(
        output_folder=output_folder.as_posix(),
        sensor_folder=sensor_folder.as_posix(),
        sensor_name=sensor_name,
        method=method,
        result_id=result_id,
        object_index=object_index,
        object_name=object_name,
        obj_id=obj_id,
    )


def read_foundationpose_pose(
    path: str | Path, *, translation_scale_to_mm: float
) -> tuple[list[float], list[float]]:
    path = Path(path)
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4):
        raise ValueError(f"FoundationPose pose must be a 4x4 matrix: {path}")
    rotation = matrix[:3, :3].reshape(-1).astype(float).tolist()
    translation = (matrix[:3, 3] * translation_scale_to_mm).astype(float).tolist()
    return rotation, translation


def _numeric_sequence(
    value: object,
    *,
    expected_count: int,
    label: str,
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} values")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc


def _numeric_matrix3x3(value: object, *, label: str) -> list[float]:
    if isinstance(value, list) and len(value) == 3 and all(
        isinstance(row, list) for row in value
    ):
        flattened = [item for row in value for item in row]
        return _numeric_sequence(flattened, expected_count=9, label=label)
    return _numeric_sequence(value, expected_count=9, label=label)


def _rotation_matrix_from_quaternion_xyzw(quaternion: list[float]) -> list[float]:
    x, y, z, w = quaternion
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0.0:
        raise ValueError("MegaPose quaternion must not be zero length")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    matrix = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )
    return matrix.reshape(-1).astype(float).tolist()


def iter_pose_files(output_folder: str | Path) -> Iterable[Path]:
    pose_folder = Path(output_folder) / "ob_in_cam"
    if not pose_folder.is_dir():
        raise FileNotFoundError(f"Missing FoundationPose ob_in_cam folder: {pose_folder}")
    for path in sorted(pose_folder.iterdir()):
        if not path.is_file():
            continue
        try:
            int(path.stem)
        except ValueError:
            continue
        yield path


def iter_sam6d_detection_files(output_folder: str | Path) -> Iterable[Path]:
    detection_folder = Path(output_folder) / "detections_pem"
    if not detection_folder.is_dir():
        raise FileNotFoundError(f"Missing SAM6D detections folder: {detection_folder}")
    for path in sorted(detection_folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            int(path.stem.split("_")[0])
        except ValueError:
            continue
        yield path


def discover_aruco_pose_files(input_folder: str | Path) -> list[Path]:
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        raise FileNotFoundError(f"ArUco input folder not found: {input_folder}")
    outputs = [
        path
        for sensor_folder in sorted(input_folder.iterdir())
        if sensor_folder.is_dir()
        for path in [sensor_folder / ARUCO_POSE_ESTIMATION]
        if path.is_file()
    ]
    if not outputs:
        raise FileNotFoundError(f"No {ARUCO_POSE_ESTIMATION} files in {input_folder}")
    return outputs


def aruco_output_metadata(
    path: str | Path,
    bop_index: BopExportIndex,
    *,
    object_name: str,
) -> ArucoOutput:
    path = Path(path)
    sensor_folder = path.parent
    sensor_name = sensor_folder.name
    if sensor_name not in bop_index.sensor_scenes:
        raise ValueError(
            f"Sensor {sensor_name!r} is not present in BOP export manifest."
        )
    try:
        obj_id = bop_index.object_ids_by_name[object_name]
    except KeyError as exc:
        raise ValueError(
            f"Object {object_name!r} is not present in BOP model metadata."
        ) from exc
    return ArucoOutput(
        path=path.as_posix(),
        sensor_folder=sensor_folder.as_posix(),
        sensor_name=sensor_name,
        object_name=object_name,
        obj_id=obj_id,
    )


def _numeric_vector(value: object, *, expected_count: int, label: str) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected_count:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ArUco {label} must contain numeric values") from exc


def _frame_id_from_image_name(image_name: str) -> int | None:
    try:
        return int(Path(image_name).stem)
    except ValueError:
        return None


def rows_from_aruco_output(
    output: ArucoOutput,
    bop_index: BopExportIndex,
    *,
    default_score: float,
    default_time: float,
    translation_scale_to_mm: float,
    min_marker_count: int,
) -> list[BopResultRow]:
    scene = bop_index.sensor_scenes[output.sensor_name]
    scene_id = int(scene["scene_id"])
    value = _load_json(output.path)
    if not isinstance(value, Mapping):
        raise ValueError(f"ArUco pose file must be a JSON object: {output.path}")

    rows = []
    for image_name, frame in sorted(value.items(), key=lambda item: item[0]):
        if not isinstance(frame, Mapping):
            continue
        pose = frame.get("aruco_pose_estimation")
        if not isinstance(pose, Mapping):
            continue
        len_ids = int(pose.get("len_ids", 0) or 0)
        if len_ids < min_marker_count:
            continue
        rvec = _numeric_vector(pose.get("rvec"), expected_count=3, label="rvec")
        tvec = _numeric_vector(pose.get("tvec"), expected_count=3, label="tvec")
        im_id = _frame_id_from_image_name(str(image_name))
        if rvec is None or tvec is None or im_id is None:
            continue
        rotation_matrix, _ = cv.Rodrigues(np.array(rvec, dtype=float).reshape(3, 1))
        rows.append(
            BopResultRow(
                scene_id=scene_id,
                im_id=im_id,
                obj_id=output.obj_id,
                score=float(default_score),
                R=rotation_matrix.reshape(-1).astype(float).tolist(),
                t=(np.array(tvec, dtype=float) * translation_scale_to_mm).tolist(),
                time=float(default_time),
                source_pose_file=f"{output.path}#{image_name}",
            )
        )
    return rows


def rows_from_megapose_output(
    output: EstimatorOutput,
    bop_index: BopExportIndex,
    *,
    default_score: float,
    default_time: float,
    translation_scale_to_mm: float,
) -> list[BopResultRow]:
    scene = bop_index.sensor_scenes[output.sensor_name]
    scene_id = int(scene["scene_id"])
    pose_file = Path(output.output_folder) / "megapose_poses.json"
    value = _load_json(pose_file)
    if not isinstance(value, Mapping):
        raise ValueError(f"MegaPose pose file must be a JSON object: {pose_file}")

    rows = []
    for frame_id, detections in sorted(value.items(), key=lambda item: str(item[0])):
        try:
            im_id = int(frame_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(detections, list) or not detections:
            continue
        detection = detections[0]
        if not isinstance(detection, Mapping):
            continue
        pose = detection.get("TWO")
        if not isinstance(pose, list) or len(pose) != 2:
            continue
        quaternion = _numeric_sequence(
            pose[0],
            expected_count=4,
            label=f"MegaPose quaternion for frame {frame_id}",
        )
        translation = _numeric_sequence(
            pose[1],
            expected_count=3,
            label=f"MegaPose translation for frame {frame_id}",
        )
        rows.append(
            BopResultRow(
                scene_id=scene_id,
                im_id=im_id,
                obj_id=output.obj_id,
                score=float(default_score),
                R=_rotation_matrix_from_quaternion_xyzw(quaternion),
                t=(np.array(translation, dtype=float) * translation_scale_to_mm).tolist(),
                time=float(default_time),
                source_pose_file=f"{pose_file.as_posix()}#{frame_id}",
            )
        )
    return rows


def rows_from_sam6d_output(
    output: EstimatorOutput,
    bop_index: BopExportIndex,
    *,
    default_score: float,
    default_time: float,
    translation_scale_to_mm: float,
) -> list[BopResultRow]:
    scene = bop_index.sensor_scenes[output.sensor_name]
    scene_id = int(scene["scene_id"])
    rows = []
    for detection_file in iter_sam6d_detection_files(output.output_folder):
        detections = _load_json(detection_file)
        if not isinstance(detections, list) or not detections:
            continue
        mappings = [item for item in detections if isinstance(item, Mapping)]
        if not mappings:
            continue
        detection = max(mappings, key=lambda item: float(item.get("score", 0.0) or 0.0))
        rotation = np.array(
            _numeric_matrix3x3(
                detection.get("R"),
                label=f"SAM6D rotation for {detection_file}",
            ),
            dtype=float,
        ).reshape(3, 3)
        translation = _numeric_sequence(
            detection.get("t"),
            expected_count=3,
            label=f"SAM6D translation for {detection_file}",
        )
        raw_score = detection.get("score", default_score)
        rows.append(
            BopResultRow(
                scene_id=scene_id,
                im_id=int(detection_file.stem.split("_")[0]),
                obj_id=output.obj_id,
                score=float(raw_score if raw_score is not None else default_score),
                R=rotation.reshape(-1).astype(float).tolist(),
                t=(np.array(translation, dtype=float) * translation_scale_to_mm).tolist(),
                time=float(default_time),
                source_pose_file=detection_file.as_posix(),
            )
        )
    return rows


def rows_from_foundationpose_output(
    output: FoundationPoseOutput,
    bop_index: BopExportIndex,
    *,
    default_score: float,
    default_time: float,
    translation_scale_to_mm: float,
) -> list[BopResultRow]:
    scene = bop_index.sensor_scenes[output.sensor_name]
    scene_id = int(scene["scene_id"])
    rows = []
    for pose_file in iter_pose_files(output.output_folder):
        rotation, translation = read_foundationpose_pose(
            pose_file,
            translation_scale_to_mm=translation_scale_to_mm,
        )
        rows.append(
            BopResultRow(
                scene_id=scene_id,
                im_id=int(pose_file.stem),
                obj_id=output.obj_id,
                score=float(default_score),
                R=rotation,
                t=translation,
                time=float(default_time),
                source_pose_file=pose_file.as_posix(),
            )
        )
    return rows


def _safe_result_id_from_output(output: EstimatorOutput) -> str | None:
    return output.result_id


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _result_filename(
    *, method: str, dataset_name: str, split: str, result_id: str | None
) -> str:
    filename = f"{method}_{dataset_name}-{split}"
    if result_id:
        safe_result_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", result_id)
        filename += f"_{safe_result_id}"
    return f"{filename}.csv"


def write_bop19_result_csv(path: str | Path, rows: list[BopResultRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BOP19_RESULT_HEADER)
        for row in sorted(rows, key=lambda item: (item.scene_id, item.im_id, item.obj_id)):
            writer.writerow(
                [
                    row.scene_id,
                    row.im_id,
                    row.obj_id,
                    _format_float(row.score),
                    " ".join(_format_float(value) for value in row.R),
                    " ".join(_format_float(value) for value in row.t),
                    _format_float(row.time),
                ]
            )
    return path


def export_foundationpose_bop_results(
    *,
    run_root: str | Path,
    input_folder: str | Path | None = None,
    foundationpose_outputs: list[str | Path] | None = None,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    default_score: float = 1.0,
    default_time: float = -1.0,
    translation_scale_to_mm: float = 1000.0,
) -> BopResultExportManifest:
    run_root = Path(run_root)
    bop_root_path = Path(bop_root) if bop_root is not None else run_root / BOP_DIR
    input_folder_path = (
        Path(input_folder)
        if input_folder is not None
        else run_root / "processed" / "synchronized"
    )
    output_folder_path = (
        Path(output_folder)
        if output_folder is not None
        else run_root / RESULTS_DIR / BOP_DIR
    )
    dataset = dataset_name or bop_root_path.name
    bop_index = load_bop_export_index(bop_root_path)
    output_paths = (
        [Path(path) for path in foundationpose_outputs]
        if foundationpose_outputs
        else discover_foundationpose_output_folders(input_folder_path)
    )

    outputs = [
        foundationpose_output_metadata(output_path, bop_index)
        for output_path in output_paths
    ]
    rows_by_group: dict[tuple[str, str | None, str], list[BopResultRow]] = {}
    sources_by_group: dict[tuple[str, str | None, str], list[str]] = {}
    for output in outputs:
        scene = bop_index.sensor_scenes[output.sensor_name]
        split = str(scene.get("split", "test"))
        group_key = (output.method, output.result_id, split)
        rows_by_group.setdefault(group_key, []).extend(
            rows_from_foundationpose_output(
                output,
                bop_index,
                default_score=default_score,
                default_time=default_time,
                translation_scale_to_mm=translation_scale_to_mm,
            )
        )
        sources_by_group.setdefault(group_key, []).append(output.output_folder)

    result_files: list[BopResultFile] = []
    for (method, result_id, split), rows in sorted(rows_by_group.items()):
        if not rows:
            continue
        result_path = output_folder_path / _result_filename(
            method=method,
            dataset_name=dataset,
            split=split,
            result_id=result_id,
        )
        write_bop19_result_csv(result_path, rows)
        result_files.append(
            BopResultFile(
                path=result_path.as_posix(),
                filename=result_path.name,
                method=method,
                dataset=dataset,
                split=split,
                result_id=result_id,
                row_count=len(rows),
                source_outputs=sources_by_group[(method, result_id, split)],
            )
        )

    if not result_files:
        raise ValueError("No BOP result rows were exported from FoundationPose outputs.")

    return BopResultExportManifest(
        schema_version=SCHEMA_VERSION,
        run_root=run_root.as_posix(),
        bop_root=bop_root_path.as_posix(),
        input_folder=input_folder_path.as_posix(),
        output_folder=output_folder_path.as_posix(),
        dataset_name=dataset,
        source_type="foundationpose",
        translation_scale_to_mm=float(translation_scale_to_mm),
        results=result_files,
    )


def _export_estimator_directory_bop_results(
    *,
    run_root: str | Path,
    source_type: str,
    methods: tuple[str, ...],
    output_paths: list[str | Path] | None,
    discover_outputs: Callable[[str | Path], list[Path]],
    rows_from_output: Callable[..., list[BopResultRow]],
    input_folder: str | Path | None = None,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    default_score: float = 1.0,
    default_time: float = -1.0,
    translation_scale_to_mm: float = 1.0,
) -> BopResultExportManifest:
    run_root = Path(run_root)
    bop_root_path = Path(bop_root) if bop_root is not None else run_root / BOP_DIR
    input_folder_path = (
        Path(input_folder)
        if input_folder is not None
        else run_root / "processed" / "synchronized"
    )
    output_folder_path = (
        Path(output_folder)
        if output_folder is not None
        else run_root / RESULTS_DIR / BOP_DIR
    )
    dataset = dataset_name or bop_root_path.name
    bop_index = load_bop_export_index(bop_root_path)
    output_folder_paths = (
        [Path(path) for path in output_paths]
        if output_paths
        else discover_outputs(input_folder_path)
    )

    outputs = [
        estimator_output_metadata(output_path, bop_index, methods=methods)
        for output_path in output_folder_paths
    ]
    rows_by_group: dict[tuple[str, str | None, str], list[BopResultRow]] = {}
    sources_by_group: dict[tuple[str, str | None, str], list[str]] = {}
    for output in outputs:
        scene = bop_index.sensor_scenes[output.sensor_name]
        split = str(scene.get("split", "test"))
        group_key = (output.method, _safe_result_id_from_output(output), split)
        rows_by_group.setdefault(group_key, []).extend(
            rows_from_output(
                output,
                bop_index,
                default_score=default_score,
                default_time=default_time,
                translation_scale_to_mm=translation_scale_to_mm,
            )
        )
        sources_by_group.setdefault(group_key, []).append(output.output_folder)

    result_files: list[BopResultFile] = []
    for (method, result_id, split), rows in sorted(rows_by_group.items()):
        if not rows:
            continue
        result_path = output_folder_path / _result_filename(
            method=method,
            dataset_name=dataset,
            split=split,
            result_id=result_id,
        )
        write_bop19_result_csv(result_path, rows)
        result_files.append(
            BopResultFile(
                path=result_path.as_posix(),
                filename=result_path.name,
                method=method,
                dataset=dataset,
                split=split,
                result_id=result_id,
                row_count=len(rows),
                source_outputs=sources_by_group[(method, result_id, split)],
            )
        )

    if not result_files:
        raise ValueError(f"No BOP result rows were exported from {source_type} outputs.")

    return BopResultExportManifest(
        schema_version=SCHEMA_VERSION,
        run_root=run_root.as_posix(),
        bop_root=bop_root_path.as_posix(),
        input_folder=input_folder_path.as_posix(),
        output_folder=output_folder_path.as_posix(),
        dataset_name=dataset,
        source_type=source_type,
        translation_scale_to_mm=float(translation_scale_to_mm),
        results=result_files,
    )


def export_megapose_bop_results(
    *,
    run_root: str | Path,
    input_folder: str | Path | None = None,
    megapose_outputs: list[str | Path] | None = None,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    default_score: float = 1.0,
    default_time: float = -1.0,
    translation_scale_to_mm: float = 1000.0,
) -> BopResultExportManifest:
    return _export_estimator_directory_bop_results(
        run_root=run_root,
        source_type="megapose",
        methods=("megapose",),
        output_paths=megapose_outputs,
        discover_outputs=discover_megapose_output_folders,
        rows_from_output=rows_from_megapose_output,
        input_folder=input_folder,
        bop_root=bop_root,
        output_folder=output_folder,
        dataset_name=dataset_name,
        default_score=default_score,
        default_time=default_time,
        translation_scale_to_mm=translation_scale_to_mm,
    )


def export_sam6d_bop_results(
    *,
    run_root: str | Path,
    input_folder: str | Path | None = None,
    sam6d_outputs: list[str | Path] | None = None,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    default_score: float = 1.0,
    default_time: float = -1.0,
    translation_scale_to_mm: float = 1.0,
) -> BopResultExportManifest:
    return _export_estimator_directory_bop_results(
        run_root=run_root,
        source_type="sam6d",
        methods=("sam6d",),
        output_paths=sam6d_outputs,
        discover_outputs=discover_sam6d_output_folders,
        rows_from_output=rows_from_sam6d_output,
        input_folder=input_folder,
        bop_root=bop_root,
        output_folder=output_folder,
        dataset_name=dataset_name,
        default_score=default_score,
        default_time=default_time,
        translation_scale_to_mm=translation_scale_to_mm,
    )


def export_aruco_bop_results(
    *,
    run_root: str | Path,
    input_folder: str | Path | None = None,
    aruco_pose_files: list[str | Path] | None = None,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    object_name: str = "aruco",
    default_score: float = 1.0,
    default_time: float = -1.0,
    translation_scale_to_mm: float = 1.0,
    min_marker_count: int = 1,
) -> BopResultExportManifest:
    run_root = Path(run_root)
    bop_root_path = Path(bop_root) if bop_root is not None else run_root / BOP_DIR
    input_folder_path = (
        Path(input_folder)
        if input_folder is not None
        else run_root / "processed" / "synchronized"
    )
    output_folder_path = (
        Path(output_folder)
        if output_folder is not None
        else run_root / RESULTS_DIR / BOP_DIR
    )
    dataset = dataset_name or bop_root_path.name
    bop_index = load_bop_export_index(bop_root_path)
    output_paths = (
        [Path(path) for path in aruco_pose_files]
        if aruco_pose_files
        else discover_aruco_pose_files(input_folder_path)
    )

    outputs = [
        aruco_output_metadata(
            output_path,
            bop_index,
            object_name=object_name,
        )
        for output_path in output_paths
    ]
    rows_by_split: dict[str, list[BopResultRow]] = {}
    sources_by_split: dict[str, list[str]] = {}
    for output in outputs:
        scene = bop_index.sensor_scenes[output.sensor_name]
        split = str(scene.get("split", "test"))
        rows_by_split.setdefault(split, []).extend(
            rows_from_aruco_output(
                output,
                bop_index,
                default_score=default_score,
                default_time=default_time,
                translation_scale_to_mm=translation_scale_to_mm,
                min_marker_count=min_marker_count,
            )
        )
        sources_by_split.setdefault(split, []).append(output.path)

    result_files: list[BopResultFile] = []
    for split, rows in sorted(rows_by_split.items()):
        if not rows:
            continue
        result_path = output_folder_path / _result_filename(
            method="aruco",
            dataset_name=dataset,
            split=split,
            result_id=None,
        )
        write_bop19_result_csv(result_path, rows)
        result_files.append(
            BopResultFile(
                path=result_path.as_posix(),
                filename=result_path.name,
                method="aruco",
                dataset=dataset,
                split=split,
                result_id=None,
                row_count=len(rows),
                source_outputs=sources_by_split[split],
            )
        )

    if not result_files:
        raise ValueError("No BOP result rows were exported from ArUco outputs.")

    return BopResultExportManifest(
        schema_version=SCHEMA_VERSION,
        run_root=run_root.as_posix(),
        bop_root=bop_root_path.as_posix(),
        input_folder=input_folder_path.as_posix(),
        output_folder=output_folder_path.as_posix(),
        dataset_name=dataset,
        source_type="aruco",
        translation_scale_to_mm=float(translation_scale_to_mm),
        results=result_files,
    )


def write_bop_result_export_manifest(
    run_root: str | Path, manifest: BopResultExportManifest
) -> Path:
    path = Path(run_root) / BOP_RESULT_EXPORT_MANIFEST
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path
