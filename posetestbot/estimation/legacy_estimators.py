"""Manifest-tracked plans for legacy estimator wrapper adapters."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from posetestbot.io.artifacts import PROCESSED_DIR, SYNCHRONIZED_DIR


@dataclass(frozen=True)
class LegacyEstimatorJob:
    sensor_name: str
    sensor_folder: str
    object_name: str
    object_id: int
    expected_output_folder: str


@dataclass(frozen=True)
class LegacyEstimatorPlan:
    schema_version: str
    dry_run: bool
    estimator_id: str
    input_folder: str
    wrapper_script: str
    wrapper_exists: bool
    object_id: int
    result_id: str | None
    command: list[str]
    options: dict[str, object]
    jobs: list[LegacyEstimatorJob]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["jobs"] = [asdict(job) for job in self.jobs]
        return data


def synchronized_input_folder(run_root: Path, explicit_input_folder: str | None) -> Path:
    if explicit_input_folder:
        return Path(explicit_input_folder)
    return run_root / PROCESSED_DIR / SYNCHRONIZED_DIR


def _load_sensor_objects(sensor_folder: Path) -> dict[str, object]:
    objects_json = sensor_folder / "blenderproc" / "objects.json"
    if not objects_json.is_file():
        raise FileNotFoundError(f"Missing BlenderProc objects file: {objects_json}")
    with open(objects_json, "r") as f:
        objects = json.load(f)
    if not isinstance(objects, dict) or not objects:
        raise ValueError(
            f"BlenderProc objects file must be a non-empty object: {objects_json}"
        )
    return objects


def object_name_for_sensor(sensor_folder: Path, object_id: int) -> str:
    object_names = list(_load_sensor_objects(sensor_folder).keys())
    try:
        return object_names[object_id]
    except IndexError as exc:
        raise ValueError(
            f"Object ID {object_id} is not present in {sensor_folder / 'blenderproc' / 'objects.json'}; "
            f"available objects: {', '.join(object_names)}"
        ) from exc


def safe_result_id(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())


def expected_output_name(
    *,
    estimator_id: str,
    object_id: int,
    result_id: str | None = None,
) -> str:
    safe_id = safe_result_id(result_id)
    if safe_id:
        return f"{estimator_id}_{safe_id}_obj{object_id}_output"
    return f"{estimator_id}_obj{object_id}_output"


def discover_estimator_jobs(
    *,
    input_folder: Path,
    estimator_id: str,
    object_id: int,
    result_id: str | None = None,
) -> list[LegacyEstimatorJob]:
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    output_name = expected_output_name(
        estimator_id=estimator_id,
        object_id=object_id,
        result_id=result_id,
    )
    jobs = []
    for sensor_folder in sorted(input_folder.iterdir()):
        if not sensor_folder.is_dir():
            continue
        object_name = object_name_for_sensor(sensor_folder, object_id)
        jobs.append(
            LegacyEstimatorJob(
                sensor_name=sensor_folder.name,
                sensor_folder=sensor_folder.as_posix(),
                object_name=object_name,
                object_id=object_id,
                expected_output_folder=(sensor_folder / output_name).as_posix(),
            )
        )
    if not jobs:
        raise FileNotFoundError(f"No synchronized sensor folders in {input_folder}")
    return jobs


def wrapper_exists(wrapper_script: str | Path, *, repo_root: Path) -> bool:
    path = Path(wrapper_script)
    if path.is_absolute():
        return path.is_file()
    return (repo_root / path).is_file()


def write_legacy_estimator_plan(
    run_root: Path,
    artifact_name: str,
    plan: LegacyEstimatorPlan,
) -> Path:
    path = run_root / artifact_name
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def existing_output_artifacts(
    plan: LegacyEstimatorPlan,
    *,
    artifact_suffix: str,
) -> dict[str, Path]:
    artifacts = {}
    for job in plan.jobs:
        output_folder = Path(job.expected_output_folder)
        if output_folder.exists():
            artifacts[f"{job.sensor_name}:{artifact_suffix}"] = output_folder
    return artifacts
