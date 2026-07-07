"""BOP Toolkit evaluation planning for PoseTestBot BOP exports."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EVALUATION_REPORT,
    BOP_TARGETS_BOP19,
    EVALUATION_DIR,
    MODELS_DIR,
)


SCHEMA_VERSION = "bop_evaluation_plan.v1"
BOP_EVALUATION_REPORT_SCHEMA_VERSION = "bop_evaluation_report.v1"
BOP19_RESULT_HEADER = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]


@dataclass(frozen=True)
class BopResultMetadata:
    path: str
    filename: str
    result_name: str
    method: str
    dataset: str
    split: str
    split_type: str | None
    extension: str
    row_count: int


@dataclass(frozen=True)
class BopEvaluationPlan:
    schema_version: str
    dry_run: bool
    bop_root: str
    bop_path: str
    dataset_folder: str
    result: BopResultMetadata
    eval_path: str
    targets_filename: str
    eval_script: str
    command: list[str]
    environment: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BopEvaluationCheck:
    name: str
    ok: bool
    value: str | bool | int | None = None
    hint: str | None = None


@dataclass(frozen=True)
class BopEvaluationOutputArtifact:
    path: str
    relative_path: str
    size_bytes: int


@dataclass(frozen=True)
class BopEvaluationScoreFile:
    path: str
    relative_path: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class BopEvaluationScoreSummary:
    score_file_count: int
    metrics: dict[str, float]
    files: list[BopEvaluationScoreFile]


@dataclass(frozen=True)
class BopEvaluationReport:
    schema_version: str
    run_root: str
    status: str
    dry_run: bool
    plan_path: str
    result: BopResultMetadata
    eval_path: str
    command: list[str]
    environment: dict[str, str]
    checks: list[BopEvaluationCheck]
    output_artifacts: list[BopEvaluationOutputArtifact]
    score_summary: BopEvaluationScoreSummary
    message: str
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_bop_result_filename(path: str | Path) -> tuple[str, str, str, str, str | None, str]:
    filename = Path(path).name
    if "." not in filename:
        raise ValueError(f"BOP result filename must include an extension: {filename}")

    result_name, extension = filename.rsplit(".", 1)
    name_parts = result_name.split("_")
    if len(name_parts) < 2:
        raise ValueError(
            "BOP result filename must look like "
            "'{method}_{dataset}-{split}.csv': "
            f"{filename}"
        )

    method = name_parts[0]
    dataset_split = name_parts[1]
    split_parts = dataset_split.split("-")
    if len(split_parts) < 2:
        raise ValueError(
            "BOP result filename must include dataset and split, for example "
            f"'foundationpose_bop-test.csv': {filename}"
        )

    dataset = split_parts[0]
    split = split_parts[1]
    split_type = split_parts[2] if len(split_parts) > 2 else None
    return result_name, method, dataset, split, split_type, extension


def validate_bop19_result_file(path: str | Path) -> BopResultMetadata:
    result_path = Path(path)
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing BOP result CSV: {result_path}")

    result_name, method, dataset, split, split_type, extension = parse_bop_result_filename(
        result_path
    )
    if extension.lower() != "csv":
        raise ValueError(f"BOP19 pose results must be CSV files: {result_path.name}")

    row_count = 0
    with open(result_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"BOP result CSV is empty: {result_path}") from exc

        if header != BOP19_RESULT_HEADER:
            raise ValueError(
                f"BOP result CSV header must be {BOP19_RESULT_HEADER}: {result_path}"
            )

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(BOP19_RESULT_HEADER):
                raise ValueError(
                    f"BOP result row {row_number} must have 7 columns: {result_path}"
                )
            try:
                int(row[0])
                int(row[1])
                int(row[2])
                float(row[3])
                rotation = [float(value) for value in row[4].split()]
                translation = [float(value) for value in row[5].split()]
                float(row[6])
            except ValueError as exc:
                raise ValueError(
                    f"BOP result row {row_number} contains invalid numeric values: "
                    f"{result_path}"
                ) from exc
            if len(rotation) != 9:
                raise ValueError(
                    f"BOP result row {row_number} rotation must have 9 values: "
                    f"{result_path}"
                )
            if len(translation) != 3:
                raise ValueError(
                    f"BOP result row {row_number} translation must have 3 values: "
                    f"{result_path}"
                )
            row_count += 1

    if row_count == 0:
        raise ValueError(f"BOP result CSV has no pose rows: {result_path}")

    return BopResultMetadata(
        path=result_path.as_posix(),
        filename=result_path.name,
        result_name=result_name,
        method=method,
        dataset=dataset,
        split=split,
        split_type=split_type,
        extension=extension,
        row_count=row_count,
    )


def validate_bop_targets_file(path: str | Path) -> int:
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing BOP target file: {target_path}")

    try:
        with open(target_path, "r") as f:
            targets = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"BOP target file is invalid JSON: {target_path}") from exc

    if not isinstance(targets, list):
        raise ValueError(f"BOP target file must be a JSON list: {target_path}")
    if not targets:
        raise ValueError(f"BOP target file has no target rows: {target_path}")

    required_fields = ("scene_id", "im_id", "obj_id", "inst_count")
    for row_number, target in enumerate(targets, start=1):
        if not isinstance(target, dict):
            raise ValueError(
                f"BOP target row {row_number} must be a JSON object: {target_path}"
            )
        for field in required_fields:
            try:
                int(target[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"BOP target row {row_number} has invalid {field!r}: "
                    f"{target_path}"
                ) from exc

    return len(targets)


def default_eval_path(run_root: str | Path, result_name: str) -> Path:
    return Path(run_root) / EVALUATION_DIR / "bop_toolkit" / result_name


def default_eval_script(
    *,
    bop_toolkit_root: str | Path | None = None,
    eval_script: str | Path | None = None,
) -> Path:
    if eval_script is not None:
        return Path(eval_script)
    root = bop_toolkit_root or os.environ.get("BOP_TOOLKIT_ROOT")
    if root:
        return Path(root) / "scripts" / "eval_bop19_pose.py"
    return Path("bop_toolkit") / "scripts" / "eval_bop19_pose.py"


def build_bop_evaluation_plan(
    *,
    run_root: str | Path,
    result_file: str | Path,
    bop_root: str | Path | None = None,
    bop_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    targets_filename: str = BOP_TARGETS_BOP19,
    eval_script: str | Path | None = None,
    bop_toolkit_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    renderer_type: str = "vispy",
    num_workers: int = 1,
    use_gpu: bool = False,
    device: str = "cuda:0",
    cleanup_eval: bool = False,
    dry_run: bool = False,
) -> BopEvaluationPlan:
    run_root = Path(run_root)
    bop_root_path = Path(bop_root) if bop_root is not None else run_root / BOP_DIR
    if not bop_root_path.is_dir():
        raise FileNotFoundError(f"Missing BOP dataset root: {bop_root_path}")

    result = validate_bop19_result_file(result_file)
    bop_path_path = Path(bop_path) if bop_path is not None else bop_root_path.parent
    dataset_folder = bop_path_path / result.dataset
    if not dataset_folder.is_dir():
        raise FileNotFoundError(
            "The BOP result filename implies dataset "
            f"{result.dataset!r}, but {dataset_folder} does not exist. "
            "Use a result filename like '<method>_bop-test.csv' for the default "
            "<run_root>/bop export, or pass --bop-path/--bop-root explicitly."
        )

    target_file = bop_root_path / targets_filename
    validate_bop_targets_file(target_file)

    script_path = default_eval_script(
        bop_toolkit_root=bop_toolkit_root,
        eval_script=eval_script,
    )
    if not dry_run and not script_path.is_file():
        raise FileNotFoundError(
            f"Missing BOP Toolkit eval script: {script_path}. "
            "Pass --bop-toolkit-root or --eval-script."
        )

    eval_path_path = Path(eval_path) if eval_path is not None else default_eval_path(
        run_root, result.result_name
    )
    python = str(python_executable or sys.executable)
    command = [
        python,
        script_path.as_posix(),
        f"--result_filenames={result.filename}",
        f"--results_path={Path(result.path).parent.as_posix()}",
        f"--eval_path={eval_path_path.as_posix()}",
        f"--targets_filename={targets_filename}",
        f"--renderer_type={renderer_type}",
        f"--num_workers={num_workers}",
    ]
    if use_gpu:
        command.extend(["--use_gpu", f"--device={device}"])
    if cleanup_eval:
        command.append("--cleanup_eval")

    return BopEvaluationPlan(
        schema_version=SCHEMA_VERSION,
        dry_run=dry_run,
        bop_root=bop_root_path.as_posix(),
        bop_path=bop_path_path.as_posix(),
        dataset_folder=dataset_folder.as_posix(),
        result=result,
        eval_path=eval_path_path.as_posix(),
        targets_filename=targets_filename,
        eval_script=script_path.as_posix(),
        command=command,
        environment={"BOP_PATH": bop_path_path.as_posix()},
    )


def run_bop_toolkit_evaluation(plan: BopEvaluationPlan) -> None:
    env = os.environ.copy()
    env.update(plan.environment)
    subprocess.run(plan.command, check=True, env=env)


def _check_path(
    name: str,
    path: Path,
    *,
    kind: str,
    hint: str,
) -> BopEvaluationCheck:
    ok = path.is_dir() if kind == "directory" else path.is_file()
    return BopEvaluationCheck(
        name=name,
        ok=ok,
        value=path.as_posix(),
        hint=None if ok else hint,
    )


def bop_evaluation_checks(plan: BopEvaluationPlan) -> list[BopEvaluationCheck]:
    models_folder = Path(plan.dataset_folder) / MODELS_DIR
    model_file_count = (
        sum(1 for path in models_folder.glob("obj_*.ply") if path.is_file())
        if models_folder.is_dir()
        else 0
    )
    return [
        _check_path(
            "result_file",
            Path(plan.result.path),
            kind="file",
            hint="Expected the validated BOP19 result CSV to remain present.",
        ),
        _check_path(
            "bop_root",
            Path(plan.bop_root),
            kind="directory",
            hint="Expected the BOP dataset root to exist.",
        ),
        _check_path(
            "dataset_folder",
            Path(plan.dataset_folder),
            kind="directory",
            hint="Expected the BOP dataset folder implied by the result filename.",
        ),
        _check_path(
            "targets_file",
            Path(plan.bop_root) / plan.targets_filename,
            kind="file",
            hint="Expected the BOP19 targets file in the dataset root.",
        ),
        _check_path(
            "models_folder",
            models_folder,
            kind="directory",
            hint="Expected BOP object models under the dataset models folder.",
        ),
        _check_path(
            "models_info",
            models_folder / "models_info.json",
            kind="file",
            hint="Expected BOP model metadata for evaluation metrics.",
        ),
        BopEvaluationCheck(
            name="model_files",
            ok=model_file_count > 0,
            value=model_file_count,
            hint=None
            if model_file_count > 0
            else "Expected at least one BOP model PLY file.",
        ),
        _check_path(
            "eval_script",
            Path(plan.eval_script),
            kind="file",
            hint=(
                "Expected BOP Toolkit eval_bop19_pose.py. Dry-run planning can "
                "succeed without it, but execution requires it."
            ),
        ),
        BopEvaluationCheck(
            name="eval_path_exists",
            ok=Path(plan.eval_path).exists(),
            value=Path(plan.eval_path).as_posix(),
            hint=(
                "The eval output folder is usually created by BOP Toolkit during "
                "non-dry-run execution."
            ),
        ),
    ]


def discover_bop_evaluation_outputs(
    eval_path: str | Path,
    *,
    limit: int = 100,
) -> list[BopEvaluationOutputArtifact]:
    root = Path(eval_path)
    if not root.exists():
        return []
    files = (
        [root]
        if root.is_file()
        else [path for path in root.rglob("*") if path.is_file()]
    )
    artifacts: list[BopEvaluationOutputArtifact] = []
    for path in sorted(files, key=lambda item: item.as_posix())[:limit]:
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.name
        artifacts.append(
            BopEvaluationOutputArtifact(
                path=path.as_posix(),
                relative_path=relative_path,
                size_bytes=path.stat().st_size,
            )
        )
    return artifacts


def _flatten_numeric_scores(value: object, *, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_flatten_numeric_scores(item, prefix=next_prefix))
    elif isinstance(value, bool):
        return {}
    elif isinstance(value, int | float):
        number = float(value)
        if prefix and math.isfinite(number):
            metrics[prefix] = number
    return metrics


def discover_bop_evaluation_scores(
    eval_path: str | Path,
    *,
    limit: int = 20,
) -> BopEvaluationScoreSummary:
    root = Path(eval_path)
    if not root.exists():
        return BopEvaluationScoreSummary(score_file_count=0, metrics={}, files=[])

    candidates = (
        [root]
        if root.is_file() and root.name.startswith("scores") and root.suffix == ".json"
        else [
            path
            for path in root.rglob("scores*.json")
            if path.is_file()
        ]
    )
    files: list[BopEvaluationScoreFile] = []
    for path in sorted(candidates, key=lambda item: item.as_posix())[:limit]:
        try:
            with open(path, "r") as f:
                value = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        metrics = _flatten_numeric_scores(value)
        if not metrics:
            continue
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.name
        files.append(
            BopEvaluationScoreFile(
                path=path.as_posix(),
                relative_path=relative_path,
                metrics=metrics,
            )
        )

    primary_metrics: dict[str, float] = {}
    for score_file in files:
        if score_file.relative_path == "scores_bop19.json":
            primary_metrics = dict(score_file.metrics)
            break
    if not primary_metrics and files:
        primary_metrics = dict(files[0].metrics)

    return BopEvaluationScoreSummary(
        score_file_count=len(files),
        metrics=primary_metrics,
        files=files,
    )


def build_bop_evaluation_report(
    *,
    run_root: str | Path,
    plan: BopEvaluationPlan,
    plan_path: str | Path,
    status: str,
    message: str,
    error: str | None = None,
) -> BopEvaluationReport:
    return BopEvaluationReport(
        schema_version=BOP_EVALUATION_REPORT_SCHEMA_VERSION,
        run_root=Path(run_root).as_posix(),
        status=status,
        dry_run=plan.dry_run,
        plan_path=Path(plan_path).as_posix(),
        result=plan.result,
        eval_path=plan.eval_path,
        command=list(plan.command),
        environment=dict(plan.environment),
        checks=bop_evaluation_checks(plan),
        output_artifacts=discover_bop_evaluation_outputs(plan.eval_path),
        score_summary=discover_bop_evaluation_scores(plan.eval_path),
        message=message,
        error=error,
    )


def write_bop_evaluation_report(
    run_root: str | Path,
    report: BopEvaluationReport,
) -> Path:
    path = Path(run_root) / BOP_EVALUATION_REPORT
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path
