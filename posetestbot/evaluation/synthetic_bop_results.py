"""Synthetic BOP19 result rows for hardware-free rewrite validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posetestbot.evaluation.bop_results import (
    BopResultExportManifest,
    BopResultFile,
    BopResultRow,
    write_bop19_result_csv,
    write_bop_result_export_manifest,
)
from posetestbot.io.artifacts import (
    BOP_DIR,
    BOP_EXPORT_MANIFEST,
    BOP_FRAME_MAP_JSON,
    BOP_RESULT_EXPORT_MANIFEST,
    BOP_TARGETS_BOP19,
    RESULTS_DIR,
)
from posetestbot.io.manifest import (
    load_or_create_run_manifest,
    upsert_stage,
    write_run_manifest,
)


SCHEMA_VERSION = "synthetic_bop_results.v1"
DEFAULT_METHOD = "synthetic"
DEFAULT_SCORE = 1.0
DEFAULT_TIME = -1.0


def _load_json_object(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _first_object_id(export_manifest: dict[str, Any], object_name: str | None) -> int:
    object_models = export_manifest.get("object_models")
    if not isinstance(object_models, list) or not object_models:
        raise ValueError("BOP export manifest does not contain object model metadata.")
    if object_name:
        for model in object_models:
            if isinstance(model, dict) and model.get("object_name") == object_name:
                return int(model["obj_id"])
        raise ValueError(f"Object {object_name!r} not found in BOP export manifest.")
    first = object_models[0]
    if not isinstance(first, dict) or first.get("obj_id") is None:
        raise ValueError("First BOP object model is missing obj_id.")
    return int(first["obj_id"])


def _frame_ids(export: dict[str, Any]) -> list[int]:
    scene_folder = Path(str(export["scene_folder"]))
    frame_map_path = scene_folder / BOP_FRAME_MAP_JSON
    if frame_map_path.is_file():
        frame_map = _load_json_object(frame_map_path)
        return sorted(int(key) for key in frame_map)
    rgb_count = int(export.get("rgb_count") or 0)
    return list(range(rgb_count))


def export_synthetic_bop_results(
    *,
    run_root: str | Path,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    method: str = DEFAULT_METHOD,
    object_name: str | None = None,
    score: float = DEFAULT_SCORE,
    time_s: float = DEFAULT_TIME,
) -> BopResultExportManifest:
    root = Path(run_root)
    bop_root_path = Path(bop_root) if bop_root is not None else root / BOP_DIR
    output_folder_path = (
        Path(output_folder) if output_folder is not None else root / RESULTS_DIR / BOP_DIR
    )
    dataset = dataset_name or bop_root_path.name
    export_manifest_path = bop_root_path / BOP_EXPORT_MANIFEST
    export_manifest = _load_json_object(export_manifest_path)
    exports = export_manifest.get("exports")
    if not isinstance(exports, list) or not exports:
        raise ValueError("BOP export manifest does not contain exported scenes.")
    obj_id = _first_object_id(export_manifest, object_name)

    rows_by_split: dict[str, list[BopResultRow]] = {}
    sources_by_split: dict[str, list[str]] = {}
    targets_by_split: dict[str, list[dict[str, int]]] = {}
    for export in exports:
        if not isinstance(export, dict):
            continue
        split = str(export.get("split") or "test")
        scene_id = int(export["scene_id"])
        scene_folder = Path(str(export["scene_folder"]))
        source = (scene_folder / BOP_FRAME_MAP_JSON).as_posix()
        for im_id in _frame_ids(export):
            rows_by_split.setdefault(split, []).append(
                BopResultRow(
                    scene_id=scene_id,
                    im_id=im_id,
                    obj_id=obj_id,
                    score=float(score),
                    R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    t=[0.0, 0.0, 1000.0],
                    time=float(time_s),
                    source_pose_file=source,
                )
            )
            targets_by_split.setdefault(split, []).append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "obj_id": obj_id,
                    "inst_count": 1,
                }
            )
        sources_by_split.setdefault(split, []).append(source)

    result_files: list[BopResultFile] = []
    for split, rows in sorted(rows_by_split.items()):
        if not rows:
            continue
        result_path = output_folder_path / f"{method}_{dataset}-{split}.csv"
        write_bop19_result_csv(result_path, rows)
        result_files.append(
            BopResultFile(
                path=result_path.as_posix(),
                filename=result_path.name,
                method=method,
                dataset=dataset,
                split=split,
                result_id=None,
                row_count=len(rows),
                source_outputs=sources_by_split[split],
            )
        )

    if not result_files:
        raise ValueError("No synthetic BOP result rows were written.")

    for split, targets in targets_by_split.items():
        target_name = BOP_TARGETS_BOP19 if split == "test" else f"{split}_targets_bop19.json"
        target_path = bop_root_path / target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(sorted(targets, key=lambda item: (item["scene_id"], item["im_id"], item["obj_id"])), f, indent=2)
            f.write("\n")

    return BopResultExportManifest(
        schema_version="bop_result_export_manifest.v1",
        run_root=root.as_posix(),
        bop_root=bop_root_path.as_posix(),
        input_folder=bop_root_path.as_posix(),
        output_folder=output_folder_path.as_posix(),
        dataset_name=dataset,
        source_type=method,
        translation_scale_to_mm=1.0,
        results=result_files,
    )


def write_synthetic_bop_results_with_manifest(
    *,
    run_root: str | Path,
    bop_root: str | Path | None = None,
    output_folder: str | Path | None = None,
    dataset_name: str | None = None,
    method: str = DEFAULT_METHOD,
    object_name: str | None = None,
    score: float = DEFAULT_SCORE,
    time_s: float = DEFAULT_TIME,
) -> tuple[Path, BopResultExportManifest]:
    root = Path(run_root)
    manifest = load_or_create_run_manifest(root)
    upsert_stage(manifest, name="synthetic_bop_results", status="running")
    write_run_manifest(manifest, root)
    try:
        result_manifest = export_synthetic_bop_results(
            run_root=root,
            bop_root=bop_root,
            output_folder=output_folder,
            dataset_name=dataset_name,
            method=method,
            object_name=object_name,
            score=score,
            time_s=time_s,
        )
        path = write_bop_result_export_manifest(root, result_manifest)
    except Exception as exc:
        upsert_stage(
            manifest,
            name="synthetic_bop_results",
            status="failed",
            message=str(exc),
        )
        write_run_manifest(manifest, root)
        raise

    artifacts: dict[str, Path] = {BOP_RESULT_EXPORT_MANIFEST: path}
    for result in result_manifest.results:
        artifacts[result.filename] = Path(result.path)
    upsert_stage(
        manifest,
        name="synthetic_bop_results",
        status="succeeded",
        artifacts=artifacts,
        run_root=root,
        message=(
            f"Wrote {sum(result.row_count for result in result_manifest.results)} "
            "synthetic BOP result row(s)."
        ),
    )
    write_run_manifest(manifest, root)
    return path, result_manifest
