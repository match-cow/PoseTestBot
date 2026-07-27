"""Generate official BOP masks and visibility information for staged exports.

The public runner executes this module inside PoseTestBot's isolated, pinned
BOP Toolkit environment.  The compute path intentionally follows the official
``calc_gt_masks.py`` and ``calc_gt_info.py`` algorithms, including their
three-times-larger canvas for truncated-object statistics and BOP19's 15 mm
visibility tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from posetestbot.io.atomic import atomic_write_json


TOOLKIT_REVISION = "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
GENERATION_REPORT = "posetestbot_gt_generation.json"
VISIBILITY_DELTA_MM = 15.0
RENDERER_TYPE = "vispy"


def bop_gt_output_sha256(paths: Iterable[Path], *, root: Path) -> str:
    """Hash the exact official mask/info bundle with stable relative paths."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _compute_masks_and_info(args: argparse.Namespace) -> dict[str, Any]:
    # Imports are deliberately local: the normal PoseTestBot environment must
    # not need the pinned toolkit's NumPy<2 dependency set.
    from bop_toolkit_lib import dataset_params, inout, misc, visibility
    from bop_toolkit_lib.rendering import renderer

    loaded_alias = os.environ.get("POSETESTBOT_BOP_ADAPTER_LOADED")
    if loaded_alias != args.dataset_alias:
        raise RuntimeError(
            "PoseTestBot custom-dataset adapter was not loaded for mask generation"
        )
    if args.toolkit_revision != TOOLKIT_REVISION:
        raise RuntimeError("BOP mask generation did not receive the pinned revision")

    bop_root = Path(args.bop_root).resolve()
    dp_split = dataset_params.get_split_params(
        Path(args.datasets_path).resolve(),
        args.dataset_alias,
        args.split,
        None,
    )
    dp_model = dataset_params.get_model_params(
        Path(args.datasets_path).resolve(),
        args.dataset_alias,
        None,
    )
    width, height = (int(value) for value in dp_split["im_size"])
    render_width, render_height = 3 * width, 3 * height
    x_offset, y_offset = width, height
    render = renderer.create_renderer(
        render_width,
        render_height,
        renderer_type=args.renderer_type,
        mode="depth",
    )
    for obj_id in dp_model["obj_ids"]:
        render.add_object(
            int(obj_id),
            dp_model["model_tpath"].format(obj_id=int(obj_id)),
        )

    scene_summaries: dict[str, dict[str, int]] = {}
    output_paths: list[Path] = []
    for scene_id in sorted(int(value) for value in dp_split["scene_ids"]):
        scene_camera = inout.load_scene_camera(
            dp_split["scene_camera_tpath"].format(scene_id=scene_id)
        )
        scene_gt = inout.load_scene_gt(
            dp_split["scene_gt_tpath"].format(scene_id=scene_id)
        )
        mask_folder = Path(
            dp_split["mask_tpath"].format(scene_id=scene_id, im_id=0, gt_id=0)
        ).parent
        mask_visib_folder = Path(
            dp_split["mask_visib_tpath"].format(
                scene_id=scene_id,
                im_id=0,
                gt_id=0,
            )
        ).parent
        for folder in (mask_folder, mask_visib_folder):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)

        scene_gt_info: dict[int, list[dict[str, Any]]] = {}
        annotation_count = 0
        for im_id in sorted(scene_gt):
            camera = scene_camera[im_id]
            intrinsic = camera["cam_K"]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            depth_path = Path(
                dp_split["depth_tpath"].format(scene_id=scene_id, im_id=im_id)
            )
            captured_depth = inout.load_depth(depth_path)
            captured_depth *= camera["depth_scale"]
            captured_distance = misc.depth_im_to_dist_im_fast(
                captured_depth,
                intrinsic,
            )
            scene_gt_info[im_id] = []

            for gt_id, gt in enumerate(scene_gt[im_id]):
                rendered_depth_large = render.render_object(
                    gt["obj_id"],
                    gt["cam_R_m2c"],
                    gt["cam_t_m2c"],
                    fx,
                    fy,
                    cx + x_offset,
                    cy + y_offset,
                )["depth"]
                rendered_depth = rendered_depth_large[
                    y_offset : y_offset + height,
                    x_offset : x_offset + width,
                ]
                rendered_distance = misc.depth_im_to_dist_im_fast(
                    rendered_depth,
                    intrinsic,
                )
                full_mask_large = rendered_depth_large > 0
                full_mask = rendered_distance > 0
                visible_mask = visibility.estimate_visib_mask_gt(
                    captured_distance,
                    rendered_distance,
                    args.delta_mm,
                    visib_mode="bop19",
                )

                px_count_all = int(np.sum(full_mask_large))
                px_count_valid = int(np.sum(captured_distance[full_mask] > 0))
                px_count_visib = int(np.sum(visible_mask))
                visibility_fraction = (
                    px_count_visib / float(px_count_all) if px_count_all else 0.0
                )
                bbox_obj = [-1, -1, -1, -1]
                bbox_visib = [-1, -1, -1, -1]
                if px_count_visib > 0:
                    ys, xs = full_mask_large.nonzero()
                    ys -= y_offset
                    xs -= x_offset
                    bbox_obj = [
                        int(value)
                        for value in misc.calc_2d_bbox(
                            xs,
                            ys,
                            (width, height),
                        )
                    ]
                    ys, xs = visible_mask.nonzero()
                    bbox_visib = [
                        int(value)
                        for value in misc.calc_2d_bbox(
                            xs,
                            ys,
                            (width, height),
                        )
                    ]
                scene_gt_info[im_id].append(
                    {
                        "bbox_obj": bbox_obj,
                        "bbox_visib": bbox_visib,
                        "px_count_all": px_count_all,
                        "px_count_valid": px_count_valid,
                        "px_count_visib": px_count_visib,
                        "visib_fract": float(visibility_fraction),
                    }
                )

                mask_path = Path(
                    dp_split["mask_tpath"].format(
                        scene_id=scene_id,
                        im_id=im_id,
                        gt_id=gt_id,
                    )
                )
                mask_visib_path = Path(
                    dp_split["mask_visib_tpath"].format(
                        scene_id=scene_id,
                        im_id=im_id,
                        gt_id=gt_id,
                    )
                )
                inout.save_im(mask_path, 255 * full_mask.astype(np.uint8))
                inout.save_im(
                    mask_visib_path,
                    255 * visible_mask.astype(np.uint8),
                )
                output_paths.extend((mask_path, mask_visib_path))
                annotation_count += 1

        info_path = Path(dp_split["scene_gt_info_tpath"].format(scene_id=scene_id))
        inout.save_json(info_path, scene_gt_info)
        output_paths.append(info_path)
        scene_summaries[str(scene_id)] = {
            "image_count": len(scene_gt),
            "annotation_count": annotation_count,
            "full_mask_count": annotation_count,
            "visible_mask_count": annotation_count,
        }

    report = {
        "schema_version": "posetestbot_bop_gt_generation.v1",
        "annotation_mode": "pose_and_masks",
        "pose_source": "blenderproc_scene_gt",
        "generator": "official_bop_toolkit_algorithms",
        "toolkit_revision": args.toolkit_revision,
        "toolkit_clean_checkout": True,
        "upstream_algorithms": [
            "scripts/calc_gt_masks.py",
            "scripts/calc_gt_info.py",
        ],
        "renderer_type": args.renderer_type,
        "visibility_delta_mm": float(args.delta_mm),
        "visibility_mode": "bop19",
        "depth_source": "exported_captured_depth",
        "artifact_path": GENERATION_REPORT,
        "split": args.split,
        "scenes": scene_summaries,
        "output_sha256": bop_gt_output_sha256(output_paths, root=bop_root),
    }
    atomic_write_json(Path(args.report_path), report)
    return report


def _checkout_revision(toolkit_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", toolkit_root.as_posix(), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _checkout_is_clean(toolkit_root: Path) -> bool:
    try:
        return not subprocess.run(
            [
                "git",
                "-C",
                toolkit_root.as_posix(),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False


def run_official_bop_mask_generation(
    bop_root: str | Path,
    *,
    split: str,
    scene_ids: Iterable[int],
    object_ids: Iterable[int],
    image_size: tuple[int, int],
    app_root: str | Path | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Invoke the pinned isolated toolkit against a staged BOP dataset."""

    root = (
        Path(app_root).resolve()
        if app_root is not None
        else Path(__file__).resolve().parents[2]
    )
    bop_root = Path(bop_root).resolve()
    toolkit_root = root / "third_party" / "bop_toolkit"
    runtime_root = root / "tools" / "bop_toolkit_runtime"
    if _checkout_revision(toolkit_root) != TOOLKIT_REVISION or not _checkout_is_clean(
        toolkit_root
    ):
        raise RuntimeError(
            "Pinned, clean BOP Toolkit checkout is unavailable; run "
            "bash scripts/install.sh --with-bop-toolkit"
        )
    if not (runtime_root / ".venv" / "bin" / "python").is_file():
        raise RuntimeError(
            "Pinned BOP Toolkit runtime is unavailable; run "
            "bash scripts/install.sh --with-bop-toolkit"
        )

    scene_ids = sorted({int(value) for value in scene_ids})
    object_ids = sorted({int(value) for value in object_ids})
    if not scene_ids:
        raise ValueError("Official BOP mask generation requires at least one scene")
    if not object_ids:
        raise ValueError("Official BOP mask generation requires exported object models")
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("Official BOP mask generation requires a valid image size")

    dataset_alias = f"ptbgt_{uuid.uuid4().hex}"
    adapter_path = bop_root / f".{dataset_alias}.adapter.json"
    report_path = bop_root / GENERATION_REPORT
    report_path.unlink(missing_ok=True)
    atomic_write_json(
        adapter_path,
        {
            "schema_version": "posetestbot_bop_toolkit_adapter.v1",
            "adapter_revision": "posetestbot_bop19_dataset_adapter.v1",
            "dataset_alias": dataset_alias,
            "bop_root": bop_root.as_posix(),
            "split": split,
            "image_size": [int(value) for value in image_size],
            "scene_ids": scene_ids,
            "object_ids": object_ids,
        },
    )

    command = [
        "uv",
        "run",
        "--project",
        runtime_root.as_posix(),
        "--no-sync",
        "python",
        "-m",
        "posetestbot.bop.mask_driver",
        "--bop-root",
        bop_root.as_posix(),
        "--datasets-path",
        bop_root.parent.as_posix(),
        "--dataset-alias",
        dataset_alias,
        "--split",
        split,
        "--renderer-type",
        RENDERER_TYPE,
        "--delta-mm",
        str(VISIBILITY_DELTA_MM),
        "--toolkit-revision",
        TOOLKIT_REVISION,
        "--report-path",
        report_path.as_posix(),
    ]
    overlay = root / "posetestbot" / "bop" / "toolkit_overlay"
    python_path = os.pathsep.join(
        item
        for item in (
            overlay.as_posix(),
            root.as_posix(),
            os.environ.get("PYTHONPATH", ""),
        )
        if item
    )
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.update(
        {
            "PYTHONPATH": python_path,
            "POSETESTBOT_BOP_ADAPTER_CONFIG": adapter_path.as_posix(),
            "BOP_PATH": bop_root.parent.as_posix(),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache"),
        }
    )
    environment.setdefault("EGL_PLATFORM", "surfaceless")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        command_runner(
            command,
            cwd=root,
            env=environment,
            check=True,
        )
    finally:
        adapter_path.unlink(missing_ok=True)

    if not report_path.is_file():
        raise RuntimeError("Official BOP mask generator produced no provenance report")
    report = json.loads(report_path.read_text())
    expected = {
        "schema_version": "posetestbot_bop_gt_generation.v1",
        "annotation_mode": "pose_and_masks",
        "pose_source": "blenderproc_scene_gt",
        "generator": "official_bop_toolkit_algorithms",
        "toolkit_revision": TOOLKIT_REVISION,
        "toolkit_clean_checkout": True,
        "upstream_algorithms": [
            "scripts/calc_gt_masks.py",
            "scripts/calc_gt_info.py",
        ],
        "renderer_type": RENDERER_TYPE,
        "visibility_delta_mm": VISIBILITY_DELTA_MM,
        "visibility_mode": "bop19",
        "depth_source": "exported_captured_depth",
        "artifact_path": GENERATION_REPORT,
        "split": split,
    }
    if not isinstance(report, dict) or any(
        report.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("Official BOP mask provenance report is inconsistent")
    if set(report.get("scenes", {})) != {str(value) for value in scene_ids}:
        raise RuntimeError("Official BOP mask report does not cover every staged scene")
    output_sha256 = report.get("output_sha256")
    if (
        not isinstance(output_sha256, str)
        or len(output_sha256) != 64
        or any(character not in "0123456789abcdef" for character in output_sha256)
    ):
        raise RuntimeError("Official BOP mask report has an invalid output hash")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop-root", required=True)
    parser.add_argument("--datasets-path", required=True)
    parser.add_argument("--dataset-alias", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--renderer-type", default=RENDERER_TYPE)
    parser.add_argument("--delta-mm", type=float, default=VISIBILITY_DELTA_MM)
    parser.add_argument("--toolkit-revision", required=True)
    parser.add_argument("--report-path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.renderer_type != RENDERER_TYPE:
        raise ValueError("PoseTestBot GT generation requires the pinned vispy renderer")
    if args.delta_mm != VISIBILITY_DELTA_MM:
        raise ValueError("PoseTestBot GT generation requires BOP19's 15 mm delta")
    _compute_masks_and_info(args)


if __name__ == "__main__":
    main()
