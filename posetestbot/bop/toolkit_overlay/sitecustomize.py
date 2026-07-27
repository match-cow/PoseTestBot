"""Runtime-only generic-dataset adapter for the pinned BOP Toolkit.

Python imports ``sitecustomize`` automatically.  The evaluation worker puts
this directory first on ``PYTHONPATH`` and supplies an immutable adapter JSON
path.  The upstream checkout itself remains unmodified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_VARIABLE = "POSETESTBOT_BOP_ADAPTER_CONFIG"


def _install() -> None:
    config_path = os.environ.get(CONFIG_VARIABLE)
    if not config_path:
        return
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "posetestbot_bop_toolkit_adapter.v1":
        raise RuntimeError("Unsupported PoseTestBot BOP Toolkit adapter config")

    from bop_toolkit_lib import dataset_params

    dataset_alias = str(config["dataset_alias"])
    bop_root = Path(config["bop_root"]).resolve()
    configured_split = str(config["split"])
    image_size = tuple(int(value) for value in config["image_size"])
    scene_ids = [int(value) for value in config["scene_ids"]]
    object_ids = [int(value) for value in config["object_ids"]]
    original_get_split_params = dataset_params.get_split_params
    original_get_model_params = dataset_params.get_model_params

    def get_split_params(datasets_path, dataset_name, split, split_type=None):
        if dataset_name != dataset_alias:
            return original_get_split_params(
                datasets_path, dataset_name, split, split_type
            )
        if split != configured_split or split_type is not None:
            raise ValueError(
                "PoseTestBot BOP adapter received an unexpected split or split type"
            )
        split_path = bop_root / split
        return {
            "name": dataset_alias,
            "split": split,
            "split_type": None,
            "base_path": bop_root.as_posix(),
            "split_path": split_path.as_posix(),
            "scene_ids": scene_ids,
            "im_size": image_size,
            "im_modalities": ["rgb", "depth"],
            "eval_sensor": None,
            "eval_modality": None,
            "supported_error_types": ["vsd", "mssd", "mspd"],
            "rgb_tpath": (
                split_path / "{scene_id:06d}" / "rgb" / "{im_id:06d}.png"
            ).as_posix(),
            "depth_tpath": (
                split_path / "{scene_id:06d}" / "depth" / "{im_id:06d}.png"
            ).as_posix(),
            "scene_camera_tpath": (
                split_path / "{scene_id:06d}" / "scene_camera.json"
            ).as_posix(),
            "scene_gt_tpath": (
                split_path / "{scene_id:06d}" / "scene_gt.json"
            ).as_posix(),
            "scene_gt_info_tpath": (
                split_path / "{scene_id:06d}" / "scene_gt_info.json"
            ).as_posix(),
            "scene_gt_coco_tpath": (
                split_path / "{scene_id:06d}" / "scene_gt_coco.json"
            ).as_posix(),
            "mask_tpath": (
                split_path / "{scene_id:06d}" / "mask" / "{im_id:06d}_{gt_id:06d}.png"
            ).as_posix(),
            "mask_visib_tpath": (
                split_path
                / "{scene_id:06d}"
                / "mask_visib"
                / "{im_id:06d}_{gt_id:06d}.png"
            ).as_posix(),
        }

    def get_model_params(datasets_path, dataset_name, model_type=None):
        if dataset_name != dataset_alias:
            return original_get_model_params(datasets_path, dataset_name, model_type)
        model_folder = "models" if model_type is None else f"models_{model_type}"
        models_root = bop_root / model_folder
        return {
            "obj_ids": object_ids,
            "symmetric_obj_ids": [],
            "model_tpath": (models_root / "obj_{obj_id:06d}.ply").as_posix(),
            "models_info_path": (models_root / "models_info.json").as_posix(),
        }

    dataset_params.get_split_params = get_split_params
    dataset_params.get_model_params = get_model_params
    os.environ["POSETESTBOT_BOP_ADAPTER_LOADED"] = dataset_alias


_install()
