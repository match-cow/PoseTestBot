from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from posetestbot.bop.writer import copy_bop_instance_models, export_sensor_scene_to_bop
from posetestbot.pipeline.run_config import (
    create_run_config,
    load_run_config_for_run_root,
    write_run_config,
)
from posetestbot.pose_templates import adapter
from posetestbot.pose_templates.catalog import (
    import_catalog_object,
    load_catalog,
    set_catalog_object_state,
)
from posetestbot.pose_templates.library import (
    build_template_preview,
    generate_template_bundle,
    set_template_archive_state,
    validate_template_bundle,
)
from posetestbot.pose_templates.selection import (
    PoseTemplateSelectionConflict,
    load_pose_template_selection,
    prepare_object_instances,
    select_pose_template,
)
from posetestbot.pose_templates.transforms import matrix_from_xyz_rpy


def mesh_file(path: Path, *, extents=(20, 10, 10)) -> Path:
    path.write_bytes(trimesh.creation.box(extents=extents).export(file_type=path.suffix[1:]))
    return path


def managed_box(tmp_path: Path) -> tuple[Path, dict]:
    catalog = tmp_path / "catalog"
    record = import_catalog_object(
        name="Box",
        description="fixture",
        cad_path=mesh_file(tmp_path / "box.stl"),
        catalog_root=catalog,
    )
    return catalog, record


def template_configuration(catalog_uuid: str) -> dict:
    return {
        "display_name": "Two boxes",
        "description": "duplicate model fixture",
        "page": {"size": "A3", "orientation": "landscape"},
        "print_compensation": {"x_scale": 1.01, "y_scale": 0.99},
        "instances": [
            {
                "instance_uuid": "11111111-1111-4111-8111-111111111111",
                "catalog_uuid": catalog_uuid,
                "pose": {
                    "x_mm": 40,
                    "y_mm": 40,
                    "z_mm": 0,
                    "roll_deg": 0,
                    "pitch_deg": 0,
                    "yaw_deg": 15,
                },
            },
            {
                "instance_uuid": "22222222-2222-4222-8222-222222222222",
                "catalog_uuid": catalog_uuid,
                "pose": {
                    "x_mm": 90,
                    "y_mm": 50,
                    "z_mm": 0,
                    "roll_deg": 20,
                    "pitch_deg": 10,
                    "yaw_deg": -25,
                },
            },
        ],
    }


def test_source_status_states_and_private_import_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = adapter.verify_posetemplatecreator_checkout(tmp_path / "missing")
    assert missing["status"] == "missing"

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    for relative in adapter._REQUIRED_FILES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    monkeypatch.setattr(
        adapter,
        "_git",
        lambda _root, *arguments: (
            "wrong" if arguments[0] == "rev-parse" else ""
        ),
    )
    assert adapter.verify_posetemplatecreator_checkout(checkout)["status"] == "revision_mismatch"
    monkeypatch.setattr(
        adapter,
        "_git",
        lambda _root, *arguments: (
            adapter.POSETEMPLATECREATOR_REVISION
            if arguments[0] == "rev-parse"
            else " M backend/mesh.py"
        ),
    )
    assert adapter.verify_posetemplatecreator_checkout(checkout)["status"] == "dirty"

    monkeypatch.undo()
    backend = adapter.load_posetemplatecreator_backend()
    assert backend.revision == adapter.POSETEMPLATECREATOR_REVISION
    assert "backend" not in sys.modules


def test_catalog_is_transactional_archivable_and_never_reuses_ids(tmp_path: Path) -> None:
    catalog, first = managed_box(tmp_path)
    assert first["obj_id"] == 1
    assert first["source_sha256"] != first["canonical_ply_sha256"]
    archived = set_catalog_object_state(first["catalog_uuid"], state="archived", catalog_root=catalog)
    assert archived["state"] == "archived"
    restored = set_catalog_object_state(first["catalog_uuid"], state="active", catalog_root=catalog)
    assert restored["obj_id"] == 1
    second = import_catalog_object(
        name="Other",
        cad_path=mesh_file(tmp_path / "other.ply", extents=(8, 8, 8)),
        catalog_root=catalog,
    )
    assert second["obj_id"] == 2
    value = load_catalog(catalog)
    assert value["next_obj_id"] == 3
    assert len(list((catalog / "revisions").glob("*.json"))) >= 4


def test_full_pose_preview_compensation_and_immutable_bundle(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    config = template_configuration(record["catalog_uuid"])
    preview = build_template_preview(config, catalog_root=catalog)
    assert preview["schema_version"] == "pose_template_preview.v1"
    assert preview["valid"] is True
    assert len(preview["instances"]) == 2
    first = preview["instances"][0]
    np.testing.assert_allclose(
        first["pose_template_from_object"]["matrix"],
        matrix_from_xyz_rpy(**config["instances"][0]["pose"]),
    )
    nominal_point = first["nominal_contours"][0][0]
    print_point = first["compensated_contours"][0][0]
    assert print_point["x_mm"] == pytest.approx(nominal_point["x_mm"] * 1.01)
    assert print_point["y_mm"] == pytest.approx(nominal_point["y_mm"] * 0.99)

    library = tmp_path / "library"
    bundle = generate_template_bundle(config, catalog_root=catalog, library_root=library)
    assert bundle["schema_version"] == "pose_template_bundle.v1"
    assert (Path(bundle["bundle_path"]) / "pose_template.pdf").read_bytes().startswith(b"%PDF")
    set_catalog_object_state(record["catalog_uuid"], state="archived", catalog_root=catalog)
    set_template_archive_state(bundle["template_uuid"], state="archived", library_root=library)
    assert validate_template_bundle(bundle["bundle_path"], library_root=library)["archive"]["state"] == "archived"
    # Historical snapshots stay valid after catalog/template archival.
    assert len(validate_template_bundle(bundle["bundle_path"], library_root=library)["instances"]) == 2

    manifest = Path(bundle["bundle_path"]) / "pose_template_bundle.json"
    tampered = json.loads(manifest.read_text())
    tampered["display_name"] = "tampered"
    manifest.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_template_bundle(bundle["bundle_path"], library_root=library)


def test_selection_resolution_blockers_and_object_instances(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    run = tmp_path / "run"
    run.mkdir()
    config = create_run_config(run_root=run, dataset_mode="pose_template")
    write_run_config(run, config)
    selection = select_pose_template(
        run,
        bundle["template_uuid"],
        placement={
            "matrix": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
        },
        confirmed=True,
        operator="pytest",
        library_root=library,
    )
    assert selection["instances"][0]["template_base_from_object"]["translation_mm"] == pytest.approx(
        [50, 60, 30]
    )
    loaded_config = load_run_config_for_run_root(run)
    assert loaded_config["schema_version"] == "run_config.v2"
    assert loaded_config["pose_template"]["template_uuid"] == bundle["template_uuid"]
    objects = prepare_object_instances(run)
    assert len(objects["instances"]) == 2
    assert {item["obj_id"] for item in objects["instances"]} == {record["obj_id"]}

    (run / "blenderproc_render_plan.json").write_text("{}")
    with pytest.raises(PoseTemplateSelectionConflict) as conflict:
        select_pose_template(
            run,
            bundle["template_uuid"],
            placement={"matrix": np.eye(4).tolist()},
            confirmed=True,
            operator="pytest",
            library_root=library,
        )
    assert "blenderproc_render_plan.json" in conflict.value.blockers
    assert load_pose_template_selection(run)["bundle_sha256"] == bundle["bundle_sha256"]


def test_bop_model_export_deduplicates_duplicate_instances(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]), catalog_root=catalog, library_root=library
    )
    run = tmp_path / "run"
    run.mkdir()
    select_pose_template(
        run,
        bundle["template_uuid"],
        placement={"matrix": np.eye(4).tolist()},
        confirmed=True,
        operator="pytest",
        library_root=library,
    )
    objects = prepare_object_instances(run)
    models = copy_bop_instance_models(tmp_path / "bop", run, objects)
    assert len(models) == 1
    assert models[0].obj_id == record["obj_id"]
    assert len(list((tmp_path / "bop" / "models").glob("obj_*.ply"))) == 1


def test_bop_export_preserves_duplicate_instance_identity_and_target_count(
    tmp_path: Path,
) -> None:
    sensor = tmp_path / "realsense_123"
    (sensor / "rgb").mkdir(parents=True)
    (sensor / "depth").mkdir()
    (sensor / "masks").mkdir()
    output = sensor / "blenderproc" / "output"
    (output / "mask_visib").mkdir(parents=True)
    assert cv2.imwrite(
        (sensor / "rgb" / "000000.png").as_posix(),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    assert cv2.imwrite(
        (sensor / "depth" / "000000.png").as_posix(),
        np.ones((8, 8), dtype=np.uint16),
    )
    (sensor / "cam_K.txt").write_text("100 0 4 0 100 4 0 0 1\n")
    instances = [
        {
            "instance_uuid": f"{index}{index}{index}{index}{index}{index}{index}{index}-1111-4111-8111-111111111111",
            "catalog_uuid": "33333333-3333-4333-8333-333333333333",
            "obj_id": 7,
            "name": "Clamp",
            "texture": None,
        }
        for index in (1, 2)
    ]
    prepared = [
        {
            "instance_uuid": item["instance_uuid"],
            "catalog_uuid": item["catalog_uuid"],
            "obj_id": item["obj_id"],
            "name": item["name"],
            "mesh": f"{item['instance_uuid']}.ply",
            "transform": f"{item['instance_uuid']}.npy",
            "texture": None,
        }
        for item in instances
    ]
    annotations = [
        {
            "obj_id": 7,
            "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "cam_t_m2c": [index * 10, 0, 100],
        }
        for index in (1, 2)
    ]
    (output / "scene_gt.json").write_text(json.dumps({"0": annotations}))
    (output / "posetestbot_render_instances.json").write_text(
        json.dumps(
            {
                "schema_version": "posetestbot_render_instances.v1",
                "blenderproc_version": "2.8.0",
                "supported_blenderproc_version": "2.8.0",
                "identity_contract": "bop_gt_index_matches_loaded_instance_order.v1",
                "instances": prepared,
                "frames": {
                    "0": [
                        {
                            "gt_id": index,
                            "obj_id": 7,
                            "instance_uuid": item["instance_uuid"],
                            "catalog_uuid": item["catalog_uuid"],
                        }
                        for index, item in enumerate(instances)
                    ]
                },
            }
        )
    )
    for index in range(2):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1 + index : 4 + index, 1 + index : 4 + index] = 255
        filename = f"000000_{index:06d}.png"
        assert cv2.imwrite((sensor / "masks" / filename).as_posix(), mask)
        assert cv2.imwrite((output / "mask_visib" / filename).as_posix(), mask)

    exported = export_sensor_scene_to_bop(
        sensor,
        tmp_path / "bop",
        object_name_to_id={item["instance_uuid"]: 7 for item in instances},
        template_instances=instances,
    )

    assert exported.targets == [{"scene_id": 1, "im_id": 0, "obj_id": 7, "inst_count": 2}]
    assert [item["instance_uuid"] for item in exported.instance_map] == [
        item["instance_uuid"] for item in instances
    ]
    assert len(list((tmp_path / "bop" / "test" / "000001" / "mask").glob("*.png"))) == 2
