from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from posetestbot.bop.writer import copy_bop_instance_models, export_sensor_scene_to_bop
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    load_run_config_for_run_root,
    write_run_config,
)
from posetestbot.pose_templates import adapter
from posetestbot.pose_templates.catalog import (
    correct_catalog_object_units,
    import_catalog_object,
    load_catalog,
    set_catalog_object_state,
)
from posetestbot.pose_templates.library import (
    THUMBNAIL_MAX_CONTOURS,
    THUMBNAIL_MAX_POINTS,
    _hash_json,
    build_template_preview,
    build_template_thumbnail,
    clone_template_configuration,
    generate_template_bundle,
    load_template_thumbnail,
    set_template_archive_state,
    validate_template_bundle,
)
from posetestbot.pose_templates.orientations import (
    ORIENTATION_THUMBNAIL_FILENAME,
    ORIENTATION_THUMBNAIL_MAX_BYTES,
    OrientationAnalysisStaleError,
    analyze_catalog_orientations,
    load_catalog_orientation_analysis,
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


def test_legacy_full_pose_preview_compensation_and_immutable_bundle(
    tmp_path: Path,
) -> None:
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
    # Upstream scales template content about the physical page centre. A3
    # landscape is 420 x 297 mm and the template origin is at (15, 15) mm.
    assert print_point["x_mm"] == pytest.approx(
        210 + (15 + nominal_point["x_mm"] - 210) * 1.01 - 15
    )
    assert print_point["y_mm"] == pytest.approx(
        148.5 + (15 + nominal_point["y_mm"] - 148.5) * 0.99 - 15
    )

    library = tmp_path / "library"
    bundle = generate_template_bundle(config, catalog_root=catalog, library_root=library)
    assert bundle["schema_version"] == "pose_template_bundle.v1"
    assert (Path(bundle["bundle_path"]) / "pose_template.pdf").read_bytes().startswith(b"%PDF")
    thumbnail_path = Path(bundle["bundle_path"]) / "pose_template_thumbnail.json"
    thumbnail_bytes = thumbnail_path.read_bytes()
    thumbnail = json.loads(thumbnail_bytes)
    assert thumbnail["schema_version"] == "pose_template_thumbnail.v1"
    assert thumbnail["template_uuid"] == bundle["template_uuid"]
    assert bundle["files"]["thumbnail"] == "pose_template_thumbnail.json"
    assert hashlib.sha256(thumbnail_bytes).hexdigest() == bundle["hashes"]["thumbnail"]
    thumbnail_path.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="thumbnail is missing or was modified"):
        validate_template_bundle(bundle["bundle_path"], library_root=library)
    thumbnail_path.write_bytes(thumbnail_bytes)
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


def test_template_thumbnail_is_deterministically_bounded_and_keeps_every_primary(
    tmp_path: Path,
) -> None:
    del tmp_path

    def contour(radius: float, count: int) -> list[dict[str, float]]:
        return [
            {
                "x_mm": radius * np.cos(2 * np.pi * index / count),
                "y_mm": radius * np.sin(2 * np.pi * index / count),
            }
            for index in range(count)
        ]

    instances = [
        {
            "instance_uuid": str(uuid.UUID(int=index + 1)),
            "catalog": {"catalog_uuid": str(uuid.UUID(int=1000 + index)), "name": f"Part {index}", "obj_id": index + 1},
            # Put the exterior second to prove primary selection is area-based.
            "compensated_contours": [contour(2, 73), contour(10, 97), contour(1, 61)],
        }
        for index in range(200)
    ]
    preview = {
        "schema_version": "pose_template_preview.v1",
        "valid": True,
        "page": {"width_mm": 420, "height_mm": 297},
        "configuration": {
            "page": {
                "size": "A3",
                "orientation": "landscape",
                "origin_from_lower_left_mm": [15, 15],
                "print_compensation_origin": "page_center",
            },
            "print_compensation": {"x_scale": 1.01, "y_scale": 0.99},
        },
        "instances": instances,
    }

    first = build_template_thumbnail(preview)
    second = build_template_thumbnail(preview)

    assert first == second
    assert len(first["instances"]) == 200
    assert all(item["compensated_contours"] for item in first["instances"])
    assert all(item["primary_contour_source_index"] == 1 for item in first["instances"])
    assert first["approximation"]["included_contours"] <= THUMBNAIL_MAX_CONTOURS
    assert first["approximation"]["included_points"] <= THUMBNAIL_MAX_POINTS
    assert first["approximation"]["truncated"] is True
    assert all(item["approximation"]["truncated"] for item in first["instances"])
    assert first["configuration"]["page"]["origin_from_lower_left_mm"] == [15, 15]
    assert first["configuration"]["print_compensation"] == {
        "x_scale": 1.01,
        "y_scale": 0.99,
    }


def test_legacy_bundle_thumbnail_fallback_does_not_mutate_bundle(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    bundle_root = Path(bundle["bundle_path"])
    manifest_path = bundle_root / "pose_template_bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].pop("thumbnail")
    manifest["hashes"].pop("thumbnail")
    manifest["bundle_sha256"] = _hash_json(
        {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    )
    manifest_path.write_text(json.dumps(manifest))
    (bundle_root / "pose_template_thumbnail.json").unlink()

    thumbnail = load_template_thumbnail(bundle["template_uuid"], library_root=library)

    assert thumbnail["schema_version"] == "pose_template_thumbnail.v1"
    assert thumbnail["template_uuid"] == bundle["template_uuid"]
    assert len(thumbnail["instances"]) == 2
    assert not (bundle_root / "pose_template_thumbnail.json").exists()
    assert "thumbnail" not in validate_template_bundle(
        bundle_root, library_root=library
    )["files"]


def test_page_centred_compensation_rejects_geometry_clipped_by_media_box(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    configuration = {
        "display_name": "Compensated edge",
        "page": {"size": "A3", "orientation": "landscape"},
        "print_compensation": {"x_scale": 1.5, "y_scale": 1.0},
        "instances": [
            {
                "instance_uuid": "11111111-1111-4111-8111-111111111111",
                "catalog_uuid": record["catalog_uuid"],
                "pose": {
                    "x_mm": 380,
                    "y_mm": 50,
                    "z_mm": 0,
                    "roll_deg": 0,
                    "pitch_deg": 0,
                    "yaw_deg": 0,
                },
            }
        ],
    }

    preview = build_template_preview(configuration, catalog_root=catalog)

    assert preview["fit"]["objects"][0]["fits"] is True
    assert preview["valid"] is False
    assert preview["errors"][0]["code"] == "compensated_outside_page"
    assert preview["errors"][0]["bounds"]["max_x_mm"] > 420
    with pytest.raises(ValueError, match="page-centred print compensation"):
        generate_template_bundle(
            configuration,
            catalog_root=catalog,
            library_root=tmp_path / "library",
        )


def test_bundle_validation_rejects_symlinked_asset_ancestor(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    bundle_root = Path(bundle["bundle_path"])
    instance_uuid = bundle["instances"][0]["instance_uuid"]
    managed_instance = bundle_root / "assets" / instance_uuid
    outside = tmp_path / "outside-instance"
    shutil.copytree(managed_instance, outside)
    shutil.rmtree(managed_instance)
    managed_instance.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_template_bundle(bundle_root, library_root=library)


def test_catalog_preview_preserves_hollow_geometry_above_legacy_limits(
    tmp_path: Path,
) -> None:
    """Recognition previews must not turn a hollow part into a convex solid."""

    source = trimesh.creation.annulus(
        r_min=4,
        r_max=10,
        height=8,
        sections=48,
    )
    cad = tmp_path / "hollow-ring.stl"
    cad.write_bytes(source.export(file_type="stl"))
    catalog = tmp_path / "catalog"
    record = import_catalog_object(
        name="Hollow ring",
        cad_path=cad,
        catalog_root=catalog,
    )

    first = analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    preview = first["recognition_mesh"]
    rendered = trimesh.Trimesh(
        vertices=preview["vertices"],
        faces=preview["faces"],
        process=False,
    )

    assert len(preview["vertices"]) > 160
    assert len(preview["faces"]) > 256
    assert len(preview["vertices"]) <= adapter.CATALOG_PREVIEW_MAX_VERTICES
    assert len(preview["faces"]) <= adapter.CATALOG_PREVIEW_MAX_FACES
    assert rendered.euler_number == 0
    radii = np.linalg.norm(np.asarray(preview["vertices"])[:, :2], axis=1)
    assert float(radii.min()) == pytest.approx(4, abs=1e-5)
    approximation = first["recognition_mesh_approximation"]
    assert approximation["strategy"] == "welded_source"
    assert approximation["topology_preserved"] is True
    canonical = catalog / record["assets"]["canonical_ply"]["path"]
    thumbnail_bytes = canonical.with_name(ORIENTATION_THUMBNAIL_FILENAME).read_bytes()
    thumbnail = json.loads(thumbnail_bytes)
    assert len(thumbnail_bytes) <= ORIENTATION_THUMBNAIL_MAX_BYTES
    assert thumbnail["preview_mesh"] == preview
    assert thumbnail["recognition_mesh_approximation"] == approximation
    assert "contours" not in thumbnail

    backend = adapter.load_posetemplatecreator_backend()
    second = backend.orientation_artifacts("canonical.ply", canonical.read_bytes())
    assert second["recognition_mesh"] == preview
    assert (
        second["provenance"]["dependencies"]["fast-simplification"] != "unavailable"
    )


def test_catalog_preview_decimation_is_deterministic_and_bounded() -> None:
    source = trimesh.creation.icosphere(subdivisions=5, radius=10)

    first = adapter._preview_mesh_payload(source)
    second = adapter._preview_mesh_payload(source.copy())

    assert first is not None
    assert first == second
    assert len(first["vertices"]) <= adapter.CATALOG_PREVIEW_MAX_VERTICES
    assert len(first["faces"]) <= adapter.CATALOG_PREVIEW_MAX_FACES
    assert len(first["faces"]) < len(source.faces)
    preview_bounds = np.asarray(first["vertices"])
    assert preview_bounds.min(axis=0) == pytest.approx(source.bounds[0], abs=0.01)
    assert preview_bounds.max(axis=0) == pytest.approx(source.bounds[1], abs=0.01)


def test_catalog_preview_spatial_fallback_keeps_dense_hole() -> None:
    source = trimesh.creation.annulus(
        r_min=4,
        r_max=10,
        height=8,
        sections=1_100,
    )

    preview, approximation = adapter._preview_mesh_artifact(source)
    repeated, repeated_approximation = adapter._preview_mesh_artifact(source.copy())

    assert preview is not None
    assert preview == repeated
    assert approximation == repeated_approximation
    assert approximation is not None
    assert approximation["strategy"] == "spatial_clustering"
    assert approximation["topology_preserved"] is True
    assert len(preview["vertices"]) <= adapter.CATALOG_PREVIEW_MAX_VERTICES
    assert len(preview["faces"]) <= adapter.CATALOG_PREVIEW_MAX_FACES
    rendered = trimesh.Trimesh(
        vertices=preview["vertices"],
        faces=preview["faces"],
        process=False,
    )
    assert rendered.euler_number == 0
    radii = np.linalg.norm(np.asarray(preview["vertices"])[:, :2], axis=1)
    assert float(radii.min()) == pytest.approx(4, abs=0.02)


def test_catalog_preview_prefers_better_topology_over_first_bounded_qem() -> None:
    size = 80
    vertices = np.asarray(
        [
            (x, y, 0.05 * np.sin(x * 0.2) * np.sin(y * 0.2))
            for y in range(size + 1)
            for x in range(size + 1)
        ],
        dtype=float,
    )
    faces: list[tuple[int, int, int]] = []
    for y in range(size):
        for x in range(size):
            if x % 5 in (1, 2) and y % 5 in (1, 2):
                continue
            first = y * (size + 1) + x
            faces.extend(
                (
                    (first, first + 1, first + size + 2),
                    (first, first + size + 2, first + size + 1),
                )
            )
    source = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    source.remove_unreferenced_vertices()

    preview, approximation = adapter._preview_mesh_artifact(source)
    first_qem = source.simplify_quadric_decimation(face_count=4_096, aggression=7)

    assert preview is not None
    assert approximation is not None
    assert approximation["strategy"] == "spatial_clustering"
    assert abs(
        approximation["result_euler_number"]
        - approximation["source_euler_number"]
    ) < abs(first_qem.euler_number - source.euler_number)


def test_catalog_preview_relative_quantization_keeps_tiny_valid_mesh() -> None:
    extent = 4e-7
    points = np.asarray(
        [
            (0, 0, 0),
            (extent, 0, 0),
            (0, extent, 0),
            (0, 0, extent),
        ],
        dtype=float,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))

    preview = adapter._bounded_preview_payload(points, faces)

    assert preview is not None
    assert len({tuple(vertex) for vertex in preview["vertices"]}) == 4
    assert len(preview["faces"]) == 4


def test_catalog_preview_failure_uses_explicit_bounded_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = adapter.load_posetemplatecreator_backend()
    payload = trimesh.creation.box(extents=(20, 10, 8)).export(file_type="stl")

    def fail_recognition(_mesh) -> tuple[None, None]:
        raise RuntimeError("injected recognition failure")

    monkeypatch.setattr(adapter, "_preview_mesh_artifact", fail_recognition)
    artifacts = backend.orientation_artifacts("box.stl", payload)

    assert artifacts["recognition_mesh"] == artifacts["preview_mesh"]
    approximation = artifacts["recognition_mesh_approximation"]
    assert approximation["strategy"] == "convex_proxy"
    assert approximation["fallback_reason"] == "RuntimeError"
    assert approximation["result_vertices"] <= adapter.TEMPLATE_PREVIEW_MAX_VERTICES
    assert approximation["result_faces"] <= adapter.TEMPLATE_PREVIEW_MAX_FACES


def test_orientation_cache_rejects_oversized_template_proxy_and_old_adapter(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    analyze_catalog_orientations(record["catalog_uuid"], catalog_root=catalog)
    canonical = catalog / record["assets"]["canonical_ply"]["path"]
    cache = canonical.with_name("pose_template_orientation_analysis.json")
    tampered = json.loads(cache.read_text())
    tampered["preview_mesh"]["vertices"].extend([[0.0, 0.0, 0.0]] * 153)
    cache.write_text(json.dumps(tampered))

    with pytest.raises(ValueError, match="preview_mesh vertices are invalid"):
        load_catalog_orientation_analysis(record["catalog_uuid"], catalog_root=catalog)

    analyze_catalog_orientations(record["catalog_uuid"], catalog_root=catalog)
    old_adapter = json.loads(cache.read_text())
    old_adapter["provenance"]["adapter_version"] = (
        "posetestbot_posetemplatecreator_adapter.v3"
    )
    cache.write_text(json.dumps(old_adapter))
    with pytest.raises(
        OrientationAnalysisStaleError,
        match="unsupported implementation revision",
    ):
        load_catalog_orientation_analysis(record["catalog_uuid"], catalog_root=catalog)

    analyze_catalog_orientations(record["catalog_uuid"], catalog_root=catalog)
    oversized = json.loads(cache.read_text())
    oversized["recognition_mesh"] = {
        "vertices": [
            [0.12345678901234568, 0.12345678901234568, 0.12345678901234568]
            for _index in range(adapter.CATALOG_PREVIEW_MAX_VERTICES)
        ],
        "faces": [[0, 1, 2]],
    }
    oversized["recognition_mesh_approximation"]["result_vertices"] = (
        adapter.CATALOG_PREVIEW_MAX_VERTICES
    )
    oversized["recognition_mesh_approximation"]["result_faces"] = 1
    cache.write_text(json.dumps(oversized))
    with pytest.raises(ValueError, match="exceeds its bounded JSON size"):
        load_catalog_orientation_analysis(record["catalog_uuid"], catalog_root=catalog)


def test_stable_orientation_cache_and_planar_pose_compose_ground_truth(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    analysis = analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    assert analysis["schema_version"] == "pose_template_orientation_analysis.v1"
    assert analysis["source"]["canonical_ply_sha256"] == record["canonical_ply_sha256"]
    assert 1 <= len(analysis["orientations"]) <= 24
    assert len(analysis["preview_mesh"]["vertices"]) <= 160
    assert len(analysis["preview_mesh"]["faces"]) <= 256
    selected = analysis["orientations"][1]
    configuration = {
        "display_name": "Stable box",
        "instances": [
            {
                "instance_uuid": "11111111-1111-4111-8111-111111111111",
                "catalog_uuid": record["catalog_uuid"],
                "orientation_id": selected["orientation_id"],
                "pose": {"x_mm": 40, "y_mm": 35, "rotation_deg": 25},
            }
        ],
    }
    preview = build_template_preview(configuration, catalog_root=catalog)
    assert preview["valid"] is True
    assert preview["configuration"]["instances"][0]["placement_mode"] == "stable_orientation"
    assert preview["instances"][0]["orientation"]["orientation_id"] == selected["orientation_id"]
    assert record["canonical_ply_sha256"] in preview["preview_meshes"]
    planar = matrix_from_xyz_rpy(
        x_mm=40,
        y_mm=35,
        z_mm=0,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=25,
    )
    np.testing.assert_allclose(
        preview["instances"][0]["pose_template_from_object"]["matrix"],
        planar @ np.asarray(selected["source_to_placed"]),
    )
    assert preview["_layout_request"]["objects"][0]["pose"] == {
        "x_mm": 40.0,
        "y_mm": 35.0,
        "rotation_deg": 25.0,
    }
    assert preview["_layout_request"]["objects"][0]["contours"] == selected["contours"]

    cache_path = (catalog / record["assets"]["canonical_ply"]["path"]).with_name(
        "pose_template_orientation_analysis.json"
    )
    stale = json.loads(cache_path.read_text())
    stale["source"]["canonical_ply_sha256"] = "0" * 64
    cache_path.write_text(json.dumps(stale))
    with pytest.raises(OrientationAnalysisStaleError, match="canonical geometry changed"):
        load_catalog_orientation_analysis(record["catalog_uuid"], catalog_root=catalog)


def test_corrected_geometry_orientation_cache_can_be_regenerated(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    original_analysis = analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        {
            "display_name": "Before unit correction",
            "instances": [
                {
                    "instance_uuid": "11111111-1111-4111-8111-111111111111",
                    "catalog_uuid": record["catalog_uuid"],
                    "orientation_id": original_analysis["orientations"][0][
                        "orientation_id"
                    ],
                    "pose": {"x_mm": 40, "y_mm": 40, "rotation_deg": 0},
                }
            ],
        },
        catalog_root=catalog,
        library_root=library,
    )
    set_catalog_object_state(
        record["catalog_uuid"], state="archived", catalog_root=catalog
    )
    corrected = correct_catalog_object_units(
        record["catalog_uuid"],
        conversion="millimeter_to_meter",
        confirm=True,
        operator="pytest",
        expected_geometry_revision=1,
        expected_canonical_sha256=record["canonical_ply_sha256"],
        catalog_root=catalog,
    )

    first = load_catalog_orientation_analysis(
        record["catalog_uuid"], catalog_root=catalog
    )
    second = analyze_catalog_orientations(
        record["catalog_uuid"], catalog_root=catalog
    )

    assert first["source"]["canonical_ply_sha256"] == corrected[
        "canonical_ply_sha256"
    ]
    assert second["source"]["canonical_ply_sha256"] == corrected[
        "canonical_ply_sha256"
    ]
    loaded = load_catalog(catalog)
    assert "orientation_analysis" not in loaded["objects"][0]["assets"]
    assert "orientation_analysis" not in loaded["objects"][0][
        "geometry_revisions"
    ][-1]
    set_catalog_object_state(
        record["catalog_uuid"], state="active", catalog_root=catalog
    )
    with pytest.raises(ValueError, match="different geometry revision"):
        clone_template_configuration(
            bundle["template_uuid"],
            library_root=library,
            catalog_root=catalog,
        )


def test_legacy_upstream_revision_bundle_remains_readable(tmp_path: Path) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    manifest_path = Path(bundle["bundle_path"]) / "pose_template_bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = {
        "name": "PoseTemplateCreator",
        "revision": adapter.LEGACY_POSETEMPLATECREATOR_REVISION,
        "adapter_version": "posetestbot_posetemplatecreator_adapter.v1",
    }
    for instance in manifest["configuration"]["instances"]:
        instance.pop("placement_mode")
    for instance in manifest["instances"]:
        instance.pop("placement_mode")
        instance.pop("orientation")
        instance.pop("preview_mesh_sha256")
    manifest["hashes"]["configuration"] = hashlib.sha256(
        json.dumps(
            manifest["configuration"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    unhashed = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    manifest["bundle_sha256"] = hashlib.sha256(
        json.dumps(
            unhashed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    loaded = validate_template_bundle(bundle["bundle_path"], library_root=library)
    assert loaded["source"]["revision"] == adapter.LEGACY_POSETEMPLATECREATOR_REVISION
    assert "placement_mode" not in loaded["configuration"]["instances"][0]
    cloned = clone_template_configuration(
        bundle["template_uuid"], library_root=library, catalog_root=catalog
    )
    cloned_preview = build_template_preview(cloned, catalog_root=catalog)
    assert cloned_preview["valid"] is True
    assert all(
        item["placement_mode"] == "legacy_arbitrary_pose"
        for item in cloned_preview["configuration"]["instances"]
    )


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
    assert loaded_config["schema_version"] == "run_config.v3"
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


def test_pose_template_selection_preserves_hardware_sync_policy(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    run = tmp_path / "hardware-run"
    run.mkdir()
    synchronization = {
        "schema_version": "capture_synchronization.v1",
        "mode": "hardware_trigger",
        "implementation": "realsense_inter_cam_sync",
        "scope": "depth_exposure",
        "group_id": "template-mixed-rig",
        "master_sensor_key": "realsense_d435:static",
        "max_depth_timestamp_skew_ms": 2.0,
    }
    write_run_config(
        run,
        create_run_config(
            run_root=run,
            dataset_mode="pose_template",
            sensors=(
                SensorRunConfig(
                    "realsense_d435",
                    "static",
                    "Static",
                    mounting_mode="static",
                ),
                SensorRunConfig(
                    "realsense_d435",
                    "hand",
                    "Robot",
                    mounting_mode="eye_in_hand",
                ),
            ),
            synchronization=synchronization,
        ),
    )

    select_pose_template(
        run,
        bundle["template_uuid"],
        placement={"matrix": np.eye(4).tolist()},
        confirmed=True,
        operator="pytest",
        library_root=library,
    )

    loaded_config = load_run_config_for_run_root(run)
    assert loaded_config["schema_version"] == "run_config.v3"
    assert loaded_config["capture"]["synchronization"] == synchronization


def test_pose_template_selection_preserves_legacy_sync_inference_warning(
    tmp_path: Path,
) -> None:
    catalog, record = managed_box(tmp_path)
    library = tmp_path / "library"
    bundle = generate_template_bundle(
        template_configuration(record["catalog_uuid"]),
        catalog_root=catalog,
        library_root=library,
    )
    run = tmp_path / "legacy-run"
    run.mkdir()
    legacy_config = create_run_config(
        run_root=run,
        dataset_mode="pose_template",
    ).to_dict()
    legacy_config["schema_version"] = "run_config.v2"
    legacy_config["capture"].pop("synchronization")
    (run / "run_config.json").write_text(json.dumps(legacy_config))

    select_pose_template(
        run,
        bundle["template_uuid"],
        placement={"matrix": np.eye(4).tolist()},
        confirmed=True,
        operator="pytest",
        library_root=library,
    )

    loaded_config = load_run_config_for_run_root(run)
    assert loaded_config["schema_version"] == "run_config.v3"
    assert loaded_config["capture"]["synchronization"]["mode"] == (
        "timestamp_aligned"
    )
    assert any(
        warning.get("code") == "legacy_capture_synchronization_inferred"
        for warning in loaded_config.get("warnings", [])
    )


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
