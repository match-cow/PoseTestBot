from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from posetestbot.io.artifacts import BOP_DIR, BOP_EXPORT_MANIFEST, BOP_TARGETS_BOP19, DEPTH_DIR, RGB_DIR
from posetestbot.pipeline.run_config import create_run_config, write_run_config
from posetestbot.web.routes import sensors as web_sensors


def load_web_interface_app():
    module_path = Path(__file__).resolve().parents[1] / "web_interface.py"
    spec = importlib.util.spec_from_file_location("web_interface", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


app = load_web_interface_app()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(path.as_posix(), image)


def make_bop_scene(run_root: Path) -> Path:
    scene = run_root / BOP_DIR / "realsense_123" / "test" / "000001"
    write_png(scene / RGB_DIR / "000000.png")
    write_png(scene / DEPTH_DIR / "000000.png")
    write_json(scene / "scene_camera.json", {"0": {"cam_K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}})
    write_json(scene / "scene_gt.json", {"0": []})
    write_json(
        run_root / BOP_DIR / BOP_EXPORT_MANIFEST,
        {
            "schema_version": "bop_export_manifest.v1",
            "exports": [
                {
                    "sensor_name": "realsense_123",
                    "scene_id": 1,
                    "split": "test",
                    "scene_folder": scene.relative_to(run_root).as_posix(),
                }
            ],
            "object_models": [{"object_name": "cube", "obj_id": 1}],
        },
    )
    write_json(run_root / BOP_DIR / BOP_TARGETS_BOP19, [{"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 1}])
    return scene


def test_index_lists_acquisition_sequences_only() -> None:
    client = app.test_client()

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "fake_capture_to_bop_dataset_dry_run" in html
    assert "capture_to_bop_dataset_dry_run" in html
    assert "Inverted RealSense" in html
    assert "foundationpose_runtime_to_bop_eval" not in html
    assert "megapose_to_bop_eval_dry_run" not in html


def test_pipeline_stage_and_sequence_endpoints_hide_downstream_ids() -> None:
    client = app.test_client()

    stages = client.get("/pipeline/stages").get_json()["stages"]
    sequences = client.get("/pipeline/sequences").get_json()["sequences"]

    stage_ids = {stage["id"] for stage in stages}
    sequence_ids = {sequence["id"] for sequence in sequences}
    assert "bop_export" in stage_ids
    assert "bop_evaluation" not in stage_ids
    assert "metric_report_export" not in stage_ids
    assert "fake_capture_to_bop_dataset_dry_run" in sequence_ids
    assert "foundationpose_to_bop_eval_dry_run" not in sequence_ids


def test_removed_metric_and_bop_result_endpoints_are_not_registered(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run"
    run_root.mkdir()

    assert client.get(f"/artifacts/metrics?run_root={run_root.as_posix()}").status_code == 404
    assert client.get(f"/artifacts/bop-result?run_root={run_root.as_posix()}&path=x.csv").status_code == 404


def test_bop_frame_endpoint_reports_dataset_frame_without_results(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run"
    scene = make_bop_scene(run_root)

    response = client.get(
        "/artifacts/bop-frame",
        query_string={
            "run_root": run_root.as_posix(),
            "path": scene.relative_to(run_root).as_posix(),
            "image_id": "0",
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["type"] == "bop_frame_detail"
    assert payload["scene"]["scene_id"] == 1
    assert payload["result"] is None


def test_pipeline_recommendations_endpoint_is_acquisition_only(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run"
    write_run_config(run_root, create_run_config(run_root=run_root))

    response = client.get(
        "/pipeline/recommendations",
        query_string={"run_root": run_root.as_posix()},
    )

    payload = response.get_json()
    ids = {item["id"] for item in payload["recommendations"]}
    assert response.status_code == 200
    assert "write_run_preflight" in ids
    assert "evaluate_bop_results" not in ids
    assert "plan_foundationpose" not in ids


def test_run_config_endpoint_round_trips_realsense_inverted(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-inverted"

    response = client.post(
        "/run-config",
        json={
            "run_root": run_root.as_posix(),
            "robot_mode": "fake",
            "sequence": "sync_aruco",
            "resolution": "720p",
            "fps": 6,
            "velocity": 0.2,
            "object_folder": "object_models",
            "sensors": [
                {
                    "sensor_type": "realsense",
                    "device_id": "123",
                    "mounting_mode": "static",
                    "display_name": "Cell RealSense",
                    "inverted": True,
                },
                {
                    "sensor_type": "oak",
                    "device_id": "auto",
                    "mounting_mode": "static",
                    "display_name": "Cell OAK-D Pro",
                },
            ],
        },
    )

    payload = response.get_json()
    assert response.status_code == 201
    assert payload["config"]["capture"]["sensors"][0]["inverted"] is True
    assert payload["config"]["capture"]["sensors"][1]["inverted"] is False

    loaded = client.get(
        "/run-config",
        query_string={"run_root": run_root.as_posix()},
    ).get_json()

    assert loaded["config"]["capture"]["sensors"][0]["inverted"] is True
    assert loaded["config"]["capture"]["sensors"][0]["sensor_type"] == "realsense_d435"


def test_sensor_alias_endpoint_round_trips_lab_local_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alias_path = tmp_path / "sensor_aliases.json"
    monkeypatch.setattr(web_sensors, "DEFAULT_SENSOR_ALIASES_PATH", alias_path)
    client = app.test_client()

    response = client.put(
        "/sensors/aliases",
        json={
            "aliases": {
                "realsense_d435:123": {
                    "alias": "Wrist Camera",
                    "mounting_mode": "eye_in_hand",
                    "inverted": True,
                }
            }
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["aliases"]["realsense_d435:123"]["alias"] == "Wrist Camera"

    loaded = client.get("/sensors/aliases").get_json()
    assert loaded["aliases"]["realsense_d435:123"]["inverted"] is True


def test_overview_endpoint_reports_sequence_steps(tmp_path: Path) -> None:
    client = app.test_client()
    run_root = tmp_path / "run-overview"
    write_run_config(
        run_root,
        create_run_config(run_root=run_root, sequence_id="sync_to_bop_dry_run"),
    )

    response = client.get(
        "/ui/overview",
        query_string={"run_root": run_root.as_posix()},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["config"]["pipeline"]["sequence_id"] == "sync_to_bop_dry_run"
    assert any(step["stage_id"] == "sync_quality" for step in payload["steps"])
    assert any(section["id"] == "sensors" for section in payload["sidebar"])


def test_sensor_snapshot_submission_queues_camera_job(monkeypatch, tmp_path: Path) -> None:
    class FakeJob:
        id = "job123"
        status = "queued"
        parameters = {"snapshot_root": (tmp_path / "snapshots").as_posix()}

        def to_dict(self):
            return {
                "id": self.id,
                "status": self.status,
                "parameters": self.parameters,
            }

    class FakeRunner:
        def __init__(self):
            self.submitted = None

        def submit(self, **kwargs):
            self.submitted = kwargs
            return FakeJob()

    fake_runner = FakeRunner()
    monkeypatch.setattr(web_sensors, "job_runner", fake_runner)
    monkeypatch.setattr(
        web_sensors,
        "snapshot_batch_root",
        lambda: tmp_path / "snapshots",
    )
    monkeypatch.setattr(
        web_sensors,
        "collect_sensor_status",
        lambda: {
            "schema_version": "sensor_status.v1",
            "families": [
                {
                    "devices": [
                        {
                            "sensor_type": "realsense_d435",
                            "device_id": "123",
                            "display_name": "RealSense 123",
                            "effective_display_name": "Wrist Camera",
                            "connected": True,
                            "metadata": {},
                        }
                    ]
                }
            ],
            "total_connected": 1,
            "all_expected_connected": True,
            "expected_counts_requested": False,
        },
    )
    client = app.test_client()

    response = client.post(
        "/sensors/snapshots",
        json={"selected": ["realsense_d435:123"]},
    )

    payload = response.get_json()
    assert response.status_code == 202
    assert payload["job_id"] == "job123"
    assert fake_runner.submitted["resources"] == ["camera"]
    assert fake_runner.submitted["command"][:4] == [
        "uv",
        "run",
        "python",
        "scripts/capture_sensor_snapshot.py",
    ]


def test_snapshot_worker_reports_inaccessible_realsense_video_nodes(
    tmp_path: Path,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_sensor_snapshot.py"
    spec = importlib.util.spec_from_file_location(
        "posetestbot_snapshot_worker_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = module._capture_one(
        snapshot_root=tmp_path / "snapshots",
        spec={
            "sensor_type": "realsense_d435",
            "device_id": "123",
            "metadata": {
                "video_nodes": [{"path": "/dev/video0", "accessible": False}],
                "video_accessible": False,
            },
        },
        fps=6,
        resolution="720p",
        max_frames=1,
    )

    assert record["status"] == "failed"
    assert "not accessible" in record["error"]
    assert record["diagnostics"][0]["code"] == "video_permission_denied"
