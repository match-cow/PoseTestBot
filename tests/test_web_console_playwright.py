from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
import cv2
import pytest
from werkzeug.serving import make_server

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from posetestbot.web.app import create_app


RUN_ROOT = "/tmp/posetestbot-console/new-run"
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class LiveServer:
    def __init__(self):
        self.server = make_server("127.0.0.1", 0, create_app(), threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


@pytest.fixture(scope="module")
def console_server():
    server = LiveServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def page():
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            pytest.fail(
                "Playwright Chromium is not installed; run "
                "`UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium`. "
                f"Original error: {exc}"
            )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            yield page
        finally:
            browser.close()


def fulfill_json(route, value: object, *, status: int = 200) -> None:
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(value),
    )


def run_config(*, plan_only: bool = True) -> dict:
    return {
        "schema_version": "run_config.v1",
        "run_name": "new-run",
        "run_root": RUN_ROOT,
        "robot_profile": {
            "mode": "real",
            "robot_ip": "172.31.1.147",
            "command_port": 30300,
        },
        "capture": {
            "resolution": "720p",
            "fps": 6,
            "velocity_m_s": 0.2,
            "sensors": [],
        },
        "object_folder": "object_models",
        "calibration_profiles": None,
        "pipeline": {
            "sequence_id": "real_full_capture_validation",
            "plan_only": plan_only,
            "options": {},
        },
    }


def overview_payload() -> dict:
    sections = [
        ("run_setup", "Run Setup", "complete"),
        ("preflight", "Preflight", "pending"),
        ("capture", "Capture", "pending"),
        ("sync", "Sync", "pending"),
        ("calibration", "Calibration", "pending"),
        ("bop", "BOP Export", "pending"),
    ]
    return {
        "run_root": RUN_ROOT,
        "config": run_config(),
        "config_error": None,
        "sidebar": [
            {"id": id_, "label": label, "status": status, "artifacts": []}
            for id_, label, status in sections
        ],
        "steps": [],
        "recommendations": [
            {
                "label": "Run preflight",
                "description": "Create fresh readiness evidence before capture.",
            }
        ],
        "recommendation_error": None,
        "artifact_count": 3,
    }


def selected_sensor_status() -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "families": [
            {
                "sensor_type": "realsense_d435",
                "display_name": "Intel RealSense D435",
                "devices": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "wrist-1",
                        "display_name": "RealSense wrist",
                        "effective_display_name": "Wrist RGB-D",
                        "connected": True,
                        "mounting_mode": "eye_in_hand",
                        "inverted": False,
                    },
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "static-1",
                        "display_name": "RealSense static",
                        "effective_display_name": "Static RGB-D",
                        "connected": True,
                        "mounting_mode": "static",
                        "inverted": True,
                    },
                ],
            }
        ],
        "total_connected": 2,
        "all_expected_connected": True,
    }


def install_common_mocks(page, *, preflight_state: dict | None = None, requests: list[dict] | None = None) -> None:
    requests = requests if requests is not None else []
    preflight_state = preflight_state if preflight_state is not None else {"blocker": None}

    page.route("**/ui/bootstrap", lambda route: fulfill_json(route, {
        "schema_version": "web_bootstrap.v1",
        "brand": {"name": "PoseTestBot", "logo_url": "/assets/cow200.png"},
        "robot": {"ip": "172.31.1.147", "port": 30300},
        "default_run_root": "/tmp/posetestbot-console/default",
        "allowed_run_roots": ["/tmp/posetestbot-console"],
    }))
    page.route("**/ui/runs", lambda route: fulfill_json(route, {
        "schema_version": "web_run_index.v1",
        "runs": [
            {"path": RUN_ROOT, "name": "new-run", "sequence": "real_full_capture_validation", "plan_only": True, "config_valid": True, "config_error": None, "modified_at": "2026-07-10T12:00:00Z"},
            {"path": "/tmp/posetestbot-console/old-run", "name": "old-run", "sequence": "sync_aruco", "plan_only": True, "config_valid": True, "config_error": None, "modified_at": "2026-07-09T12:00:00Z"},
        ],
    }))
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview_payload()))
    page.route("**/sensors/status", lambda route: fulfill_json(route, {"schema_version": "sensor_status.v1", "families": [], "total_connected": 0, "all_expected_connected": True}))
    page.route("**/robot/status", lambda route: fulfill_json(route, {"schema_version": "robot_status.v2", "selected_profile": {"mode": "real"}}))
    page.route("**/runtime/status", lambda route: fulfill_json(route, {"schema_version": "runtime_status.v1", "runtimes": [{"runtime_id": "blenderproc", "available": True}]}))
    page.route("**/capture/jobs**", lambda route: fulfill_json(route, {"jobs": [], "active_count": 0, "resources": {}, "status_artifact": None}))
    page.route("**/jobs", lambda route: fulfill_json(route, {"jobs": [], "resources": {}}))
    page.route("**/monitoring/webcam", lambda route: fulfill_json(route, {"job": {"id": "monitor-1", "status": "failed"}, "webrtc_status": {"schema_version": "monitor_webrtc.v1", "transport": "webrtc", "status": "failed", "signaling_ready": False, "peer_count": 0, "frame_count": 0, "selected_node": None, "error": "mock camera offline"}}))
    page.route("**/pipeline/sequences", lambda route: fulfill_json(route, {"sequences": [{"id": "real_full_capture_validation", "label": "Real Full Capture Validation", "description": "Safe plan", "steps": []}]}))
    page.route("**/pipeline/stages", lambda route: fulfill_json(route, {"stages": [{"id": "capture_plan", "label": "Capture Plan", "description": "Write a command plan without hardware.", "resources": ["disk_io"], "parameters": []}]}))

    def config_handler(route) -> None:
        if route.request.method == "POST":
            requests.append({"path": "/run-config", "body": route.request.post_data_json})
            fulfill_json(route, {"config": run_config(), "output": "written"}, status=201)
        else:
            fulfill_json(route, {"config": run_config(), "preflight": {"queue_blocker": preflight_state["blocker"]}})
    page.route("**/run-config**", config_handler)

    def pipeline_handler(route) -> None:
        requests.append({"path": "/pipeline/run", "body": route.request.post_data_json})
        fulfill_json(route, {"job_id": f"job-{len(requests)}", "status": "queued"}, status=202)
    page.route("**/pipeline/run", pipeline_handler)
    page.route("**/sensors/previews/stop", lambda route: (requests.append({"path": "/sensors/previews/stop", "body": {}}), fulfill_json(route, {"jobs": []}))[1])


def test_navigation_run_fallback_persistence_and_both_themes(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script("if (!localStorage.getItem('posetestbot.selectedRun')) localStorage.setItem('posetestbot.selectedRun', '/tmp/posetestbot-console/../outside'); if (!localStorage.getItem('posetestbot.theme')) localStorage.setItem('posetestbot.theme', 'dark')")

    page.goto(console_server.url, wait_until="networkidle")

    expect(page.locator("html")).to_have_class("dark")
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("new-run")
    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    page.get_by_role("link", name="Devices").click()
    expect(page).to_have_url(f"{console_server.url}/#/devices")
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    page.get_by_role("combobox", name="Theme").click()
    page.get_by_role("option", name="Light").click()
    expect(page.locator("html")).to_have_class("light")
    sidebar_rgb = page.locator("aside").evaluate(
        """element => {
            const canvas = document.createElement("canvas")
            canvas.width = 1
            canvas.height = 1
            const context = canvas.getContext("2d")
            context.fillStyle = getComputedStyle(element).backgroundColor
            context.fillRect(0, 0, 1, 1)
            return Array.from(context.getImageData(0, 0, 1, 1).data)
        }"""
    )
    assert min(sidebar_rgb[:3]) > 220
    expect(page.get_by_text("Physical capture always requires fresh operator acknowledgement.", exact=True)).to_have_count(0)
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_css("background-color", "rgba(0, 0, 0, 0)")
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_css("padding", "0px")


def test_run_config_preflight_blocker_and_fresh_capture_gates(console_server, page) -> None:
    requests: list[dict] = []
    preflight_state = {"blocker": "missing_preflight"}
    install_common_mocks(page, preflight_state=preflight_state, requests=requests)
    page.route("**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status()))
    page.add_init_script(
        "localStorage.setItem('posetestbot.selectedSensors', "
        "JSON.stringify(['realsense_d435:wrist-1', 'realsense_d435:static-1']))"
    )
    page.goto(f"{console_server.url}/#/workflow/setup", wait_until="networkidle")

    page.get_by_role("button", name="Write run config").click()
    expect(page.get_by_text("Run configuration written")).to_be_visible()
    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    assert written["plan_only"] is True
    assert "mounting_mode" not in written
    assert [sensor["mounting_mode"] for sensor in written["sensors"]] == [
        "eye_in_hand",
        "static",
    ]
    assert "allow_cameras" not in json.dumps(written)
    assert "allow_real_robot" not in json.dumps(written)

    page.goto(f"{console_server.url}/#/workflow/capture", wait_until="networkidle")
    expect(page.get_by_text("Capture blocked: missing preflight")).to_be_visible()
    page.get_by_role("button", name="Run preflight").click()
    preflight_request = next(item["body"] for item in requests if item["path"] == "/pipeline/run")
    assert preflight_request["stage"] == "run_preflight"
    assert "allow_cameras" not in json.dumps(preflight_request)

    preflight_state["blocker"] = None
    page.reload(wait_until="networkidle")
    page.get_by_role("button", name="Open capture gate").click()
    submit = page.locator('[data-testid="capture-submit"]')
    expect(submit).to_be_disabled()
    page.locator('[data-testid="capture-robot-ack"]').click()
    expect(submit).to_be_disabled()
    page.locator('[data-testid="capture-camera-ack"]').click()
    expect(submit).to_be_enabled()
    submit.click()
    expect(page.get_by_text("Physical capture queued")).to_be_visible()
    capture_request = [item["body"] for item in requests if item["path"] == "/pipeline/run" and item["body"]["stage"] == "capture_execution"][-1]
    assert capture_request["options"]["allow_cameras"] is True
    assert capture_request["options"]["allow_real_robot"] is True
    assert any(item["path"] == "/sensors/previews/stop" for item in requests)


def test_robot_controls_validate_and_confirm_start_and_stop(console_server, page) -> None:
    commands: list[dict] = []
    install_common_mocks(page)

    def command_handler(route) -> None:
        commands.append(route.request.post_data_json)
        fulfill_json(route, {"job_id": f"robot-{len(commands)}", "status": "queued"}, status=202)

    page.route("**/run-command", command_handler)
    page.goto(f"{console_server.url}/#/devices", wait_until="networkidle")

    page.get_by_label("Robot IP").fill("")
    page.get_by_role("button", name="Start IIWA").click()
    expect(page.get_by_text("Enter a valid robot IP and port")).to_be_visible()
    expect(page.get_by_role("dialog")).to_have_count(0)
    assert commands == []

    page.get_by_label("Robot IP").fill("172.31.1.200")
    page.get_by_label("Command port").fill("30301")
    page.get_by_role("button", name="Start IIWA").click()
    expect(page.get_by_role("dialog")).to_contain_text("172.31.1.200:30301")
    expect(page.get_by_role("button", name="Queue start")).to_be_disabled()
    page.get_by_text("I confirm this is the intended lab IIWA target.").click()
    page.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    page.get_by_role("button", name="Stop IIWA").click()
    expect(page.get_by_role("button", name="Queue stop")).to_be_disabled()
    assert [item["command"] for item in commands] == ["start_iiwa"]
    page.get_by_text("I confirm this is the intended lab IIWA target.").click()
    page.get_by_role("button", name="Queue stop").click()
    expect(page.get_by_text("IIWA stop queued")).to_be_visible()

    assert commands == [
        {"command": "start_iiwa", "robot_ip": "172.31.1.200", "robot_port": 30301},
        {"command": "stop_iiwa", "robot_ip": "172.31.1.200", "robot_port": 30301},
    ]


def test_jobs_log_cancel_and_bop_overlay_preview(console_server, page) -> None:
    install_common_mocks(page)
    canceled: list[str] = []
    job = {
        "id": "capture-1", "name": "pipeline:sync_run", "command": ["uv"], "cwd": "/repo", "status": "running", "created_at": "2026-07-10T12:00:00Z", "log_path": "/tmp/log", "started_at": "2026-07-10T12:00:01Z", "ended_at": None, "returncode": None, "message": None, "tail": ["working"], "resources": ["disk_io"], "parameters": {"pipeline_stage": "sync_run"},
    }
    page.route("**/jobs", lambda route: fulfill_json(route, {"jobs": [job], "resources": {"disk_io": "capture-1"}}))
    page.route("**/jobs/capture-1/log", lambda route: route.fulfill(status=200, content_type="text/plain", body="line one\nline two\n"))
    page.route("**/jobs/capture-1/cancel", lambda route: (canceled.append("capture-1"), fulfill_json(route, {"job": {**job, "status": "canceling"}}))[1])
    artifact = {"key": "bop_scene", "source": "bop_export", "path": f"{RUN_ROOT}/bop/test/000001", "relative_path": "bop/test/000001", "kind": "directory", "exists": True, "preview_type": "directory", "size_bytes": None, "modified_at": "2026-07-10T12:00:00Z", "child_count": 8, "summary": {"type": "bop_scene"}}
    page.route("**/artifacts?**", lambda route: fulfill_json(route, {"run_root": RUN_ROOT, "artifacts": [artifact]}))
    page.route("**/artifacts/bop-scene**", lambda route: fulfill_json(route, {"relative_path": "bop/test/000001", "frame_count": 1, "frames": [{"image_id": 0, "gt_count": 1, "rgb": {"exists": True}, "depth": {"exists": True}, "mask_files": ["000000_000000.png"]}]}))
    page.route("**/artifacts/bop-frame**", lambda route: fulfill_json(route, {"type": "bop_frame_detail", "relative_path": "bop/test/000001", "image_id": 0, "gt_count": 1, "rgb": {"exists": True}, "depth": {"exists": True}, "mask_artifacts": [{"name": "000000_000000.png", "relative_path": "mask/000000_000000.png"}], "mask_visib_artifacts": [], "camera": {"cam_K": [1, 0, 0, 0, 1, 0, 0, 0, 1]}, "gt": [{"obj_id": 1}], "gt_info": []}))
    page.route("**/artifacts/bop-frame-overlay**", lambda route: route.fulfill(status=200, content_type="image/png", body=ONE_PIXEL_PNG))

    page.goto(f"{console_server.url}/#/jobs", wait_until="networkidle")
    page.get_by_role("button", name="Log").click()
    expect(page.locator('[data-testid="job-log"]')).to_contain_text("line two")
    page.get_by_role("button", name="Cancel job").click()
    assert canceled == ["capture-1"]

    page.goto(f"{console_server.url}/#/artifacts", wait_until="networkidle")
    page.get_by_text("bop_scene", exact=True).click()
    expect(page.get_by_role("button", name="Frame 0")).to_be_visible()
    expect(page.get_by_alt_text("BOP frame 0 GT and mask overlay")).to_be_visible()
    expect(page.get_by_text("1 masks")).to_be_visible()


def cell_scene_payload(*, objectless: bool = False) -> dict:
    identity = {"semantics": "entity_to_parent", "parent_frame": "template_base", "translation_mm": [0, 0, 0], "rotation_quaternion_wxyz": [1, 0, 0, 0]}
    return {
        "schema_version": "cell_scene.v1",
        "coordinate_system": {"units": "millimetres", "handedness": "right", "up_axis": "+Z", "reference_frame": "template_base", "transform_semantics": "entity_to_parent"},
        "run_root": RUN_ROOT,
        "entities": [
            {"id": "template_base", "type": "reference_frame", "label": "Template base", "status": "planned", "transform": {**identity, "parent_frame": None}, "unresolved_reason": None, "geometry": {"kind": "axes", "size_mm": 100}, "provenance": {"source": "config"}},
            {"id": "robot_flange", "type": "robot_flange", "label": "Robot flange", "status": "recorded", "transform": identity, "unresolved_reason": None, "geometry": {"kind": "flange_proxy"}, "provenance": {"source": "match_robot_ee_poses.json"}},
            {"id": "camera:missing", "type": "camera", "label": "Uncalibrated camera", "status": "unresolved", "transform": None, "unresolved_reason": "No valid calibration profile", "geometry": {"kind": "camera_frustum"}, "provenance": {"source": "calibration_profiles"}},
        ],
        "warnings": [{"code": "missing_calibration_profiles", "message": "No calibration profile collection is available"}],
        "timelines": [{"id": "sensor:realsense_123", "label": "realsense_123", "kind": "synchronized", "frame_count": 2, "default": True, "exact": True, "interpolation": "none", "page_limit": 2000, "source": "match_robot_ee_poses.json"}],
        "default_timeline_id": "sensor:realsense_123",
        "trajectory_preview": [
            {"index": 0, "frame_index": 0, "frame_id": "000000.png", "timestamp_ns": 1, "motion": "arc", "transform": identity},
            {"index": 1, "frame_index": 1, "frame_id": "000001.png", "timestamp_ns": 2, "motion": "arc", "transform": {**identity, "translation_mm": [10, 20, 30]}},
        ],
        "object_selection": {"selected_objects": [] if objectless else ["cube"], "objectless": objectless, "registry": {"valid_count": 1}},
    }


def test_cell_canvas_layers_inspection_and_exact_seeking(console_server, page) -> None:
    install_common_mocks(page)
    scene = cell_scene_payload()
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, scene))
    page.route("**/ui/cell-scene/timeline?**", lambda route: fulfill_json(route, {"schema_version": "cell_timeline.v1", "timeline": scene["timelines"][0], "offset": 0, "limit": 2000, "total": 2, "next_offset": None, "previous_offset": None, "poses": scene["trajectory_preview"]}))

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_test_id("cell-webgl-canvas")).to_be_visible()
    expect(page.get_by_text("Scene has unresolved provenance")).to_be_visible()
    page.get_by_text("Robot flange", exact=True).click()
    expect(page.get_by_text("10.00, 20.00, 30.00")).not_to_be_visible()
    page.get_by_role("slider", name="Frame scrubber").fill("1")
    expect(page.get_by_text("Exact frame 000001.png · arc")).to_be_visible()
    page.get_by_text("Recorded trajectory").click()
    expect(page.get_by_role("checkbox", name="Recorded trajectory")).not_to_be_checked()


def test_cell_webgl_fallback_and_objectless_state(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script("HTMLCanvasElement.prototype.getContext = () => null")
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, cell_scene_payload(objectless=True)))
    page.route("**/ui/cell-scene/timeline?**", lambda route: fulfill_json(route, {"schema_version": "cell_timeline.v1", "timeline": cell_scene_payload()["timelines"][0], "offset": 0, "limit": 2000, "total": 0, "next_offset": None, "previous_offset": None, "poses": []}))

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_test_id("cell-webgl-fallback")).to_be_visible()
    expect(page.get_by_text("WebGL is unavailable")).to_be_visible()
    expect(page.get_by_text("Explicit objectless RGB-D run")).to_be_visible()
    expect(page.get_by_text("Robot flange", exact=True)).to_be_visible()


def test_deterministic_1440_dashboard_screenshot(console_server, page, tmp_path: Path) -> None:
    install_common_mocks(page)
    page.emulate_media(color_scheme="light", reduced_motion="reduce")
    page.add_init_script("localStorage.setItem('posetestbot.theme', 'light')")
    page.goto(console_server.url, wait_until="networkidle")
    path = tmp_path / "dashboard-1440.png"

    page.screenshot(path=path.as_posix(), full_page=True, animations="disabled")

    image = cv2.imread(path.as_posix())
    assert image is not None
    assert image.shape[1] == 1440
    assert image.shape[0] >= 900
