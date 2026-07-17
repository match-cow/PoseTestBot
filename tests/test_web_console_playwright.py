from __future__ import annotations

import json
import threading
from pathlib import Path
import cv2
import numpy as np
import pytest
from werkzeug.serving import make_server

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from posetestbot.web.app import create_app


RUN_ROOT = "/tmp/posetestbot-console/new-run"


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


def install_common_mocks(
    page,
    *,
    preflight_state: dict | None = None,
    requests: list[dict] | None = None,
    generator_available: bool = False,
) -> None:
    requests = requests if requests is not None else []
    preflight_state = preflight_state if preflight_state is not None else {"blocker": None}

    page.route("**/ui/bootstrap", lambda route: fulfill_json(route, {
        "schema_version": "web_bootstrap.v1",
        "brand": {
            "name": "PoseTestBot",
            "logo_url": "/assets/cow_light.png",
            "logo_urls": {
                "light": "/assets/cow_light.png",
                "dark": "/assets/cow_dark.png",
            },
            "favicon_url": "/assets/cow_favicon.png",
        },
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
    page.route("**/calibration-targets/status", lambda route: fulfill_json(route, {
        "schema_version": "calibration_target_generator_status.v1",
        "generation_available": generator_available,
        "generator": {
            "checkout": "/repo/third_party/PoseGridGen",
            "required_revision": "ad152e369e8d2746d0cf66cb1455f2371b0ec0f0",
            "reason": None if generator_available else "Pinned source checkout is unavailable",
        },
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
    page.emulate_media(color_scheme="dark")
    page.add_init_script("if (!localStorage.getItem('posetestbot.selectedRun')) localStorage.setItem('posetestbot.selectedRun', '/tmp/posetestbot-console/../outside'); localStorage.removeItem('posetestbot.theme')")

    page.goto(console_server.url, wait_until="networkidle")

    expect(page.locator("html")).to_have_class("dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_attribute(
        "src", "/assets/cow_dark.png"
    )
    assert page.evaluate("localStorage.getItem('posetestbot.theme')") is None
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("new-run")
    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    page.get_by_role("link", name="Devices").click()
    expect(page).to_have_url(f"{console_server.url}/#/devices")
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    theme_toggle = page.get_by_role("button", name="Switch to light theme")
    theme_toggle_box = theme_toggle.bounding_box()
    assert theme_toggle_box is not None
    assert theme_toggle_box["width"] == pytest.approx(34)
    assert theme_toggle_box["height"] == pytest.approx(34)
    theme_toggle.click()
    expect(page.locator("html")).to_have_class("light")
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_attribute(
        "src", "/assets/cow_light.png"
    )
    assert page.evaluate("localStorage.getItem('posetestbot.theme')") == "light"
    expect(page.get_by_role("link", name="Open PoseTestBot on GitHub")).to_have_attribute(
        "href", "https://github.com/match-cow/PoseTestBot"
    )
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


def test_dashboard_quick_robot_controls_use_configured_target(console_server, page) -> None:
    commands: list[dict] = []
    install_common_mocks(page)

    def command_handler(route) -> None:
        commands.append(route.request.post_data_json)
        fulfill_json(
            route,
            {"job_id": f"dashboard-robot-{len(commands)}", "status": "queued"},
            status=202,
        )

    page.route("**/run-command", command_handler)
    page.goto(f"{console_server.url}/#/dashboard", wait_until="networkidle")

    controls = page.get_by_test_id("iiwa-quick-controls")
    expect(controls.get_by_role("button", name="Start IIWA")).to_be_visible()
    expect(controls.get_by_role("button", name="Stop IIWA")).to_be_visible()
    expect(page.get_by_label("Robot IP")).to_have_count(0)
    expect(page.get_by_label("Command port")).to_have_count(0)

    controls.get_by_role("button", name="Start IIWA").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("172.31.1.147:30300")
    expect(dialog.get_by_role("button", name="Queue start")).to_be_disabled()
    dialog.get_by_text("I confirm this is the intended lab IIWA target.").click()
    dialog.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    controls.get_by_role("button", name="Stop IIWA").click()
    expect(dialog.get_by_role("button", name="Queue stop")).to_be_disabled()
    dialog.get_by_text("I confirm this is the intended lab IIWA target.").click()
    dialog.get_by_role("button", name="Queue stop").click()
    expect(page.get_by_text("IIWA stop queued")).to_be_visible()

    assert commands == [
        {"command": "start_iiwa", "robot_ip": "172.31.1.147", "robot_port": 30300},
        {"command": "stop_iiwa", "robot_ip": "172.31.1.147", "robot_port": 30300},
    ]


def test_jobs_log_cancel_and_removed_artifacts_route(console_server, page) -> None:
    install_common_mocks(page)
    canceled: list[str] = []
    job = {
        "id": "capture-1", "name": "pipeline:sync_run", "command": ["uv"], "cwd": "/repo", "status": "running", "created_at": "2026-07-10T12:00:00Z", "log_path": "/tmp/log", "started_at": "2026-07-10T12:00:01Z", "ended_at": None, "returncode": None, "message": None, "tail": ["working"], "resources": ["disk_io"], "parameters": {"pipeline_stage": "sync_run"},
    }
    page.route("**/jobs", lambda route: fulfill_json(route, {"jobs": [job], "resources": {"disk_io": "capture-1"}}))
    page.route("**/jobs/capture-1/log", lambda route: route.fulfill(status=200, content_type="text/plain", body="line one\nline two\n"))
    page.route("**/jobs/capture-1/cancel", lambda route: (canceled.append("capture-1"), fulfill_json(route, {"job": {**job, "status": "canceling"}}))[1])
    page.goto(f"{console_server.url}/#/jobs", wait_until="networkidle")
    page.get_by_role("button", name="Log").click()
    expect(page.locator('[data-testid="job-log"]')).to_contain_text("line two")
    page.get_by_role("button", name="Cancel job").click()
    assert canceled == ["capture-1"]

    expect(page.get_by_role("link", name="Artifacts")).to_have_count(0)
    page.goto(f"{console_server.url}/#/artifacts", wait_until="networkidle")
    expect(page).to_have_url(f"{console_server.url}/#/dashboard")


def test_calibration_target_unavailable_keeps_saved_library_navigation(
    console_server, page
) -> None:
    install_common_mocks(page, generator_available=False)
    page.route("**/calibration-targets/bundles?**", lambda route: fulfill_json(route, {
        "schema_version": "calibration_target_library.v1",
        "run_root": RUN_ROOT,
        "bundles": [],
    }))
    page.goto(console_server.url, wait_until="networkidle")

    expect(page.get_by_role("link", name="Calibration Targets")).to_be_visible()
    page.goto(f"{console_server.url}/#/calibration-targets", wait_until="networkidle")
    expect(page.get_by_text("Target generation is unavailable")).to_be_visible()
    expect(page.get_by_text("Saved target library")).to_be_visible()
    expect(page.get_by_text("git submodule update --init third_party/PoseGridGen")).to_be_visible()


def test_two_mode_calibration_workflow_progress_results_overrides_and_saved_state(
    console_server, page
) -> None:
    requests: list[dict] = []
    promoted = {"value": False}
    install_common_mocks(page)
    setup = {
        "schema_version": "calibration_setup.v1",
        "run_root": RUN_ROOT,
        "cameras": [
            {"sensor_key": "realsense_d435:wrist-1", "sensor_name": "realsense_wrist-1", "display_name": "Wrist RGB-D", "sensor_type": "realsense_d435", "device_id": "wrist-1"},
            {"sensor_key": "oak_d_pro:static-1", "sensor_name": "luxonis_static-1", "display_name": "Static OAK-D", "sensor_type": "oak_d_pro", "device_id": "static-1"},
        ],
        "unavailable_cameras": [],
        "saved_targets": [
            {"target_id": "5f09f41c-dd91-44ef-a048-1f43fc990e17", "display_name": "Lab board", "valid": True},
            {"target_id": "9ab5ff1c-60f6-46b1-823d-2a912d5d4e3f", "display_name": "Alternate board", "valid": True},
        ],
        "modes": [
            {"id": "eye_in_hand", "label": "Robot-mounted camera (eye-in-hand)", "primary_transform": "camera → robot_flange", "target_mounting": "stationary relative to template_base"},
            {"id": "eye_to_hand", "label": "Static camera (eye-to-hand)", "primary_transform": "camera → template_base", "target_mounting": "rigidly attached to robot_flange"},
        ],
        "solver": {"default_pnp_methods": ["IPPE", "ITERATIVE", "SQPNP"], "default_extrinsic_methods": ["tsai", "park", "horaud", "andreff", "daniilidis", "shah", "li"]},
        "latest_attempt": None,
    }
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))

    def create_handler(route) -> None:
        requests.append({"path": "/calibration/attempts", "body": route.request.post_data_json})
        fulfill_json(route, {"attempt_id": "a" * 32, "job_id": "calculation-1", "status": "queued"}, status=202)

    page.route("**/calibration/attempts", create_handler)
    transform = {
        "from": "camera",
        "to": "robot_flange",
        "matrix": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
        "rotation_quaternion_wxyz": [1, 0, 0, 0],
        "translation_mm": [10, 20, 30],
    }
    recommended = {
        "candidate_id": "realsense_d435:wrist-1|IPPE|park",
        "profile_id": "wrist_ippe_park",
        "pnp_method": "IPPE",
        "extrinsic_method": "park",
        "algorithms": ["IPPE", "park"],
        "status": "passing",
        "validation_state": "passed",
        "recommended": True,
        "score": 0.12,
        "observation_count": 10,
        "inlier_count": 9,
        "outlier_count": 1,
        "outlier_ratio": 0.1,
        "mean_reprojection_error_px": 0.25,
        "primary_transform": transform,
        "companion_transform": {**transform, "from": "aruco_grid", "to": "template_base"},
        "held_out_residuals": {"mean_translation_mm": 0.8, "median_translation_mm": 0.7, "mean_rotation_deg": 0.3, "median_rotation_deg": 0.2},
    }
    override = {**recommended, "candidate_id": "realsense_d435:wrist-1|SQPNP|tsai", "profile_id": "wrist_sqpnp_tsai", "pnp_method": "SQPNP", "extrinsic_method": "tsai", "recommended": False, "score": 0.2}
    failed = {
        "candidate_id": "oak_d_pro:static-1|ITERATIVE|li", "pnp_method": "ITERATIVE", "extrinsic_method": "li", "algorithms": ["ITERATIVE", "li"], "status": "error", "validation_state": "failed", "score": None, "observation_count": 3, "inlier_count": 0, "outlier_count": 3, "outlier_ratio": 1, "error": "leave-one-pose-out validation requires at least four poses",
    }

    def attempt_payload() -> dict:
        return {
            "schema_version": "calibration_attempt.v1",
            "attempt_id": "a" * 32,
            "request": {"mode": "eye_in_hand", "sensor_keys": ["realsense_d435:wrist-1", "oak_d_pro:static-1"], "target_id": setup["saved_targets"][0]["target_id"], "solver_policy": "auto_compare"},
            "progress": {"status": "complete", "message": "Calibration calculations are complete and awaiting review.", "phases": [
                {"id": "prepare_data", "label": "Prepare data", "status": "complete"},
                {"id": "estimate_target_poses", "label": "Estimate target poses", "status": "complete"},
                {"id": "compare_robot_camera_solutions", "label": "Compare robot-camera solutions", "status": "complete"},
                {"id": "validate_and_rank", "label": "Validate and rank", "status": "complete"},
            ]},
            "results": {"status": "partial", "recommended_camera_count": 1, "failed_camera_count": 1, "results": [
                {**setup["cameras"][0], "status": "passing", "recommended_candidate_id": recommended["candidate_id"], "recommendation": recommended, "candidates": [recommended, override]},
                {**setup["cameras"][1], "status": "failed", "recommended_candidate_id": None, "recommendation": None, "candidates": [failed]},
            ]},
            "promotion": ({"status": "promoted", "promoted_profile_ids": ["wrist_sqpnp_tsai"]} if promoted["value"] else None),
        }

    page.route("**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?**", lambda route: fulfill_json(route, attempt_payload()))

    def promote_handler(route) -> None:
        requests.append({"path": "/calibration/promote", "body": route.request.post_data_json})
        promoted["value"] = True
        fulfill_json(route, {"attempt_id": "a" * 32, "job_id": "promotion-1", "status": "queued", "selections": route.request.post_data_json["candidate_ids"]}, status=202)

    page.route("**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/promote", promote_handler)
    page.goto(f"{console_server.url}/#/workflow/calibration", wait_until="networkidle")

    expect(page.get_by_test_id("calibration-workflow")).to_be_visible()
    expect(page.locator('input[name="calibration-mode"]')).to_have_count(2)
    expect(page.get_by_text("Robot-mounted camera (eye-in-hand)")).to_be_visible()
    expect(page.get_by_text("Static camera (eye-to-hand)")).to_be_visible()
    expect(page.locator("[data-stage-id]")).to_have_count(0)
    expect(page.get_by_text("Auto compare — recommended")).to_be_visible()
    page.locator('input[value="eye_to_hand"]').check()
    expect(page.locator('input[value="eye_to_hand"]')).to_be_checked()
    page.locator('input[value="eye_in_hand"]').check()
    camera_choices = page.get_by_role("checkbox")
    camera_choices.nth(1).click()
    expect(camera_choices.nth(1)).not_to_be_checked()
    camera_choices.nth(1).click()
    page.get_by_role("combobox", name="Saved calibration target").click()
    page.get_by_role("option", name="Alternate board").click()
    expect(page.get_by_role("button", name="Run calibration")).to_be_enabled()
    page.get_by_role("button", name="Run calibration").click()
    expect(page.get_by_text("Calibration queued")).to_be_visible()
    assert requests[0]["body"]["mode"] == "eye_in_hand"
    assert requests[0]["body"]["sensor_keys"] == ["realsense_d435:wrist-1", "oak_d_pro:static-1"]
    assert requests[0]["body"]["target_id"] == "9ab5ff1c-60f6-46b1-823d-2a912d5d4e3f"

    expect(page.get_by_text("Prepare data")).to_be_visible()
    expect(page.get_by_test_id("calibration-results")).to_be_visible()
    expect(page.get_by_text("camera → robot_flange").last).to_be_visible()
    expect(page.get_by_text("Every attempted candidate and failure").first).to_be_visible()
    page.get_by_label("Candidate override").click()
    page.get_by_role("option", name="SQPNP + tsai · score 0.2000").click()
    page.get_by_role("button", name="Accept recommendations").click()
    expect(page.get_by_text("Calibration acceptance queued")).to_be_visible()
    assert requests[-1]["body"]["candidate_ids"] == {"realsense_d435:wrist-1": override["candidate_id"]}
    expect(page.get_by_role("button", name="Recommendations saved")).to_be_visible()
    expect(page.get_by_text("Saved 1 camera profile(s).")).to_be_visible()


def test_calibration_target_preview_fit_generate_download_select_and_run_switch(
    console_server, page
) -> None:
    requests: list[dict] = []
    library_urls: list[str] = []
    blocked = {"value": True}
    deleted = {"value": False}
    selected_runs: set[str] = set()
    target_id = "5f09f41c-dd91-44ef-a048-1f43fc990e17"
    old_run = "/tmp/posetestbot-console/old-run"
    configuration = {
        "schema_version": "2.0",
        "page": {"paper_size": "A4", "orientation": "landscape"},
        "board": {
            "type": "aruco",
            "dictionary": "DICT_5X5_50",
            "rows": 2,
            "columns": 3,
            "marker_size_mm": 30,
            "separation_mm": 10,
            "show_ids": False,
            "id_font_size_pt": 8,
        },
        "print_compensation": {"x_percent": 101, "y_percent": 99},
        "annotations": {
            "show_ruler": True,
            "show_parameters": True,
            "show_frame_legend": False,
        },
        "coordinate_frame": {
            "enabled": True,
            "pose": {
                "translation_x_m": 0.1,
                "translation_y_m": -0.2,
                "translation_z_m": 0.3,
                "roll_deg": 10,
                "pitch_deg": 20,
                "yaw_deg": 30,
            },
        },
    }
    install_common_mocks(page, requests=requests, generator_available=True)
    page.route("**/calibration-targets/capabilities", lambda route: fulfill_json(route, {
        "schema_version": "posegridgen_capabilities.v1",
        "paper_sizes_mm": {"A4": [210, 297], "A3": [297, 420]},
        "dictionaries": {"DICT_5X5_50": 50},
        "defaults": configuration,
    }))

    def bundle_payload(run_root: str) -> dict:
        selected = run_root in selected_runs
        return {
            "schema_version": "calibration_target_library.v1",
            "run_root": run_root,
            "bundles": [] if deleted["value"] else [{
                "target_id": target_id,
                "display_name": "Anisotropic calibration board",
                "created_at": "2026-07-16T12:00:00Z",
                "valid": True,
                "selected": selected,
                "selected_placement": (
                    {"mode": "posegridgen_board_to_base"} if selected else None
                ),
                "geometry_sha256": "a" * 64,
                "target": {
                    "target_bounds": {"width_mm": 111.1, "height_mm": 69.3},
                    "print_compensation": {"x_percent": 101, "y_percent": 99},
                    "grid_size": [3, 2],
                },
            }],
        }

    def library_handler(route) -> None:
        library_urls.append(route.request.url)
        run_root = route.request.url.split("run_root=", 1)[-1]
        run_root = run_root.replace("%2F", "/")
        fulfill_json(route, bundle_payload(run_root))

    page.route("**/calibration-targets/bundles?**", library_handler)
    png = cv2.imencode(".png", np.full((12, 16, 3), 220, dtype=np.uint8))[1].tobytes()

    def preview_handler(route) -> None:
        requests.append({"path": "/calibration-targets/preview", "body": route.request.post_data_json})
        route.fulfill(status=200, content_type="image/png", body=png)

    page.route("**/calibration-targets/preview", preview_handler)

    def fit_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/calibration-targets/fit", "body": body})
        fulfill_json(route, {"request": body, "adjusted": False, "scale_factor": 1, "changes": []})

    page.route("**/calibration-targets/fit", fit_handler)

    def generate_handler(route) -> None:
        requests.append({"path": "/calibration-targets/generate", "body": route.request.post_data_json})
        fulfill_json(route, {"job_id": "generate-1", "job": {"id": "generate-1", "status": "queued"}}, status=202)

    page.route("**/calibration-targets/generate", generate_handler)
    page.route("**/jobs/generate-1", lambda route: fulfill_json(route, {"job": {"id": "generate-1", "status": "succeeded", "message": None, "tail": []}}))
    page.route("**/jobs/select-1", lambda route: fulfill_json(route, {"job": {"id": "select-1", "status": "succeeded", "message": None, "tail": []}}))

    def select_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/calibration-targets/select", "body": body})
        if blocked["value"]:
            fulfill_json(
                route,
                {
                    "output": "The active calibration target cannot be replaced; create a new run.",
                    "blockers": ["calibration_observations.json"],
                },
                status=409,
            )
            return
        selected_runs.add(body["run_root"])
        fulfill_json(route, {"job_id": "select-1", "job": {"id": "select-1", "status": "queued"}}, status=202)

    page.route(f"**/calibration-targets/bundles/{target_id}/select", select_handler)

    def delete_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/calibration-targets/delete", "body": body})
        deleted["value"] = True
        fulfill_json(route, {
            "status": "deleted",
            "target_id": target_id,
            "display_name": "Anisotropic calibration board",
        })

    page.route(f"**/calibration-targets/bundles/{target_id}", delete_handler)
    page.route(
        f"**/calibration-targets/bundles/{target_id}/download/pdf",
        lambda route: route.fulfill(
            status=200,
            content_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=target-artifact.bin"},
            body=b"target artifact",
        ),
    )

    page.goto(f"{console_server.url}/#/calibration-targets", wait_until="networkidle")
    expect(page.get_by_role("link", name="Calibration Targets")).to_be_visible()
    expect(page.get_by_role("img", name="Calibration target preview")).to_be_visible()
    expect(page.get_by_text("297 × 210 mm", exact=True)).to_be_visible()
    preview_page_ratio = page.get_by_test_id("calibration-preview-page").evaluate(
        "element => { const box = element.getBoundingClientRect(); return box.width / box.height }"
    )
    assert preview_page_ratio == pytest.approx(297 / 210, abs=0.002)
    assert any(item["path"] == "/calibration-targets/preview" for item in requests)

    page.get_by_role("button", name="Fit to page").click()
    expect(page.get_by_text("Board fitted to the selected page")).to_be_visible()
    page.get_by_label("Target display name").fill("Printed target 01")
    page.get_by_role("button", name="Generate bundle").click()
    expect(page.get_by_text("Calibration target generated")).to_be_visible()
    generated = next(item["body"] for item in requests if item["path"] == "/calibration-targets/generate")
    assert generated["display_name"] == "Printed target 01"
    assert generated["configuration"]["print_compensation"] == {"x_percent": 101, "y_percent": 99}
    expect(page.get_by_text(f"Active for {RUN_ROOT}")).to_have_count(0)

    pdf_link = page.get_by_role("link", name="PDF")
    expect(pdf_link).to_have_attribute(
        "href", f"/calibration-targets/bundles/{target_id}/download/pdf"
    )
    expect(pdf_link).to_have_attribute("download", "")

    page.get_by_role("button", name="Select for run").click()
    page.get_by_role("combobox", name="Target placement").click()
    page.get_by_role("option", name="Use PoseGridGen board pose").click()
    page.get_by_role("button", name="Select target").click()
    expect(page.get_by_text("Target was not selected")).to_be_visible()
    expect(page.get_by_text("The active calibration target cannot be replaced; create a new run.")).to_be_visible()
    expect(page.get_by_text("calibration_observations.json", exact=False)).to_be_visible()

    blocked["value"] = False
    page.get_by_role("button", name="Select target").click()
    expect(page.get_by_text("Calibration target selected")).to_be_visible()
    selection = [item["body"] for item in requests if item["path"] == "/calibration-targets/select"][-1]
    assert selection == {"run_root": RUN_ROOT, "placement": "posegridgen_board_to_base"}
    expect(page.get_by_text(f"Active for {RUN_ROOT}")).to_be_visible()
    expect(page.get_by_role("button", name="Delete Anisotropic calibration board")).to_be_disabled()

    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_text(f"Active for {old_run}")).to_have_count(0)
    expect(page.get_by_role("button", name="Select for run")).to_be_visible()
    assert any("old-run" in url for url in library_urls)

    page.get_by_role("button", name="Delete Anisotropic calibration board").click()
    expect(page.get_by_role("heading", name="Delete Anisotropic calibration board?")).to_be_visible()
    assert not any(item["path"] == "/calibration-targets/delete" for item in requests)
    page.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Calibration target deleted")).to_be_visible()
    expect(page.get_by_role("heading", name="Anisotropic calibration board")).to_have_count(0)
    deletion = next(item["body"] for item in requests if item["path"] == "/calibration-targets/delete")
    assert deletion == {"run_root": old_run, "confirm": True}


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
