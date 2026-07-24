from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.parse import urlparse
import cv2
import numpy as np
import pytest
import trimesh
from werkzeug.serving import make_server

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from posetestbot.web.app import create_app
from posetestbot.pose_templates.catalog import import_catalog_object
from posetestbot.web.routes import pose_templates as pose_template_routes


pytestmark = pytest.mark.playwright

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


def run_config(*, plan_only: bool = True, sensors: list[dict] | None = None) -> dict:
    return {
        "schema_version": "run_config.v2",
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
            "sensors": sensors or [],
        },
        "dataset_mode": "objectless",
        "pose_template": None,
        "calibration_profiles": None,
        "pipeline": {
            "sequence_id": "real_full_capture_validation",
            "plan_only": plan_only,
            "options": {},
        },
    }


def calibration_selection_artifact(
    *,
    bundle_sha256: str,
    calibration_profiles: str,
    intrinsic_calibration_profiles: str,
    source_run_root: str = "/tmp/posetestbot-console/calibration-source",
    source_run_name: str = "Reusable calibration",
) -> dict:
    return {
        "schema_version": "calibration_profile_selection.v1",
        "selected_at": "2026-07-22T12:00:00+00:00",
        "operator": "web_operator",
        "source": {
            "run_root": source_run_root,
            "run_name": source_run_name,
            "bundle_sha256": bundle_sha256,
        },
        "snapshot": {
            "calibration_profiles": {
                "relative_path": calibration_profiles,
                "sha256": "b" * 64,
            },
            "intrinsic_calibration_profiles": {
                "relative_path": intrinsic_calibration_profiles,
                "sha256": "c" * 64,
            },
        },
        "sensor_profiles": {
            "realsense_d435:wrist-1": "profile-wrist-1",
        },
    }


def valid_library_selection(**kwargs) -> dict:
    return {
        **calibration_selection_artifact(**kwargs),
        "valid": True,
        "issues": [],
    }


def overview_payload(config: dict | None = None) -> dict:
    resolved_config = config or run_config()
    selected_bundle = (
        resolved_config.get("calibration_profile_selection") or {}
    ).get("bundle_sha256", "a" * 64)
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
        "config": resolved_config,
        "config_error": None,
        "calibration_sync": {
            "status": "ready",
            "bundle_sha256": selected_bundle,
            "sensors": [
                {
                    "sensor_key": "realsense_d435:wrist-1",
                    "sensor_name": "Wrist RGB-D",
                    "sensor_folder": "realsense_wrist-1",
                    "profile_id": "profile-wrist-1",
                    "robot_pose_time_offset_ms": 70.0,
                    "sync_delta_ms": -70.0,
                    "frame_timestamp_source": "sensor",
                    "robot_timestamp_source": "host_wall",
                    "required_frame_timestamp_domain": "global_time",
                    "timestamp_fallback_allowed": False,
                    "max_nearest_pose_delta_ms": 20.0,
                }
            ],
        },
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
    config_payload: dict | None = None,
) -> None:
    requests = requests if requests is not None else []
    preflight_state = preflight_state if preflight_state is not None else {"blocker": None}
    config_payload = config_payload if config_payload is not None else run_config()

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
    page.route(
        "**/ui/overview**",
        lambda route: fulfill_json(route, overview_payload(config_payload)),
    )
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
            fulfill_json(route, {"config": config_payload, "output": "written"}, status=201)
        else:
            fulfill_json(route, {"config": config_payload, "preflight": {"queue_blocker": preflight_state["blocker"]}})
    page.route("**/run-config**", config_handler)

    def pipeline_handler(route) -> None:
        requests.append({"path": "/pipeline/run", "body": route.request.post_data_json})
        fulfill_json(route, {"job_id": f"job-{len(requests)}", "status": "queued"}, status=202)
    page.route("**/pipeline/run", pipeline_handler)
    page.route("**/sensors/previews/stop", lambda route: (requests.append({"path": "/sensors/previews/stop", "body": {}}), fulfill_json(route, {"jobs": []}))[1])


def test_navigation_run_fallback_persistence_and_both_themes(console_server, page) -> None:
    install_common_mocks(page)
    page.emulate_media(color_scheme="dark")
    page.add_init_script("if (!localStorage.getItem('posetestbot.selectedRun')) localStorage.setItem('posetestbot.selectedRun', '/tmp/posetestbot-console/deleted-run'); localStorage.removeItem('posetestbot.theme')")

    page.goto(console_server.url, wait_until="networkidle")

    expect(page.locator("html")).to_have_class("dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_attribute(
        "src", "/assets/cow_dark.png"
    )
    primary_navigation = page.get_by_role("navigation", name="Primary navigation")
    assert primary_navigation.get_by_role("link").all_inner_texts() == [
        "Dashboard",
        "Workflow",
        "Devices",
        "Calibration Targets",
        "Workpiece Catalogue",
        "Pose Templates",
        "Cell View",
        "Jobs",
    ]
    expect(primary_navigation.get_by_role("link", name="Workflow")).to_have_attribute(
        "href", "#/workflow/setup"
    )
    expect(page.get_by_role("link", name="Open camera calibration", exact=True)).to_have_attribute(
        "href", "#/workflow/calibration"
    )
    assert page.evaluate("localStorage.getItem('posetestbot.theme')") is None
    assert page.evaluate("localStorage.getItem('posetestbot.selectedRun')") is None
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("new-run")
    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    page.get_by_role("complementary", name="Application sidebar").get_by_role(
        "link", name="Devices"
    ).click()
    expect(page).to_have_url(f"{console_server.url}/#/devices")
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text("old-run")
    page.get_by_role("button", name="Choose run path").click()
    expect(page.locator("#new-run-path")).to_have_value("/tmp/posetestbot-console/old-run")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Open operator console guide").click()
    expect(page.get_by_role("heading", name="Operator console guide")).to_be_visible()
    expect(page.get_by_text("Choose an outcome in Workflow", exact=True)).to_be_visible()
    expect(page.get_by_text("IIWA STOP is not a safety stop", exact=False)).to_be_visible()
    page.keyboard.press("Escape")
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
    sidebar_rgb = page.get_by_role(
        "complementary", name="Application sidebar"
    ).evaluate(
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


def test_primary_navigation_resets_document_scroll_position(console_server, page) -> None:
    install_common_mocks(page)
    page.goto(f"{console_server.url}/#/workflow/dataset", wait_until="networkidle")
    page.evaluate("window.scrollTo(0, 1200)")
    assert page.evaluate("window.scrollY") > 500

    page.get_by_role("complementary", name="Application sidebar").get_by_role(
        "link", name="Devices"
    ).click()

    expect(page).to_have_url(f"{console_server.url}/#/devices")
    expect(page.get_by_role("heading", name="Devices")).to_be_visible()
    page.wait_for_function("window.scrollY === 0")
    assert page.evaluate("window.scrollY") == 0


def test_workflow_chooser_distinguishes_numbered_required_journeys(
    console_server,
    page,
) -> None:
    install_common_mocks(page)

    page.goto(f"{console_server.url}/#/workflow/setup", wait_until="networkidle")

    expect(page.get_by_role("heading", name="What do you want to do?")).to_be_visible()
    expect(page.get_by_role("heading", name="Calibrate cameras")).to_be_visible()
    expect(page.get_by_role("heading", name="Record an object dataset")).to_be_visible()
    expect(page.get_by_text("Each guided workflow shows the required order", exact=False)).to_be_visible()

    outlines = page.locator("main ol")
    expect(outlines).to_have_count(2)
    assert outlines.nth(0).locator("li").all_inner_texts() == [
        "01Configure the run and cameras",
        "02Choose the printed calibration grid",
        "03Check readiness",
        "04Record calibration images",
        "05Calculate, review, and publish",
    ]
    assert outlines.nth(1).locator("li").all_inner_texts() == [
        "01Configure cameras and select calibration",
        "02Choose the object template and placement",
        "03Check readiness",
        "04Record the object dataset",
        "05Synchronize and verify frames",
        "06Export the BOP dataset",
    ]

    workflow_links = page.get_by_role("link", name="Start this workflow")
    expect(workflow_links).to_have_count(2)
    expect(workflow_links.nth(0)).to_have_attribute("href", "#/workflow/calibration")
    expect(workflow_links.nth(1)).to_have_attribute("href", "#/workflow/dataset")


def test_workflow_stepper_connectors_follow_numbered_steps(
    console_server,
    page,
) -> None:
    install_common_mocks(page)

    page.goto(f"{console_server.url}/#/workflow/calibration", wait_until="networkidle")

    stepper = page.get_by_role("navigation", name="Required workflow steps")
    expect(stepper).to_be_visible()
    expect(page).to_have_url(
        f"{console_server.url}/#/workflow/calibration?step=target"
    )
    expect(page.get_by_role("heading", name="Calibrate cameras")).to_be_in_viewport()
    expect(stepper.locator('[aria-current="step"]')).to_contain_text(
        "Choose the printed calibration grid"
    )
    steps = stepper.locator("li")
    connectors = stepper.locator("[data-workflow-step-connector]")
    expect(steps).to_have_count(5)
    expect(connectors).to_have_count(4)

    for index in range(4):
        step_box = steps.nth(index).bounding_box()
        number_box = steps.nth(index).locator("[data-workflow-step-number]").bounding_box()
        connector_box = connectors.nth(index).bounding_box()
        next_step_box = steps.nth(index + 1).bounding_box()
        assert step_box is not None
        assert number_box is not None
        assert connector_box is not None
        assert next_step_box is not None
        assert connector_box["width"] == pytest.approx(1, abs=0.5)
        assert connector_box["height"] == pytest.approx(20, abs=0.5)
        assert connector_box["x"] + connector_box["width"] / 2 == pytest.approx(
            number_box["x"] + number_box["width"] / 2,
            abs=1,
        )
        assert connector_box["y"] >= step_box["y"] + step_box["height"] - 21
        assert connector_box["y"] + connector_box["height"] <= next_step_box["y"] + 1

    page.set_viewport_size({"width": 900, "height": 900})
    for index in range(4):
        button_box = steps.nth(index).get_by_role("button").bounding_box()
        connector_box = connectors.nth(index).bounding_box()
        assert button_box is not None
        assert connector_box is not None
        assert connector_box["width"] >= 20
        assert connector_box["height"] == pytest.approx(1, abs=0.5)
        assert connector_box["x"] >= button_box["x"] + button_box["width"]


def test_new_run_path_renders_guided_setup_when_run_config_is_missing(
    console_server,
    page,
) -> None:
    install_common_mocks(page)
    empty_overview = overview_payload()
    empty_overview["config"] = None
    empty_overview["config_error"] = "run_config.json does not exist"
    empty_overview["steps"] = []
    page.route(
        "**/ui/overview**",
        lambda route: fulfill_json(route, empty_overview),
    )
    page.route(
        "**/run-config**",
        lambda route: fulfill_json(
            route,
            {"output": "run_config.json does not exist"},
            status=404,
        ),
    )
    page.route(
        "**/calibration/setup**",
        lambda route: fulfill_json(
            route,
            {"output": "run_config.json does not exist"},
            status=404,
        ),
    )
    page.route(
        "**/sensors/status",
        lambda route: fulfill_json(route, selected_sensor_status()),
    )
    page.add_init_script(
        "localStorage.setItem('posetestbot.selectedSensors', "
        "JSON.stringify(['realsense_d435:wrist-1', 'realsense_d435:static-1']))"
    )
    fresh_run = "/tmp/posetestbot-console/fresh-calibration-run"

    page.goto(f"{console_server.url}/#/devices", wait_until="networkidle")
    page.get_by_role("button", name="Choose run path").click()
    page.locator("#new-run-path").fill(fresh_run)
    page.get_by_role("button", name="Use run", exact=True).click()
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    expect(page.get_by_role("heading", name="Calibrate cameras")).to_be_visible()
    setup = page.get_by_test_id("camera_calibration-run-setup")
    expect(setup).to_be_visible()
    expect(setup.get_by_test_id("run-camera-row")).to_have_count(2)
    expect(setup.get_by_role("button", name="Save setup")).to_be_enabled()
    expect(page.get_by_role("navigation", name="Required workflow steps")).to_be_visible()
    expect(page.get_by_role("combobox", name="Selected run")).to_contain_text(fresh_run)


def test_responsive_shell_and_dataset_workflow_links(console_server, page) -> None:
    config = run_config()
    config["dataset_mode"] = "pose_template"
    install_common_mocks(page, config_payload=config)
    page.set_viewport_size({"width": 900, "height": 900})

    page.goto(f"{console_server.url}/#/dashboard", wait_until="networkidle")

    expect(page.locator("aside")).to_be_hidden()
    primary_navigation = page.get_by_role("navigation", name="Primary navigation")
    expect(primary_navigation).to_have_count(1)
    expect(primary_navigation).to_be_visible()
    expect(primary_navigation.get_by_role("link", name="Calibration Targets")).to_be_visible()
    expect(primary_navigation.get_by_role("link", name="Pose Templates")).to_be_visible()
    expect(page.get_by_role("link", name="Open object dataset", exact=True)).to_have_attribute(
        "href", "#/workflow/dataset"
    )
    expect(page.get_by_text("Object dataset evidence", exact=True)).to_be_visible()
    assert page.evaluate("getComputedStyle(document.body).minWidth") == "0px"
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )


def pose_template_source(*, available: bool) -> dict:
    return {
        "status": "available" if available else "missing",
        "available": available,
        "checkout": "/repo/third_party/PoseTemplateCreator",
        "required_revision": "97ddb9b7b756912deb8c2d2d6dde186b461e5d9d",
        "revision": "97ddb9b7b756912deb8c2d2d6dde186b461e5d9d" if available else None,
        "reason": None if available else "PoseTemplateCreator checkout is missing",
        "capabilities": {
            "formats": ["ply", "stl", "obj"],
            "limits": {"cad_bytes": 52428800, "batch_bytes": 104857600, "faces": 1000000, "contour_vertices": 10000, "instances": 200},
        },
    }


def pose_template_catalog() -> dict:
    return {
        "schema_version": "object_catalog.v1",
        "objects": [{
            "catalog_uuid": "11111111-1111-4111-8111-111111111111",
            "obj_id": 7,
            "name": "Clamp",
            "alias": "Small clamp",
            "description": "Textured fixture",
            "tags": ["metal", "reflective"],
            "groups": ["clamps", "validation set"],
            "attributes": {"owner": "vision", "finish": "matte"},
            "source_filename": "clamp.stl",
            "source_format": "stl",
            "source_sha256": "a" * 64,
            "canonical_ply_sha256": "b" * 64,
            "geometry_revision": 1,
            "source_to_mm_scale": 1.0,
            "texture_sha256": "c" * 64,
            "created_at": "2026-07-20T09:00:00Z",
            "updated_at": "2026-07-20T10:00:00Z",
            "archived_at": None,
            "state": "active",
            "extraction": {"vertices": 8, "faces": 12, "bounds_mm": [[-5, -5, -5], [5, 5, 5]], "watertight": True},
            "assets": {
                "source": {"path": "objects/1/source/clamp.stl", "sha256": "a" * 64, "size_bytes": 100, "media_type": "application/octet-stream"},
                "canonical_ply": {"path": "objects/1/derived/canonical.ply", "sha256": "b" * 64, "size_bytes": 80, "media_type": "application/octet-stream"},
                "texture": {"path": "objects/1/texture/texture.png", "sha256": "c" * 64, "size_bytes": 40, "media_type": "image/png"},
            },
            "usage": {"template_count": 0, "templates": []},
        }],
    }


def workpiece_catalog() -> dict:
    value = pose_template_catalog()
    value.update({
        "version": 4,
        "created_at": "2026-07-20T09:00:00Z",
        "updated_at": "2026-07-21T11:00:00Z",
        "next_obj_id": 9,
        "tombstones": [],
    })
    value["objects"].append({
        "catalog_uuid": "88888888-8888-4888-8888-888888888888",
        "obj_id": 8,
        "name": "Gauge block",
        "alias": "Archived gauge",
        "description": "Reference ceramic block",
        "tags": ["Metal", "reference"],
        "groups": ["gauges"],
        "attributes": {"length_mm": "25"},
        "source_filename": "gauge.ply",
        "source_format": "ply",
        "source_sha256": "d" * 64,
        "canonical_ply_sha256": "e" * 64,
        "geometry_revision": 1,
        "source_to_mm_scale": 1.0,
        "texture_sha256": None,
        "created_at": "2026-07-20T09:30:00Z",
        "updated_at": "2026-07-21T11:00:00Z",
        "archived_at": "2026-07-21T11:00:00Z",
        "state": "archived",
        "extraction": {"vertices": 8, "faces": 12, "bounds_mm": [[-12.5, -5, -2.5], [12.5, 5, 2.5]], "watertight": True},
        "assets": {
            "source": {"path": "objects/8/source/gauge.ply", "sha256": "d" * 64, "size_bytes": 120, "media_type": "application/octet-stream"},
            "canonical_ply": {"path": "objects/8/derived/canonical.ply", "sha256": "e" * 64, "size_bytes": 90, "media_type": "application/octet-stream"},
        },
        "usage": {"template_count": 0, "templates": []},
    })
    return value


def pose_template_library() -> dict:
    return {
        "schema_version": "pose_template_library.v1",
        "templates": [{
            "template_uuid": "22222222-2222-4222-8222-222222222222",
            "display_name": "Clamp pair",
            "description": "fixture",
            "created_at": "2026-07-20T10:00:00Z",
            "bundle_sha256": "d" * 64,
            "archive": {"state": "active"},
            "page": {"size": "A3", "orientation": "landscape", "width_mm": 420, "height_mm": 297},
            "instances": [{
                "instance_uuid": "33333333-3333-4333-8333-333333333333",
                "catalog_uuid": "11111111-1111-4111-8111-111111111111",
                "catalog": {"catalog_uuid": "11111111-1111-4111-8111-111111111111", "name": "Clamp", "obj_id": 7},
                "pose_template_from_object": {"matrix": [[1, 0, 0, 45], [0, 1, 0, 55], [0, 0, 1, 0], [0, 0, 0, 1]]},
            }],
        }],
    }


def pose_template_orientation_analysis(catalog_uuid: str = "11111111-1111-4111-8111-111111111111") -> dict:
    return {
        "schema_version": "pose_template_orientation_analysis.v1",
        "catalog_uuid": catalog_uuid,
        "preview_mesh": {
            "vertices": [[-10, -5, 0], [10, -5, 0], [10, 5, 0], [-10, 5, 0], [0, 0, 12]],
            "faces": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]],
        },
        "recognition_mesh": {
            "vertices": [[-10, -5, 0], [10, -5, 0], [10, 5, 0], [-10, 5, 0], [-10, -5, 12], [10, -5, 12], [10, 5, 12], [-10, 5, 12]],
            "faces": [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]],
        },
        "recognition_mesh_approximation": {
            "strategy": "welded_source",
            "implementation_revision": "posetestbot_posetemplatecreator_adapter.v4",
            "source_vertices": 8,
            "source_faces": 12,
            "welded_vertices": 8,
            "welded_faces": 12,
            "result_vertices": 8,
            "result_faces": 12,
            "source_components": 1,
            "source_euler_number": 2,
            "result_components": 1,
            "result_euler_number": 2,
            "topology_preserved": True,
            "spatial_resolution": None,
            "fallback_reason": None,
        },
        "orientations": [
            {
                "orientation_id": "stable-wide",
                "label": "Wide base",
                "probability": 0.82,
                "source_to_placed": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "slice_z_mm": 0.1,
                "contours": [{"points": [{"x_mm": -10, "y_mm": -5}, {"x_mm": 10, "y_mm": -5}, {"x_mm": 7, "y_mm": 5}, {"x_mm": -10, "y_mm": 3}]}],
            },
            {
                "orientation_id": "stable-side",
                "label": "Side base",
                "probability": 0.18,
                "source_to_placed": [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 5], [0, 0, 0, 1]],
                "slice_z_mm": 0.1,
                "contours": [{"points": [{"x_mm": -10, "y_mm": -6}, {"x_mm": 10, "y_mm": -6}, {"x_mm": 10, "y_mm": 6}, {"x_mm": -10, "y_mm": 6}]}],
            },
        ],
    }


def pose_template_orientation_thumbnail(catalog_uuid: str = "11111111-1111-4111-8111-111111111111") -> dict:
    analysis = pose_template_orientation_analysis(catalog_uuid)
    orientation = analysis["orientations"][0]
    return {
        "schema_version": "pose_template_orientation_thumbnail.v1",
        "catalog_uuid": catalog_uuid,
        "catalog": {"catalog_uuid": catalog_uuid},
        "source": {"canonical_ply_sha256": "b" * 64, "geometry_revision": 1},
        "preview_mesh": analysis["preview_mesh"],
        "orientation": {
            "orientation_id": orientation["orientation_id"],
            "label": orientation["label"],
            "rank": 1,
            "probability": orientation["probability"],
            "slice_z_mm": orientation["slice_z_mm"],
            "source_to_placed": orientation["source_to_placed"],
        },
    }


def immutable_template_preview() -> dict:
    return {
        "schema_version": "pose_template_preview.v1",
        "valid": True,
        "configuration_sha256": "e" * 64,
        "configuration": {
            "page": {"origin_from_lower_left_mm": [15, 15]},
            "print_compensation": {"x_scale": 1.01, "y_scale": 0.99},
        },
        "page": {"width_mm": 420, "height_mm": 297},
        "instances": [{
            "instance_uuid": "33333333-3333-4333-8333-333333333333",
            "catalog_uuid": "11111111-1111-4111-8111-111111111111",
            "catalog": {"name": "Clamp", "obj_id": 7},
            "pose_template_from_object": {"matrix": [[1, 0, 0, 45], [0, 1, 0, 55], [0, 0, 1, 0], [0, 0, 0, 1]]},
            "preview_mesh_sha256": "b" * 64,
            "compensated_contours": [
                [{"x_mm": 30, "y_mm": 30}, {"x_mm": 50, "y_mm": 30}, {"x_mm": 47, "y_mm": 42}, {"x_mm": 30, "y_mm": 40}],
                [{"x_mm": 36, "y_mm": 34}, {"x_mm": 40, "y_mm": 34}, {"x_mm": 40, "y_mm": 37}, {"x_mm": 36, "y_mm": 37}],
            ],
        }],
        "preview_meshes": {
            "b" * 64: {
                "vertices": [[-10, -5, 0], [10, -5, 0], [10, 5, 0], [-10, 5, 0], [0, 0, 12]],
                "faces": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]],
            }
        },
        "errors": [],
    }


def immutable_template_thumbnail() -> dict:
    preview = immutable_template_preview()
    contours = preview["instances"][0]["compensated_contours"]
    point_count = sum(len(contour) for contour in contours)
    return {
        "schema_version": "pose_template_thumbnail.v1",
        "template_uuid": "22222222-2222-4222-8222-222222222222",
        "valid": True,
        "configuration": preview["configuration"],
        "page": preview["page"],
        "instances": [{
            "instance_uuid": preview["instances"][0]["instance_uuid"],
            "catalog": preview["instances"][0]["catalog"],
            "compensated_contours": contours,
            "primary_contour_source_index": 0,
            "approximation": {
                "truncated": False,
                "source_contours": len(contours),
                "included_contours": len(contours),
                "source_points": point_count,
                "included_points": point_count,
            },
        }],
        "approximation": {
            "approximate": False,
            "truncated": False,
            "strategy": "largest-primary-then-round-robin-even-decimation",
            "source_contours": len(contours),
            "included_contours": len(contours),
            "source_points": point_count,
            "included_points": point_count,
            "limits": {"instances": 200, "contours": 400, "points": 4096, "points_per_contour": 48},
        },
    }


def test_pose_templates_editor_catalog_generation_and_unavailable_browse(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script("Object.defineProperty(Crypto.prototype, 'randomUUID', { value: undefined, configurable: true })")
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    requests: list[dict] = []
    preview_posts = {"count": 0}
    availability = {"available": True}
    orientation_ready = {"value": False}
    library_payload = pose_template_library()
    page.route("**/pose-templates/status", lambda route: fulfill_json(route, pose_template_source(available=availability["available"])))
    page.route("**/workpieces/catalog", lambda route: fulfill_json(route, pose_template_catalog()))
    page.route("**/pose-templates/library", lambda route: fulfill_json(route, library_payload))
    def orientation_handler(route) -> None:
        if route.request.method == "POST":
            orientation_ready["value"] = True
            fulfill_json(route, {"job_id": "orientation-job", "request_id": "d" * 32}, status=202)
        elif orientation_ready["value"]:
            fulfill_json(route, pose_template_orientation_analysis())
        else:
            fulfill_json(route, {"output": "Cached orientation analysis is stale", "analysis_required": True}, status=409)

    page.route("**/pose-templates/workpieces/*/orientations", orientation_handler)
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(route, pose_template_orientation_thumbnail()) if orientation_ready["value"] else fulfill_json(route, {"output": "Orientation thumbnail unavailable", "analysis_required": True}, status=404),
    )
    page.route("**/pose-templates/library/*/preview", lambda route: fulfill_json(route, immutable_template_preview()))
    page.route("**/pose-templates/library/*/thumbnail", lambda route: fulfill_json(route, immutable_template_thumbnail()))
    page.route("**/jobs/generate-job", lambda route: fulfill_json(route, {"job": {"id": "generate-job", "status": "succeeded", "message": None, "tail": []}}))
    page.route("**/jobs/clone-job", lambda route: fulfill_json(route, {"job": {"id": "clone-job", "status": "failed", "message": "Command exited with status 1", "tail": ["Canonical geometry changed; analyze stable orientations again.", "Command exited with code 1"]}}))
    page.route("**/jobs/orientation-job", lambda route: fulfill_json(route, {"job": {"id": "orientation-job", "status": "succeeded", "message": None, "tail": []}}))
    page.route("**/jobs/delete-cleanup-job", lambda route: fulfill_json(route, {"job": {"id": "delete-cleanup-job", "status": "running", "message": None, "tail": []}}))

    def preview_handler(route) -> None:
        if route.request.method == "POST":
            preview_posts["count"] += 1
            if preview_posts["count"] == 1:
                fulfill_json(route, {"output": "Resources busy: cpu, disk_io"}, status=409)
                return
            requests.append({"path": "/pose-templates/preview", "body": route.request.post_data_json})
            fulfill_json(route, {"job_id": "preview-job", "request_id": "a" * 32}, status=202)
        else:
            fulfill_json(route, immutable_template_preview())
    page.route("**/pose-templates/preview**", preview_handler)
    page.route("**/pose-templates/generate", lambda route: (requests.append({"path": "/pose-templates/generate", "body": route.request.post_data_json}), fulfill_json(route, {"job_id": "generate-job"}, status=202))[1])
    page.route("**/pose-templates/library/*/clone", lambda route: (requests.append({"path": "/library/clone", "body": {}}), fulfill_json(route, {"job_id": "clone-job"}, status=202))[1])
    def template_delete_handler(route) -> None:
        requests.append({
            "path": "/library/delete",
            "method": route.request.method,
            "body": route.request.post_data_json,
        })
        library_payload["templates"].clear()
        fulfill_json(
            route,
            {
                "schema_version": "pose_template_library_delete.v1",
                "status": "deleted_cleanup_pending",
                "job_id": "delete-cleanup-job",
                "asset_cleanup": {"status": "pending", "last_error": None},
            },
            status=202,
        )

    page.route(
        "**/pose-templates/library/22222222-2222-4222-8222-222222222222",
        template_delete_handler,
    )

    page.goto(f"{console_server.url}/#/pose-templates", wait_until="networkidle")
    expect(page.get_by_test_id("pose-templates-page")).to_be_visible()
    expect(page.get_by_role("link", name="Pose Templates")).to_be_visible()
    expect(page.get_by_text("Clamp", exact=True)).to_be_visible()
    expect(page.get_by_text("Small clamp", exact=False)).to_be_visible()
    manage = page.get_by_role("link", name="Manage catalogue")
    expect(manage).to_be_visible()
    expect(manage).to_have_attribute("href", "#/workpieces")
    expect(page.get_by_role("button", name="Upload CAD")).to_have_count(0)
    library_thumbnail = page.get_by_test_id("template-thumbnail-22222222-2222-4222-8222-222222222222")
    expect(library_thumbnail.locator("path")).to_have_count(1)
    expect(library_thumbnail.locator("path")).to_have_attribute("fill-rule", "evenodd")
    expect(library_thumbnail.locator('g[transform="translate(0 297) scale(1 -1)"]')).to_have_count(1)
    page.get_by_role("textbox", name="Filter template workpieces").fill("no such object")
    expect(page.get_by_text("No active workpieces match these filters.")).to_be_visible()
    page.get_by_role("textbox", name="Filter template workpieces").fill("small clamp")
    expect(page.get_by_text("Clamp", exact=True)).to_be_visible()
    page.get_by_role("textbox", name="Filter template workpieces").fill("")
    expect(page.get_by_label("X print %")).to_have_value("100")
    expect(page.get_by_label("Y print %")).to_have_value("100")
    assert page.get_by_test_id("pose-template-preview-canvas").evaluate("element => getComputedStyle(element).backgroundColor") != "rgb(255, 255, 255)"
    page.get_by_role("button", name="Choose orientation for Clamp").click()
    chooser = page.get_by_test_id("orientation-chooser")
    expect(chooser).to_contain_text("same-scale high-detail recognition surface")
    expect(chooser).to_contain_text("tiny printable-layout proxy is not used")
    wide_slice = chooser.get_by_role("img", name="Wide base exact selected slice contour")
    expect(wide_slice.locator("path")).to_have_attribute("fill-rule", "evenodd")
    expect(wide_slice.locator("path")).to_have_attribute("transform", "translate(0 0) scale(1 -1)")
    expect(page.get_by_test_id("workpiece-isometric-11111111-1111-4111-8111-111111111111").locator("polygon")).to_have_count(12)
    wide_preview = page.get_by_test_id("orientation-isometric-stable-wide")
    expect(wide_preview.locator("polygon")).to_have_count(12)
    expect(wide_preview.locator("xpath=..")).to_have_attribute("data-preview-quality", "recognition")
    expect(page.get_by_test_id("orientation-preview-quality-stable-wide")).to_contain_text("Full recognition surface · 12 of 12 source faces")
    wide_points = wide_preview.locator("polygon").first.get_attribute("points")
    side_points = page.get_by_test_id("orientation-isometric-stable-side").locator("polygon").first.get_attribute("points")
    assert wide_points != side_points
    chooser.get_by_role("radio").filter(has_text="Side base").click()
    chooser.get_by_role("button", name="Add selected orientation").click()
    expect(page.get_by_label("Clamp X mm")).to_be_visible()
    expect(page.get_by_test_id("selected-instance-isometric").locator("polygon")).to_have_count(12)
    assert page_errors == []
    page.get_by_label("Clamp Rotation °").fill("27.5")
    page.get_by_label("X print %").fill("101")
    expect(page.get_by_role("button", name="Generate immutable version")).to_be_enabled(timeout=15_000)
    assert preview_posts["count"] >= 2
    page.get_by_role("button", name="Generate immutable version").click()
    expect(page.get_by_text("Immutable template generation queued")).to_be_visible()
    assert requests[-1]["body"]["configuration"]["instances"][0]["orientation_id"] == "stable-side"
    assert requests[-1]["body"]["configuration"]["instances"][0]["pose"]["rotation_deg"] == 27.5
    expect(page.get_by_role("button", name="Clone")).to_be_enabled(timeout=15_000)
    page.get_by_role("button", name="Clone").click()
    expect(page.get_by_text("Canonical geometry changed; analyze stable orientations again.")).to_be_visible(timeout=15_000)
    assert {item["path"] for item in requests} >= {"/pose-templates/generate", "/library/clone"}
    generation = next(item for item in reversed(requests) if item["path"] == "/pose-templates/generate")
    assert generation["body"]["configuration"]["print_compensation"]["x_scale"] == 1.01
    page.get_by_role("button", name="Delete Clamp pair").click()
    deletion = page.get_by_test_id("pose-template-delete-confirmation")
    expect(deletion).to_contain_text("library entry is removed immediately")
    expect(deletion).to_contain_text("continues after navigation")
    expect(deletion).to_contain_text("Existing run-owned snapshots remain intact")
    deletion.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Pose template deleted")).to_be_visible()
    expect(page.get_by_test_id("pose-template-library-card-22222222-2222-4222-8222-222222222222")).to_have_count(0)
    cleanup = page.get_by_test_id("pose-template-cleanup-job")
    expect(cleanup).to_contain_text("cleanup continues after navigation")
    expect(cleanup.get_by_role("link", name="Open Jobs")).to_have_attribute("href", "#/jobs")
    delete_request = next(item for item in requests if item["path"] == "/library/delete")
    assert delete_request == {
        "path": "/library/delete",
        "method": "DELETE",
        "body": {"confirm": True},
    }

    availability["available"] = False
    page.reload(wait_until="networkidle")
    expect(page.get_by_text("PoseTemplateCreator checkout is missing")).to_be_visible()
    expect(page.get_by_text("bash scripts/install.sh --with-posetemplatecreator")).to_be_visible()
    expect(page.get_by_text("Clamp", exact=True)).to_be_visible()
    expect(page.get_by_role("link", name="Manage catalogue")).to_be_visible()


def test_workpiece_catalogue_metadata_filters_actions_import_and_upload(
    console_server, page
) -> None:
    install_common_mocks(page)
    page.add_init_script("HTMLCanvasElement.prototype.getContext = () => null")
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    catalogue = workpiece_catalog()
    catalog_requests = {"count": 0}
    delete_requests = {"count": 0}
    requests: list[dict] = []

    def item(catalog_uuid: str) -> dict:
        return next(
            value
            for value in catalogue["objects"]
            if value["catalog_uuid"] == catalog_uuid
        )

    def status_handler(route) -> None:
        active = sum(value["state"] == "active" for value in catalogue["objects"])
        fulfill_json(route, {
            "schema_version": "workpiece_catalog_status.v1",
            "available": True,
            "status": "available",
            "reason": None,
            "catalog_root": "/repo/working_data/object_catalog",
            "formats": ["ply", "stl", "obj"],
            "limits": {"cad_bytes": 52428800, "batch_bytes": 104857600},
            "counts": {
                "active": active,
                "archived": len(catalogue["objects"]) - active,
                "total": len(catalogue["objects"]),
            },
        })

    def catalog_handler(route) -> None:
        request = route.request
        path = urlparse(request.url).path
        if path == "/workpieces/catalog" and request.method == "GET":
            catalog_requests["count"] += 1
            fulfill_json(route, catalogue)
            return
        if path == "/workpieces/catalog/import" and request.method == "POST":
            requests.append({"path": path, "method": request.method, "body": request.post_data or ""})
            fulfill_json(route, {
                "schema_version": "workpiece_catalog_import.v1",
                "updated": [catalogue["objects"][0]["catalog_uuid"]],
                "unchanged": [catalogue["objects"][1]["catalog_uuid"]],
                "skipped_missing_assets": [],
            })
            return
        if path == "/workpieces/catalog/upload" and request.method == "POST":
            requests.append({"path": path, "method": request.method, "body": request.post_data or ""})
            catalogue["objects"].append({
                "catalog_uuid": "99999999-9999-4999-8999-999999999999",
                "obj_id": 9,
                "name": "New clamp",
                "alias": "Queued workpiece",
                "description": None,
                "tags": ["new", "metal"],
                "groups": ["incoming"],
                "attributes": {},
                "source_filename": "new-clamp.stl",
                "source_format": "stl",
                "source_sha256": "f" * 64,
                "canonical_ply_sha256": "1" * 64,
                "texture_sha256": None,
                "created_at": "2026-07-22T12:00:00Z",
                "updated_at": "2026-07-22T12:00:00Z",
                "archived_at": None,
                "state": "active",
                "extraction": {"vertices": 8, "faces": 12, "bounds_mm": [[-4, -4, -4], [4, 4, 4]], "watertight": True},
                "assets": {
                    "source": {"path": "objects/9/source/new-clamp.stl", "sha256": "f" * 64},
                    "canonical_ply": {"path": "objects/9/derived/canonical.ply", "sha256": "1" * 64},
                },
                "usage": {"template_count": 0, "templates": []},
            })
            fulfill_json(route, {"job_id": "workpiece-upload-job", "request_id": "a" * 32}, status=202)
            return
        parts = path.removeprefix("/workpieces/catalog/").split("/")
        catalog_uuid = parts[0]
        current = item(catalog_uuid)
        if len(parts) == 2 and parts[1] == "unit-corrections" and request.method == "POST":
            body = request.post_data_json
            requests.append({"path": path, "method": request.method, "body": body})
            current["geometry_revision"] = 2
            current["source_to_mm_scale"] = 0.001 if body["conversion"] == "millimeter_to_meter" else 1000.0
            current["canonical_ply_sha256"] = "2" * 64
            factor = 0.001 if body["conversion"] == "millimeter_to_meter" else 1000.0
            current["extraction"]["bounds_mm"] = [[coordinate * factor for coordinate in corner] for corner in current["extraction"]["bounds_mm"]]
            fulfill_json(route, {"job_id": "unit-correction-job", "request_id": "b" * 32}, status=202)
            return
        if len(parts) == 1 and request.method == "PATCH":
            body = request.post_data_json
            requests.append({"path": path, "method": request.method, "body": body})
            current.update(body)
            current["updated_at"] = "2026-07-22T12:30:00Z"
            fulfill_json(route, current)
            return
        if len(parts) == 2 and parts[1] in {"archive", "restore"} and request.method == "POST":
            requests.append({"path": path, "method": request.method, "body": None})
            current["state"] = "archived" if parts[1] == "archive" else "active"
            current["archived_at"] = "2026-07-22T12:45:00Z" if parts[1] == "archive" else None
            fulfill_json(route, current)
            return
        if len(parts) == 1 and request.method == "DELETE":
            requests.append({"path": path, "method": request.method, "body": request.post_data_json})
            delete_requests["count"] += 1
            if delete_requests["count"] == 1:
                fulfill_json(route, {
                    "output": "Workpiece is referenced by or cannot be checked against pose-template bundles",
                    "blockers": [{
                        "template_uuid": "22222222-2222-4222-8222-222222222222",
                        "display_name": "Clamp pair",
                        "state": "active",
                        "reason": "catalog_reference",
                    }],
                }, status=409)
                return
            catalogue["objects"].remove(current)
            fulfill_json(route, {"schema_version": "workpiece_catalog_delete.v1", "status": "deleted"})
            return
        fulfill_json(route, {"output": "Unexpected workpiece request"}, status=404)

    page.route("**/workpieces/status", status_handler)
    page.route("**/workpieces/catalog**", catalog_handler)
    page.route(
        "**/pose-templates/workpieces/*/orientations",
        lambda route: fulfill_json(
            route,
            pose_template_orientation_analysis(
                urlparse(route.request.url).path.split("/")[-2]
            ),
        ),
    )
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(
            route,
            pose_template_orientation_thumbnail(
                urlparse(route.request.url).path.split("/")[-2]
            ),
        ),
    )
    page.route("**/jobs/workpiece-upload-job", lambda route: fulfill_json(route, {"job": {"id": "workpiece-upload-job", "status": "succeeded", "message": None, "tail": []}}))
    page.route("**/jobs/unit-correction-job", lambda route: fulfill_json(route, {"job": {"id": "unit-correction-job", "status": "succeeded", "message": None, "tail": []}}))

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")

    expect(page.get_by_test_id("workpieces-page")).to_be_visible()
    expect(page.get_by_role("link", name="Workpiece Catalogue")).to_be_visible()
    expect(page.get_by_test_id("workpiece-preview-fallback")).to_be_visible()
    expect(page.get_by_text("3D preview is unavailable")).to_be_visible()
    expect(page.get_by_role("heading", name="3D preview")).to_be_visible()
    expect(page.get_by_test_id("workpiece-previews")).to_have_count(0)
    expect(page.get_by_role("button", name="Select Clamp")).to_be_visible()
    expect(page.get_by_test_id("workpiece-isometric-11111111-1111-4111-8111-111111111111").locator("polygon")).to_have_count(6)
    expect(page.get_by_role("button", name="Select Gauge block")).to_have_count(0)

    page.get_by_label("Search workpieces").fill("not present")
    expect(page.get_by_text("No matches", exact=True)).to_be_visible()
    page.get_by_label("Search workpieces").fill("small clamp")
    expect(page.get_by_role("button", name="Select Clamp")).to_be_visible()
    page.get_by_label("Search workpieces").fill("")

    page.get_by_role("combobox", name="Filter by tag").click()
    expect(page.get_by_role("option", name="metal", exact=True)).to_have_count(1)
    page.get_by_role("option", name="reflective").click()
    expect(page.get_by_text("1 of 2 visible")).to_be_visible()
    page.get_by_role("combobox", name="Filter by group").click()
    page.get_by_role("option", name="clamps").click()
    expect(page.get_by_role("button", name="Select Clamp")).to_be_visible()
    page.get_by_role("button", name="Clear").click()
    page.get_by_role("combobox", name="Filter by state").click()
    page.get_by_role("option", name="Archived").click()
    expect(page.get_by_role("button", name="Select Gauge block")).to_be_visible()
    expect(page.get_by_role("button", name="Select Clamp")).to_have_count(0)
    page.get_by_role("button", name="Select Gauge block").click()
    page.get_by_role("button", name="Correct model units").click()
    correction = page.get_by_test_id("workpiece-unit-correction-dialog")
    expect(correction.get_by_text("File was authored in metres — enlarge ×1000")).to_be_visible()
    expect(correction.get_by_text("Model is 1000× too large — shrink ÷1000")).to_be_visible()
    expect(correction.get_by_text("Current dimensions")).to_be_visible()
    expect(correction.get_by_text("After correction")).to_be_visible()
    correction.get_by_role("radio").filter(has_text="shrink ÷1000").click()
    correction.get_by_label("Unit correction operator").fill("qa-operator")
    correction.get_by_label("Confirm unit correction").click()
    correction.get_by_role("button", name="Queue unit correction").click()
    expect(page.get_by_text("Workpiece units corrected")).to_be_visible()
    unit_request = next(value for value in requests if value["path"].endswith("/unit-corrections"))
    assert unit_request["body"] == {
        "conversion": "millimeter_to_meter",
        "confirm": True,
        "operator": "qa-operator",
        "expected_geometry_revision": 1,
        "expected_canonical_sha256": "e" * 64,
    }
    page.get_by_role("button", name="Clear").click()
    page.get_by_role("button", name="Select Clamp").click()

    page.get_by_role("button", name="Edit metadata").click()
    page.get_by_test_id("workpiece-edit-alias").fill("Fixture A")
    page.get_by_test_id("workpiece-edit-tags").fill("metal, QA, qa")
    page.get_by_test_id("workpiece-edit-groups").fill("clamps, set-a")
    page.get_by_test_id("workpiece-edit-attribute-value-0").fill("metrology")
    page.get_by_role("button", name="Add attribute").click()
    page.get_by_role("button", name="Save metadata").click()
    expect(page.get_by_test_id("workpiece-edit-attribute-error")).to_contain_text(
        "Add a name or remove attribute row 3"
    )
    assert not any(value["method"] == "PATCH" for value in requests)
    page.get_by_test_id("workpiece-edit-attribute-key-2").fill("OWNER")
    page.get_by_test_id("workpiece-edit-attribute-value-2").fill("duplicate")
    page.get_by_role("button", name="Save metadata").click()
    expect(page.get_by_test_id("workpiece-edit-attribute-error")).to_contain_text(
        "Attribute names must be unique"
    )
    assert not any(value["method"] == "PATCH" for value in requests)
    page.get_by_test_id("workpiece-edit-attribute-key-2").fill("station")
    page.get_by_test_id("workpiece-edit-attribute-value-2").fill("2")
    page.get_by_role("button", name="Save metadata").click()
    expect(page.get_by_text("Workpiece metadata saved")).to_be_visible()
    patch_request = next(value for value in requests if value["method"] == "PATCH")
    assert patch_request["body"] == {
        "name": "Clamp",
        "alias": "Fixture A",
        "description": "Textured fixture",
        "tags": ["metal", "QA"],
        "groups": ["clamps", "set-a"],
        "attributes": {"owner": "metrology", "finish": "matte", "station": "2"},
    }

    page.get_by_test_id("workpiece-catalog-import").click()
    page.get_by_test_id("workpiece-import-input").set_input_files({
        "name": "object_catalog.json",
        "mimeType": "application/json",
        "buffer": json.dumps(workpiece_catalog()).encode(),
    })
    page.get_by_role("button", name="Import metadata").click()
    expect(page.get_by_text("Catalogue metadata imported")).to_be_visible()
    import_request = next(value for value in requests if value["path"].endswith("/import"))
    assert "object_catalog.json" in import_request["body"]

    page.get_by_test_id("workpiece-upload-button").click()
    page.get_by_test_id("workpiece-cad-input").set_input_files({
        "name": "new-clamp.stl",
        "mimeType": "application/octet-stream",
        "buffer": b"solid clamp",
    })
    page.get_by_test_id("workpiece-upload-name").fill("New clamp")
    page.get_by_test_id("workpiece-upload-alias").fill("Queued workpiece")
    page.get_by_test_id("workpiece-upload-tags").fill("new, metal")
    page.get_by_test_id("workpiece-upload-groups").fill("incoming")
    page.get_by_role("button", name="Upload and inspect").click()
    expect(page.get_by_text("Workpiece inspection queued")).to_be_visible()
    expect(page.get_by_text("Workpiece added to the catalogue")).to_be_visible()
    upload_request = next(value for value in requests if value["path"].endswith("/upload"))
    assert "new-clamp.stl" in upload_request["body"]
    assert "Queued workpiece" in upload_request["body"]
    expect(page.get_by_role("button", name="Select New clamp")).to_be_visible()
    assert catalog_requests["count"] > 1

    page.get_by_role("button", name="Select New clamp").click()
    page.get_by_role("button", name="Archive").click()
    confirmation = page.get_by_test_id("workpiece-action-confirmation")
    expect(confirmation).to_contain_text("hidden from active-object workflows")
    confirmation.get_by_role("button", name="Confirm archive").click()
    expect(page.get_by_text("Workpiece archived")).to_be_visible()
    assert any(value["path"].endswith("/archive") for value in requests)

    page.get_by_role("button", name="Restore").click()
    confirmation = page.get_by_test_id("workpiece-action-confirmation")
    expect(confirmation).to_contain_text("returns to active-object workflows")
    confirmation.get_by_role("button", name="Confirm restore").click()
    expect(page.get_by_text("Workpiece restored")).to_be_visible()
    assert any(value["path"].endswith("/restore") for value in requests)

    expect(page.get_by_role("button", name="Delete New clamp")).to_be_enabled()
    page.get_by_role("button", name="Delete New clamp").click()
    confirmation = page.get_by_test_id("workpiece-action-confirmation")
    expect(confirmation).to_contain_text("permanently removes")
    confirmation.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Catalogue action failed")).to_be_visible()
    expect(page.get_by_text("pose-template bundles")).to_be_visible()
    expect(confirmation).to_be_visible()
    confirmation.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Workpiece deleted")).to_be_visible()
    expect(page.get_by_role("button", name="Select New clamp")).to_have_count(0)
    assert delete_requests["count"] == 2
    assert page_errors == []


def test_workpiece_selected_detail_renders_exact_canonical_mesh(
    console_server, page, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page.set_viewport_size({"width": 1920, "height": 1080})
    working = tmp_path / "working"
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    cad = tmp_path / "preview-ring.stl"
    cad.write_bytes(
        trimesh.creation.annulus(
            r_min=4,
            r_max=10,
            height=8,
            sections=64,
        ).export(file_type="stl")
    )
    record = import_catalog_object(
        name="Preview ring",
        cad_path=cad,
        catalog_root=working / "object_catalog",
    )
    install_common_mocks(page)
    page_errors: list[str] = []
    canonical_mesh_requests: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "request",
        lambda request: canonical_mesh_requests.append(request.url)
        if urlparse(request.url).path.endswith("/assets/canonical_ply")
        else None,
    )
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(
            route,
            {
                "output": "Stable orientations have not been analyzed.",
                "analysis_required": True,
            },
            status=404,
        ),
    )
    page.route(
        "**/pose-templates/workpieces/*/orientations",
        lambda route: fulfill_json(
            route,
            {"job_id": "recognition-preview-job", "request_id": "f" * 32},
            status=202,
        ),
    )
    page.route(
        "**/jobs/recognition-preview-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "recognition-preview-job",
                    "status": "succeeded",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )

    with page.expect_response(
        lambda response: urlparse(response.url).path.endswith(
            f"/workpieces/catalog/{record['catalog_uuid']}/assets/canonical_ply"
        ),
        timeout=15_000,
    ) as canonical_response:
        page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")

    response = canonical_response.value
    response.finished()
    assert response.status == 200
    expect(page.get_by_test_id("workpiece-previews")).to_be_visible()
    expect(page.get_by_text("Full canonical model", exact=True)).to_be_visible()
    canvas = page.get_by_test_id("workpiece-previews").locator("canvas")
    expect(canvas).to_have_count(1)
    expect(canvas).to_have_css("height", "320px")
    expect(page.get_by_text("Loading full model…")).to_have_count(0, timeout=15_000)
    expect(page.get_by_test_id("workpiece-preview-error")).to_have_count(0)
    expect(page.get_by_test_id("workpiece-preview-fallback")).to_have_count(0)
    screenshot = cv2.imdecode(
        np.frombuffer(canvas.screenshot(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert screenshot is not None
    background = screenshot[2, 2].astype(np.int16)
    foreground = (
        np.linalg.norm(screenshot.astype(np.int16) - background, axis=2) > 24
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        connectivity=8,
    )
    height, width = foreground.shape
    object_components = [
        index
        for index in range(1, component_count)
        if stats[index, cv2.CC_STAT_AREA] > 1_000
        and stats[index, cv2.CC_STAT_LEFT] > 2
        and stats[index, cv2.CC_STAT_TOP] > 2
        and stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]
        < width - 2
        and stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT]
        < height - 2
    ]
    assert object_components
    component = max(
        object_components,
        key=lambda index: stats[index, cv2.CC_STAT_AREA],
    )
    left, top, component_width, component_height, _area = stats[component]
    rows, columns = np.indices(foreground.shape)
    central_opening = (
        (labels == component)
        & (columns > left + 0.43 * component_width)
        & (columns < left + 0.57 * component_width)
        & (rows > top + 0.12 * component_height)
        & (rows < top + 0.36 * component_height)
    )
    surrounding_top = (
        (labels == component)
        & (
            (
                (columns > left + 0.15 * component_width)
                & (columns < left + 0.35 * component_width)
            )
            | (
                (columns > left + 0.65 * component_width)
                & (columns < left + 0.85 * component_width)
            )
        )
        & (rows > top + 0.12 * component_height)
        & (rows < top + 0.36 * component_height)
    )
    grayscale = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    assert central_opening.sum() > 100
    assert surrounding_top.sum() > 100
    assert (
        float(np.median(grayscale[surrounding_top]))
        - float(np.median(grayscale[central_opening]))
        > 20
    ), "the exact annulus inner wall was not visibly distinct from its top surface"
    assert len(canonical_mesh_requests) == 1
    assert urlparse(canonical_mesh_requests[0]).query == (
        f"revision={record['canonical_ply_sha256']}"
    )
    page.get_by_role("button", name="Refresh card preview").click()
    expect(page.get_by_text("Recognition preview refreshed")).to_be_visible()
    assert page_errors == []


def test_workpiece_thumbnail_revision_mismatch_is_actionable_and_refresh_recovers(
    console_server, page
) -> None:
    install_common_mocks(page)
    catalogue = pose_template_catalog()
    workpiece = catalogue["objects"][0]
    preview_ready = {"value": False}
    page.route(
        "**/workpieces/status",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "workpiece_catalog_status.v1",
                "available": True,
                "status": "available",
                "reason": None,
                "counts": {"active": 1, "archived": 0, "total": 1},
            },
        ),
    )
    page.route(
        "**/workpieces/catalog",
        lambda route: fulfill_json(route, catalogue),
    )
    page.route(
        "**/pose-templates/status",
        lambda route: fulfill_json(route, pose_template_source(available=True)),
    )
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(
            route,
            {"schema_version": "pose_template_library.v1", "templates": []},
        ),
    )

    def thumbnail_handler(route) -> None:
        if preview_ready["value"]:
            fulfill_json(
                route,
                pose_template_orientation_thumbnail(workpiece["catalog_uuid"]),
            )
            return
        fulfill_json(
            route,
            {
                "output": (
                    "Orientation thumbnail was produced by an unsupported "
                    "implementation revision"
                ),
                "analysis_required": True,
            },
            status=409,
        )

    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        thumbnail_handler,
    )

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")
    failure = page.get_by_test_id(
        f"workpiece-thumbnail-error-{workpiece['catalog_uuid']}"
    )
    expect(failure).to_contain_text(
        "Preview/server revision mismatch. Restart PoseTestBot, then reload."
    )
    expect(failure).to_have_attribute(
        "title",
        "Orientation thumbnail was produced by an unsupported implementation revision",
    )

    page.goto(f"{console_server.url}/#/pose-templates", wait_until="networkidle")
    expect(
        page.get_by_test_id(
            f"workpiece-thumbnail-error-{workpiece['catalog_uuid']}"
        )
    ).to_contain_text("Preview/server revision mismatch")

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")
    expect(
        page.get_by_test_id(
            f"workpiece-thumbnail-error-{workpiece['catalog_uuid']}"
        )
    ).to_be_visible()
    preview_ready["value"] = True
    page.get_by_role("button", name="Refresh workpiece catalogue").click()
    expect(
        page.get_by_test_id(f"workpiece-isometric-{workpiece['catalog_uuid']}")
    ).to_be_visible()
    expect(failure).to_have_count(0)


def test_workpiece_dense_card_uses_lazy_canvas_and_accessible_lod_evidence(
    console_server, page
) -> None:
    install_common_mocks(page)
    page.add_init_script("""
      const originalGetContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(kind, ...args) {
        if (kind === "webgl" || kind === "webgl2") return null;
        return originalGetContext.call(this, kind, ...args);
      };
    """)
    catalogue = pose_template_catalog()
    workpiece = catalogue["objects"][0]
    workpiece["extraction"] = {
        **workpiece["extraction"],
        "vertices": 1_024,
        "faces": 2_048,
    }
    sections = 513
    vertices = [[0, 0, 1.5], *[
        [
            10 * np.cos(2 * np.pi * index / sections),
            7 * np.sin(2 * np.pi * index / sections),
            0.8 * np.sin(6 * np.pi * index / sections),
        ]
        for index in range(sections)
    ]]
    faces = [
        [0, index + 1, (index + 1) % sections + 1]
        for index in range(sections)
    ]
    thumbnail = pose_template_orientation_thumbnail(workpiece["catalog_uuid"])
    thumbnail["preview_mesh"] = {"vertices": vertices, "faces": faces}
    thumbnail["recognition_mesh_approximation"] = {
        "strategy": "spatial_clustering",
        "implementation_revision": "posetestbot_posetemplatecreator_adapter.v4",
        "source_vertices": 1_024,
        "source_faces": 2_048,
        "welded_vertices": 1_024,
        "welded_faces": 2_048,
        "result_vertices": len(vertices),
        "result_faces": len(faces),
        "source_components": 1,
        "source_euler_number": 1,
        "result_components": 1,
        "result_euler_number": 1,
        "topology_preserved": True,
        "spatial_resolution": 32,
        "fallback_reason": None,
    }

    page.route(
        "**/workpieces/status",
        lambda route: fulfill_json(route, {
            "schema_version": "workpiece_catalog_status.v1",
            "available": True,
            "status": "available",
            "reason": None,
            "counts": {"active": 1, "archived": 0, "total": 1},
        }),
    )
    page.route(
        "**/workpieces/catalog",
        lambda route: fulfill_json(route, catalogue),
    )
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(route, thumbnail),
    )

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")

    rendered = page.get_by_test_id(
        f"workpiece-isometric-{workpiece['catalog_uuid']}"
    )
    expect(rendered).to_have_count(1)
    expect(rendered).to_have_js_property("tagName", "CANVAS")
    expect(rendered.locator("polygon")).to_have_count(0)
    detail = (
        "Bounded preview level of detail: 513 of 2,048 source faces shown "
        "as a spatially clustered surface"
    )
    badge = page.get_by_label(detail)
    expect(badge).to_have_text("LOD")
    badge.focus()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "513 of 2,048 source faces shown"
    )
    expect(
        page.get_by_role("button", name="Select Clamp").locator(
            "[aria-label^='Bounded preview level of detail']"
        )
    ).to_have_count(0)


def test_pose_templates_add_instance_with_real_catalog_and_preview(
    console_server, page, tmp_path: Path, monkeypatch
) -> None:
    working = tmp_path / "working"
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setattr(
        pose_template_routes,
        "REQUEST_ROOT",
        working / "jobs" / "pose_template_requests",
    )
    cad = tmp_path / "browser-box.stl"
    cad.write_bytes(trimesh.creation.box(extents=(20, 10, 10)).export(file_type="stl"))
    record = import_catalog_object(
        name="Browser box",
        cad_path=cad,
        catalog_root=working / "object_catalog",
    )

    install_common_mocks(page)
    page.add_init_script("Object.defineProperty(Crypto.prototype, 'randomUUID', { value: undefined, configurable: true })")
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route("**/pose-templates/status", lambda route: fulfill_json(route, pose_template_source(available=True)))
    page.route("**/pose-templates/library", lambda route: fulfill_json(route, {"schema_version": "pose_template_library.v1", "templates": []}))
    page.route("**/pose-templates/workpieces/*/orientations", lambda route: fulfill_json(route, pose_template_orientation_analysis(record["catalog_uuid"])))
    page.route("**/pose-templates/workpieces/*/orientation-thumbnail", lambda route: fulfill_json(route, pose_template_orientation_thumbnail(record["catalog_uuid"])))

    def preview_handler(route) -> None:
        if route.request.method == "POST":
            fulfill_json(route, {"job_id": "preview-job", "request_id": "c" * 32}, status=202)
        else:
            fulfill_json(route, immutable_template_preview())

    page.route("**/pose-templates/preview**", preview_handler)
    page.goto(f"{console_server.url}/#/pose-templates", wait_until="networkidle")

    expect(page.get_by_text("Browser box", exact=True)).to_be_visible()
    page.get_by_role("button", name="Choose orientation for Browser box").click()
    page.get_by_test_id("orientation-chooser").get_by_role("button", name="Add selected orientation").click()
    expect(page.get_by_role("button", name="Select and move Browser box")).to_have_count(1)
    page.get_by_role("button", name="Choose orientation for Browser box").click()
    page.get_by_test_id("orientation-chooser").get_by_role("button", name="Add selected orientation").click()
    expect(page.get_by_role("button", name="Select and move Browser box")).to_have_count(2)
    page.get_by_role("button", name="Remove Browser box instance").click()
    expect(page.get_by_role("button", name="Select and move Browser box")).to_have_count(1)
    expect(page.get_by_role("button", name="Generate immutable version")).to_be_enabled(timeout=15_000)
    assert page_errors == []


def test_ground_truth_workflow_selection_and_full_placement(console_server, page) -> None:
    install_common_mocks(page)
    submitted: list[dict] = []
    exact_asset_requests: list[str] = []
    library_payload = pose_template_library()
    second_instance = {
        "instance_uuid": "55555555-5555-4555-8555-555555555555",
        "catalog_uuid": "88888888-8888-4888-8888-888888888888",
        "catalog": {
            "catalog_uuid": "88888888-8888-4888-8888-888888888888",
            "name": "Gauge block",
            "obj_id": 8,
            "canonical_ply_sha256": "f" * 64,
        },
        "pose_template_from_object": {
            "matrix": [
                [1, 0, 0, 105],
                [0, 1, 0, 75],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        },
    }
    library_payload["templates"][0]["instances"][0]["catalog"][
        "canonical_ply_sha256"
    ] = "b" * 64
    library_payload["templates"][0]["instances"].append(second_instance)
    second_template = json.loads(json.dumps(library_payload["templates"][0]))
    second_template["template_uuid"] = "44444444-4444-4444-8444-444444444444"
    second_template["display_name"] = "Clamp portrait"
    library_payload["templates"].append(second_template)
    preview_payload = immutable_template_preview()
    preview_payload["instances"][0]["catalog"]["canonical_ply_sha256"] = "b" * 64
    preview_payload["instances"][0]["orientation"] = {"label": "Wide base"}
    preview_payload["instances"].append({
        **second_instance,
        "preview_mesh_sha256": "f" * 64,
        "orientation": {"label": "Flat face"},
        "compensated_contours": [[
            {"x_mm": 80, "y_mm": 55},
            {"x_mm": 105, "y_mm": 55},
            {"x_mm": 105, "y_mm": 65},
            {"x_mm": 80, "y_mm": 65},
        ]],
    })
    preview_payload["preview_meshes"]["f" * 64] = {
        "vertices": [
            [-12.5, -5, -2.5],
            [12.5, -5, -2.5],
            [12.5, 5, -2.5],
            [-12.5, 5, -2.5],
            [-12.5, -5, 2.5],
            [12.5, -5, 2.5],
            [12.5, 5, 2.5],
            [-12.5, 5, 2.5],
        ],
        "faces": [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
    }
    page.route("**/pose-templates/library", lambda route: fulfill_json(route, library_payload))
    page.route("**/pose-templates/library/*/preview", lambda route: fulfill_json(route, preview_payload))
    page.route("**/pose-templates/library/*/thumbnail", lambda route: fulfill_json(route, immutable_template_thumbnail()))
    exact_assets = {
        "33333333-3333-4333-8333-333333333333": trimesh.creation.box(
            extents=(20, 10, 12)
        ).export(file_type="ply"),
        "55555555-5555-4555-8555-555555555555": trimesh.creation.box(
            extents=(25, 10, 5)
        ).export(file_type="ply"),
    }

    def exact_asset_handler(route) -> None:
        exact_asset_requests.append(route.request.url)
        instance_uuid = urlparse(route.request.url).path.split("/")[-2]
        route.fulfill(
            status=200,
            content_type="application/octet-stream",
            body=exact_assets[instance_uuid],
        )

    page.route(
        "**/pose-templates/library/*/assets/*/canonical_ply*",
        exact_asset_handler,
    )
    page.route("**/jobs/selection-job", lambda route: fulfill_json(route, {"job": {"id": "selection-job", "status": "succeeded", "message": None, "tail": []}}))

    def selection_handler(route) -> None:
        if route.request.method == "POST":
            submitted.append(route.request.post_data_json)
            fulfill_json(route, {"job_id": "selection-job"}, status=202)
        else:
            fulfill_json(route, {"selection": None, "replacement_blockers": [], "ready": False})
    page.route("**/pose-templates/runs/selection**", selection_handler)
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=template",
        wait_until="networkidle",
    )
    expect(page.get_by_test_id("ground-truth-workflow")).to_be_visible()
    template_thumbnail = page.get_by_test_id(
        "template-thumbnail-22222222-2222-4222-8222-222222222222"
    )
    expect(template_thumbnail).to_be_visible()
    expect(template_thumbnail.locator("svg")).to_have_attribute(
        "data-compensated-origin-mm", "13.050,16.335"
    )
    expect(template_thumbnail.locator("svg g g")).to_have_attribute(
        "transform", "translate(15 15)"
    )
    page.get_by_role("radio", name="Select Clamp pair").click()
    expect(page.get_by_test_id("selected-template-scene")).to_be_visible(timeout=15_000)
    expect(page.get_by_test_id("selected-template-scene")).to_have_attribute("data-origin-offset-mm", "15,15")
    expect(page.get_by_test_id("selected-template-scene").locator("canvas")).to_have_count(1)
    expect(page.get_by_text("Exact immutable PLY detail")).to_be_visible()
    object_index = page.get_by_test_id("selected-template-object-index")
    expect(
        object_index.get_by_role(
            "button", name="Focus Clamp, obj_000007, instance 1"
        )
    ).to_be_visible()
    gauge_focus = object_index.get_by_role(
        "button", name="Focus Gauge block, obj_000008, instance 2"
    )
    expect(gauge_focus).to_be_visible()
    expect(object_index.get_by_text("20 × 10 × 12 mm · 12 faces")).to_be_visible(
        timeout=15_000
    )
    expect(object_index.get_by_text("25 × 10 × 5 mm · 12 faces")).to_be_visible(
        timeout=15_000
    )
    assert len(exact_asset_requests) == 2
    assert all("sha256=" in request for request in exact_asset_requests)
    gauge_focus.click()
    expect(gauge_focus).to_have_attribute("aria-pressed", "true")
    sheet_focus = page.get_by_role("button", name="Fit printed sheet")
    sheet_focus.click()
    expect(sheet_focus).to_have_attribute("aria-pressed", "true")
    confirmation = page.get_by_label("I confirm this measured physical placement")
    confirmation.click()
    expect(confirmation).to_be_checked()
    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(confirmation).not_to_be_checked()
    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="new-run · real_full_capture_validation").click()
    expect(confirmation).not_to_be_checked()
    confirmation.click()
    page.get_by_role("radio", name="Select Clamp portrait").click()
    expect(confirmation).not_to_be_checked()
    page.get_by_role("radio", name="Select Clamp pair").click()
    confirmation.click()
    page.get_by_label("Template placement X mm").fill("12")
    expect(confirmation).not_to_be_checked()
    page.get_by_label("Template placement Z mm").fill("34")
    page.get_by_label("Template placement Yaw °").fill("90")
    confirmation.click()
    page.get_by_role("button", name="Select for run").click()
    expect(page.get_by_text("Object layout selection queued")).to_be_visible()
    assert submitted[0]["confirmed"] is True
    assert submitted[0]["template_uuid"] == "22222222-2222-4222-8222-222222222222"
    assert submitted[0]["placement"]["matrix"][0][3] == 12
    assert submitted[0]["placement"]["matrix"][2][3] == 34


def test_run_config_preflight_blocker_and_fresh_capture_gates(console_server, page) -> None:
    requests: list[dict] = []
    preflight_state = {"blocker": "missing_preflight"}
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
            },
            {
                "sensor_type": "realsense_d435",
                "device_id": "static-1",
                "display_name": "Static RGB-D",
                "mounting_mode": "static",
                "enabled": True,
                "inverted": True,
            },
        ]
    )
    configured["calibration_target"] = {
        "target_id": "5f09f41c-dd91-44ef-a048-1f43fc990e17",
        "placement": {"mode": "stationary_template_base"},
    }
    install_common_mocks(
        page,
        preflight_state=preflight_state,
        requests=requests,
        config_payload=configured,
    )
    page.route("**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status()))
    capture_setup = {"queued": False, "post_queue_reads": 0}

    def calibration_setup_handler(route) -> None:
        cameras = []
        if capture_setup["queued"]:
            capture_setup["post_queue_reads"] += 1
            if capture_setup["post_queue_reads"] >= 2:
                cameras = [
                    {
                        "sensor_key": "realsense_d435:wrist-1",
                        "sensor_name": "realsense_wrist-1",
                        "display_name": "Wrist RGB-D",
                        "sensor_type": "realsense_d435",
                        "device_id": "wrist-1",
                        "current_mounting_mode": "eye_in_hand",
                    }
                ]
        fulfill_json(
            route,
            {
                "schema_version": "calibration_setup.v1",
                "run_root": RUN_ROOT,
                "cameras": cameras,
                "unavailable_cameras": [],
                "saved_targets": [
                    {
                        "target_id": configured["calibration_target"]["target_id"],
                        "display_name": "Lab board",
                        "valid": True,
                        "selected": True,
                    }
                ],
                "modes": [
                    {
                        "id": "eye_in_hand",
                        "label": "Robot-mounted camera (eye-in-hand)",
                        "primary_transform": "camera → robot_flange",
                        "target_mounting": "stationary relative to template_base",
                    },
                    {
                        "id": "eye_to_hand",
                        "label": "Static camera (eye-to-hand)",
                        "primary_transform": "camera → template_base",
                        "target_mounting": "rigidly attached to robot_flange",
                    },
                ],
                "solver": {
                    "default_pnp_methods": ["IPPE", "ITERATIVE", "SQPNP"],
                    "default_extrinsic_methods": ["tsai", "park"],
                    "intrinsics_policy": "compare_factory_opencv",
                    "intrinsics_policies": [],
                    "thresholds": {
                        "min_pnp_common_inliers": 12,
                        "min_pnp_common_inlier_ratio": 0.5,
                        "max_pnp_all_point_mean_reprojection_error_px": 3.0,
                        "min_pnp_supported_markers": 4,
                        "min_pnp_grid_rows": 2,
                        "min_pnp_grid_columns": 2,
                        "min_accepted_views": 15,
                        "min_coverage_cells": 6,
                        "max_per_view_reprojection_error_px": 3.0,
                        "max_intrinsic_rms_reprojection_error_px": 1.5,
                        "min_motion_poses": 4,
                        "min_translation_span_mm": 20.0,
                        "min_rotation_span_deg": 5.0,
                        "min_rotation_axis_second_to_first_ratio": 0.15,
                        "max_nearest_pose_delta_ms": 20.0,
                    },
                },
                "latest_attempt": None,
            },
        )

    def pipeline_sequence_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/pipeline/run-sequence", "body": body})
        capture_setup["queued"] = True
        fulfill_json(
            route,
            {"job_id": f"job-{len(requests)}", "status": "queued"},
            status=202,
        )

    page.route("**/calibration/setup?**", calibration_setup_handler)
    page.route("**/pipeline/run-sequence", pipeline_sequence_handler)
    page.add_init_script(
        "localStorage.setItem('posetestbot.selectedSensors', "
        "JSON.stringify(['realsense_d435:wrist-1', 'realsense_d435:static-1']))"
    )
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    speed = page.locator("#velocity")
    expect(speed).to_have_value("0.03")
    expect(speed).to_have_attribute("max", "0.03")
    expect(page.get_by_text("Full capture is an A1 joint PTP", exact=False)).to_be_visible()
    page.get_by_role("button", name="Save setup").click()
    expect(page.get_by_text("Calibration recording setup saved")).to_be_visible()
    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    assert written["plan_only"] is True
    assert written["velocity"] == 0.03
    assert "mounting_mode" not in written
    assert [sensor["mounting_mode"] for sensor in written["sensors"]] == [
        "eye_in_hand",
        "static",
    ]
    assert "allow_cameras" not in json.dumps(written)
    assert "allow_real_robot" not in json.dumps(written)

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=readiness",
        wait_until="networkidle",
    )
    readiness = page.get_by_test_id("calibration-readiness-check")
    expect(readiness).to_have_count(1)
    expect(readiness).to_be_visible()
    expect(readiness).to_contain_text("Readiness has not been checked")
    readiness.get_by_role("button", name="Check readiness", exact=True).click()
    preflight_request = next(item["body"] for item in requests if item["path"] == "/pipeline/run")
    assert preflight_request["stage"] == "run_preflight"
    assert "allow_cameras" not in json.dumps(preflight_request)

    preflight_state["blocker"] = None
    page.reload(wait_until="networkidle")
    page.get_by_role("button", name="Review and start capture", exact=True).click()
    expect(page.get_by_test_id("capture-timeout-envelope")).to_contain_text(
        "720 s total · 15 s sustained camera readiness (3 frames each) · 120 s to first robot packet · 60 s between robot packets"
    )
    submit = page.locator('[data-testid="capture-submit"]')
    expect(submit).to_be_disabled()
    page.locator('[data-testid="capture-robot-ack"]').click()
    expect(submit).to_be_disabled()
    page.locator('[data-testid="capture-camera-ack"]').click()
    expect(submit).to_be_enabled()
    submit.click()
    expect(page.get_by_text("Calibration capture queued")).to_be_visible()
    capture_request = [
        item["body"]
        for item in requests
        if item["path"] == "/pipeline/run-sequence"
    ][-1]
    assert capture_request == {
        "sequence": "real_full_capture_validation",
        "run_root": RUN_ROOT,
        "plan_only": False,
        "options": {
            "capture_plan_preflight": {"allow_real_robot": True},
            "capture_execution_plan": {
                "allow_cameras": True,
                "allow_real_robot": True,
                "include_sensors": True,
            },
            "capture_execution": {
                "allow_cameras": True,
                "allow_real_robot": True,
                "include_sensors": True,
                "timeout_s": 720,
                "startup_wait_s": 15,
                "receive_start_timeout_s": 120,
                "receive_idle_timeout_s": 60,
            },
        },
    }
    assert any(item["path"] == "/sensors/previews/stop" for item in requests)
    expect(page.locator('input[value="eye_in_hand"]')).to_be_checked(timeout=6_000)
    expect(
        page.get_by_test_id("calibration-workflow").get_by_text(
            "Wrist RGB-D", exact=True
        )
    ).to_be_visible()
    assert capture_setup["post_queue_reads"] >= 2


def test_readiness_background_refresh_preserves_visible_evidence(
    console_server,
    page,
) -> None:
    preflight_state = {"blocker": "missing_preflight"}
    configured = run_config()
    install_common_mocks(
        page,
        preflight_state=preflight_state,
        config_payload=configured,
    )
    config_reads = {"count": 0}

    def delayed_config_handler(route) -> None:
        config_reads["count"] += 1
        if config_reads["count"] > 1:
            threading.Event().wait(0.3)
        fulfill_json(
            route,
            {
                "config": configured,
                "preflight": {"queue_blocker": preflight_state["blocker"]},
            },
        )

    page.unroute("**/run-config**")
    page.route("**/run-config**", delayed_config_handler)
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=readiness",
        wait_until="networkidle",
    )

    readiness = page.get_by_test_id("calibration-readiness-check")
    expect(readiness).to_contain_text("Readiness has not been checked")
    expect(
        readiness.get_by_role("button", name="Check readiness", exact=True)
    ).to_be_enabled()
    readiness.evaluate(
        """card => {
            window.__readinessTextSnapshots = []
            const record = () => window.__readinessTextSnapshots.push(card.innerText)
            record()
            window.__readinessTextObserver = new MutationObserver(record)
            window.__readinessTextObserver.observe(card, {
                attributes: true,
                characterData: true,
                childList: true,
                subtree: true,
            })
        }"""
    )

    with page.expect_request(
        lambda request: (
            request.method == "GET"
            and urlparse(request.url).path == "/run-config"
        ),
        timeout=5_000,
    ):
        page.wait_for_timeout(2_300)
    page.wait_for_timeout(500)

    snapshots = page.evaluate(
        """() => {
            window.__readinessTextObserver.disconnect()
            return window.__readinessTextSnapshots
        }"""
    )
    assert config_reads["count"] >= 2
    assert all("Checking saved run readiness" not in text for text in snapshots)
    assert all(
        "Reading the latest durable readiness evidence" not in text
        for text in snapshots
    )
    expect(readiness).to_contain_text("Readiness has not been checked")
    expect(
        readiness.get_by_role("button", name="Check readiness", exact=True)
    ).to_be_enabled()


def test_dataset_setup_requires_and_snapshots_a_prior_calibration(
    console_server,
    page,
) -> None:
    requests: list[dict] = []
    selection_requests: list[dict] = []
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
            },
            {
                "sensor_type": "realsense_d435",
                "device_id": "static-1",
                "display_name": "Static RGB-D",
                "mounting_mode": "static",
                "enabled": True,
                "inverted": True,
            },
        ]
    )
    configured["schema_version"] = "run_config.v3"
    configured["capture"]["synchronization"] = {
        "schema_version": "capture_synchronization.v1",
        "mode": "hardware_trigger",
        "implementation": "realsense_inter_cam_sync",
        "scope": "depth_exposure",
        "group_id": "existing-mixed-rig",
        "master_sensor_key": "realsense_d435:wrist-1",
        "max_depth_timestamp_skew_ms": 2,
    }
    source_run_root = "/tmp/posetestbot-console/calibration-july-21"
    source_bundle_sha256 = "a" * 64
    selected_calibration_path = (
        "processed/calibration_selection/calibration_profiles.json"
    )
    selected_intrinsics_path = (
        "processed/calibration_selection/intrinsic_calibration_profiles.json"
    )
    selection_artifact = calibration_selection_artifact(
        bundle_sha256=source_bundle_sha256,
        calibration_profiles=selected_calibration_path,
        intrinsic_calibration_profiles=selected_intrinsics_path,
        source_run_root=source_run_root,
        source_run_name="Calibration run July 21",
    )
    source = {
        "source_run_root": source_run_root,
        "source_run_name": "Calibration run July 21",
        "bundle_sha256": source_bundle_sha256,
        "valid": True,
        "compatible": True,
        "issues": [],
        "calibration_profiles": {
            "sha256": "b" * 64,
            "valid_profile_count": 2,
            "profiles": [
                {
                    "profile_id": "profile-wrist-1",
                    "sensor_type": "realsense_d435",
                    "sensor_id": "wrist-1",
                    "mounting_mode": "eye_in_hand",
                    "method": "IPPE + park",
                    "quality": {
                        "num_observations": 18,
                        "num_inliers": 17,
                        "mean_reprojection_error_px": 0.31,
                    },
                },
                {
                    "profile_id": "profile-static-1",
                    "sensor_type": "realsense_d435",
                    "sensor_id": "static-1",
                    "mounting_mode": "static",
                    "method": "IPPE + park",
                    "quality": {
                        "num_observations": 18,
                        "num_inliers": 17,
                        "mean_reprojection_error_px": 0.29,
                    },
                },
            ],
        },
        "intrinsic_calibration_profiles": {
            "sha256": "c" * 64,
            "profile_count": 2,
        },
    }
    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route("**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status()))
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {"selected": None, "calibrations": [source]},
        ),
    )

    def select_calibration_handler(route) -> None:
        selection_requests.append(route.request.post_data_json)
        fulfill_json(
            route,
            {
                "calibration_profiles": selected_calibration_path,
                "intrinsic_calibration_profiles": selected_intrinsics_path,
                "sensor_profile_mapping": [
                    {
                        "sensor_key": "realsense_d435:wrist-1",
                        "profile_id": "profile-wrist-1",
                    },
                    {
                        "sensor_key": "realsense_d435:static-1",
                        "profile_id": "profile-static-1",
                    },
                ],
                "sensor_profiles": {
                    "realsense_d435:wrist-1": "profile-wrist-1",
                    "realsense_d435:static-1": "profile-static-1",
                },
                "selection": selection_artifact,
            },
        )

    page.route("**/ui/calibrations/select", select_calibration_handler)
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(route, {"templates": []}),
    )
    page.route(
        "**/pose-templates/runs/selection?**",
        lambda route: fulfill_json(
            route,
            {"selection": None, "replacement_blockers": [], "ready": False},
        ),
    )

    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=configure",
        wait_until="networkidle",
    )

    expect(page.get_by_role("heading", name="Record an object-template dataset")).to_be_visible()
    expect(page.get_by_role("heading", name="Saved camera calibration")).to_contain_text(
        "Required"
    )
    expect(page.get_by_text("Required: select and validate a previously published calibration below.")).to_be_visible()
    save_setup = page.get_by_role("button", name="Save setup")
    expect(save_setup).to_be_disabled()
    readiness_action = page.get_by_role(
        "button", name="Check readiness", exact=True
    )
    expect(readiness_action).to_have_count(1)
    expect(readiness_action).to_be_visible()

    source_choice = page.get_by_role("radio").filter(
        has_text="Calibration run July 21"
    )
    expect(source_choice).to_have_count(1)
    expect(source_choice).to_contain_text("Camera settings match")
    source_choice.click()
    expect(source_choice).to_have_attribute("aria-checked", "true")
    synchronization_mode = page.get_by_role(
        "combobox", name="Synchronization mode"
    )
    expect(synchronization_mode).to_contain_text(
        "Hardware-triggered RealSense depth exposure"
    )
    synchronization_mode.click()
    page.get_by_role("option", name="Timestamp-aligned RGB-D streams").click()
    expect(page.get_by_text("Timestamp association", exact=True)).to_be_visible()
    synchronization_mode.click()
    page.get_by_role(
        "option", name="Hardware-triggered RealSense depth exposure"
    ).click()
    expect(page.get_by_text("Depth-only hardware synchronization boundary")).to_be_visible()
    expect(page.get_by_text("not certified as hardware-synchronous across cameras")).to_be_visible()
    expect(page.get_by_test_id("hardware-sync-contract-status")).to_contain_text(
        "Hardware trigger configuration is complete"
    )
    expect(page.get_by_test_id("hardware-sync-qualification-requirement")).to_contain_text(
        "Current physical qualification required before recording"
    )
    expect(page.get_by_test_id("hardware-sync-qualification-requirement")).to_contain_text(
        "hardware_sync_qualification.json"
    )
    page.get_by_label("Trigger group ID").fill("research-mixed-rig")
    validate_and_save = page.get_by_role(
        "button", name="Validate and save setup", exact=True
    )
    expect(validate_and_save).to_be_enabled()
    validate_and_save.click()

    expect(page.get_by_text("Object dataset setup saved")).to_be_visible()
    assert selection_requests == [
        {
            "run_root": RUN_ROOT,
            "source_run_root": source_run_root,
            "expected_bundle_sha256": source_bundle_sha256,
            "expected_current_bundle_sha256": None,
            "confirm_replace": False,
            "resolution": "720p",
            "sensors": [
                {
                    "sensor_type": "realsense_d435",
                    "device_id": "wrist-1",
                    "mounting_mode": "eye_in_hand",
                    "inverted": False,
                },
                {
                    "sensor_type": "realsense_d435",
                    "device_id": "static-1",
                    "mounting_mode": "static",
                    "inverted": True,
                },
            ],
        }
    ]
    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    assert written["dataset_mode"] == "pose_template"
    assert written["plan_only"] is False
    assert written["sequence"] == "calibrated_capture_to_bop_dataset_dry_run"
    assert written["calibration_profiles"] == selected_calibration_path
    assert written["expected_calibration_bundle_sha256"] == source_bundle_sha256
    assert written["sequence_options"] == {
        "camera_rectification": {"intrinsic_profiles": selected_intrinsics_path}
    }
    assert written["sensors"][0]["calibration_profile_id"] == "profile-wrist-1"
    assert written["sensors"][1]["calibration_profile_id"] == "profile-static-1"
    assert written["synchronization"] == {
        "schema_version": "capture_synchronization.v1",
        "mode": "hardware_trigger",
        "implementation": "realsense_inter_cam_sync",
        "scope": "depth_exposure",
        "group_id": "research-mixed-rig",
        "master_sensor_key": "realsense_d435:wrist-1",
        "max_depth_timestamp_skew_ms": 2,
    }

    readiness_steps = page.locator('[data-workflow-step="readiness"]')
    expect(readiness_steps).to_have_count(1)
    expect(readiness_steps).to_be_visible()
    expect(page.get_by_test_id("dataset-readiness-check")).to_have_count(1)
    expect(page.get_by_test_id("dataset-readiness-check")).to_contain_text(
        "One readiness check"
    )


def test_dataset_setup_requires_confirmation_to_replace_selected_calibration(
    console_server,
    page,
) -> None:
    requests: list[dict] = []
    selection_requests: list[dict] = []
    current_bundle_sha256 = "1" * 64
    replacement_bundle_sha256 = "2" * 64
    current_calibration_path = (
        "processed/calibration_inputs/current/calibration_profiles.json"
    )
    current_intrinsics_path = (
        "processed/calibration_inputs/current/intrinsic_calibration_profiles.json"
    )
    replacement_calibration_path = (
        "processed/calibration_inputs/replacement/calibration_profiles.json"
    )
    replacement_intrinsics_path = (
        "processed/calibration_inputs/replacement/intrinsic_calibration_profiles.json"
    )
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
                "calibration_profile_id": "profile-wrist-1",
            }
        ]
    )
    configured["dataset_mode"] = "pose_template"
    configured["calibration_profiles"] = current_calibration_path
    configured["intrinsic_calibration_profiles"] = current_intrinsics_path
    configured["calibration_profile_selection"] = {
        "selection_artifact": "calibration_profile_selection.json",
        "bundle_sha256": current_bundle_sha256,
        "selected_at": "2026-07-22T12:00:00+00:00",
    }
    replacement_source_root = "/tmp/posetestbot-console/replacement-calibration"
    replacement_source = {
        "source_run_root": replacement_source_root,
        "source_run_name": "Replacement calibration",
        "bundle_sha256": replacement_bundle_sha256,
        "valid": True,
        "compatible": False,
        "issues": [
            {
                "code": "saved_setup_changed",
                "message": "The saved setup differs; current choices will be checked when saved.",
            }
        ],
        "calibration_profiles": {
            "sha256": "4" * 64,
            "valid_profile_count": 1,
            "profiles": [],
        },
        "intrinsic_calibration_profiles": {
            "sha256": "5" * 64,
            "profile_count": 1,
        },
    }
    replacement_selection = calibration_selection_artifact(
        bundle_sha256=replacement_bundle_sha256,
        calibration_profiles=replacement_calibration_path,
        intrinsic_calibration_profiles=replacement_intrinsics_path,
        source_run_root=replacement_source_root,
        source_run_name="Replacement calibration",
    )

    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route("**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status()))
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {
                "selected": valid_library_selection(
                    bundle_sha256=current_bundle_sha256,
                    calibration_profiles=current_calibration_path,
                    intrinsic_calibration_profiles=current_intrinsics_path,
                ),
                "calibrations": [replacement_source],
            },
        ),
    )

    def select_calibration_handler(route) -> None:
        selection_requests.append(route.request.post_data_json)
        fulfill_json(
            route,
            {
                "calibration_profiles": replacement_calibration_path,
                "intrinsic_calibration_profiles": replacement_intrinsics_path,
                "sensor_profile_mapping": [
                    {
                        "sensor_key": "realsense_d435:wrist-1",
                        "profile_id": "profile-wrist-1",
                    }
                ],
                "selection": replacement_selection,
            },
        )

    page.route("**/ui/calibrations/select", select_calibration_handler)
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(route, {"templates": []}),
    )
    page.route(
        "**/pose-templates/runs/selection?**",
        lambda route: fulfill_json(
            route,
            {"selection": None, "replacement_blockers": [], "ready": False},
        ),
    )
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=configure",
        wait_until="networkidle",
    )

    expect(page.get_by_text("A verified calibration snapshot is selected")).to_be_visible()
    replacement_choice = page.get_by_role("radio").filter(
        has_text="Replacement calibration"
    )
    expect(replacement_choice).to_contain_text("Review camera settings")
    expect(replacement_choice).to_contain_text(
        "The saved setup differs; current choices will be checked when saved."
    )
    replacement_choice.click()

    validate_and_save = page.get_by_role(
        "button", name="Validate and save setup", exact=True
    )
    expect(validate_and_save).to_be_disabled()
    assert selection_requests == []
    confirmation = page.get_by_label(
        "Confirm replacing the current calibration selection"
    )
    expect(confirmation).not_to_be_checked()
    confirmation.click()
    expect(validate_and_save).to_be_enabled()
    validate_and_save.click()

    expect(page.get_by_text("Object dataset setup saved")).to_be_visible()
    assert selection_requests == [
        {
            "run_root": RUN_ROOT,
            "source_run_root": replacement_source_root,
            "expected_bundle_sha256": replacement_bundle_sha256,
            "expected_current_bundle_sha256": current_bundle_sha256,
            "confirm_replace": True,
            "resolution": "720p",
            "sensors": [
                {
                    "sensor_type": "realsense_d435",
                    "device_id": "wrist-1",
                    "mounting_mode": "eye_in_hand",
                    "inverted": False,
                }
            ],
        }
    ]
    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    assert written["calibration_profiles"] == replacement_calibration_path
    assert (
        written["expected_calibration_bundle_sha256"]
        == replacement_bundle_sha256
    )


def test_dataset_workflow_blocks_an_invalid_saved_timing_contract(
    console_server,
    page,
) -> None:
    selected_bundle_sha256 = "9" * 64
    selected_calibration_path = (
        "processed/calibration_inputs/current/calibration_profiles.json"
    )
    selected_intrinsics_path = (
        "processed/calibration_inputs/current/intrinsic_calibration_profiles.json"
    )
    configured = run_config(
        plan_only=False,
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
                "calibration_profile_id": "profile-wrist-1",
            }
        ],
    )
    configured["dataset_mode"] = "pose_template"
    configured["calibration_profiles"] = selected_calibration_path
    configured["intrinsic_calibration_profiles"] = selected_intrinsics_path
    configured["calibration_profile_selection"] = {
        "selection_artifact": "calibration_profile_selection.json",
        "bundle_sha256": selected_bundle_sha256,
        "selected_at": "2026-07-22T12:00:00+00:00",
    }
    overview = overview_payload(configured)
    overview["calibration_sync"] = {
        "status": "error",
        "bundle_sha256": selected_bundle_sha256,
        "sensors": [],
        "error": (
            "Profile profile-wrist-1 has no verified robot pose time offset."
        ),
    }

    install_common_mocks(page, config_payload=configured)
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview))
    page.route(
        "**/sensors/status",
        lambda route: fulfill_json(route, selected_sensor_status()),
    )
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {
                "selected": valid_library_selection(
                    bundle_sha256=selected_bundle_sha256,
                    calibration_profiles=selected_calibration_path,
                    intrinsic_calibration_profiles=selected_intrinsics_path,
                ),
                "calibrations": [],
            },
        ),
    )
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(route, {"templates": []}),
    )
    page.route(
        "**/pose-templates/runs/selection?**",
        lambda route: fulfill_json(
            route,
            {"selection": None, "replacement_blockers": [], "ready": False},
        ),
    )

    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=configure",
        wait_until="networkidle",
    )

    timing_policy = page.get_by_test_id("calibration-sync-policy")
    expect(timing_policy).to_contain_text("Automatic calibration timing")
    expect(timing_policy).to_contain_text(
        "This dataset cannot use the selected calibration"
    )
    expect(timing_policy.get_by_role("alert")).to_contain_text(
        "Profile profile-wrist-1 has no verified robot pose time offset."
    )
    readiness = page.get_by_test_id("dataset-readiness-check")
    expect(readiness).to_contain_text(
        "Calibration geometry and automatic timing verified"
    )
    expect(readiness).to_contain_text(
        "The selected calibration timing contract is invalid"
    )
    expect(page.get_by_test_id("dataset-sync-timing-contract")).to_contain_text(
        "Return to Step 1 and select a calibration with valid timing"
    )


def test_dataset_processing_is_one_ordered_operator_action(
    console_server,
    page,
) -> None:
    processing_requests: list[dict] = []
    selected_bundle_sha256 = "d" * 64
    selected_calibration_path = (
        "processed/calibration_inputs/current/calibration_profiles.json"
    )
    selected_intrinsics_path = (
        "processed/calibration_inputs/current/intrinsic_calibration_profiles.json"
    )
    configured = run_config(
        plan_only=False,
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
                "calibration_profile_id": "profile-wrist-1",
            }
        ],
    )
    configured["dataset_mode"] = "pose_template"
    configured["calibration_profiles"] = selected_calibration_path
    configured["intrinsic_calibration_profiles"] = selected_intrinsics_path
    configured["calibration_profile_selection"] = {
        "selection_artifact": "calibration_profile_selection.json",
        "bundle_sha256": selected_bundle_sha256,
        "selected_at": "2026-07-22T12:00:00+00:00",
    }
    configured["pose_template"] = {
        "template_uuid": "22222222-2222-4222-8222-222222222222",
        "placement_confirmed": True,
    }
    overview = overview_payload(configured)
    capture_section = next(
        section for section in overview["sidebar"] if section["id"] == "capture"
    )
    capture_section["artifacts"] = [
        {
            "path": "capture_execution_report.json",
            "exists": True,
            "status": "complete",
        }
    ]

    install_common_mocks(page, config_payload=configured)
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview))
    page.route("**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status()))
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {
                "selected": valid_library_selection(
                    bundle_sha256=selected_bundle_sha256,
                    calibration_profiles=selected_calibration_path,
                    intrinsic_calibration_profiles=selected_intrinsics_path,
                ),
                "calibrations": [],
            },
        ),
    )
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(route, {"templates": []}),
    )
    page.route(
        "**/pose-templates/runs/selection?**",
        lambda route: fulfill_json(
            route,
            {"selection": None, "replacement_blockers": [], "ready": False},
        ),
    )

    def processing_handler(route) -> None:
        processing_requests.append(route.request.post_data_json)
        next(section for section in overview["sidebar"] if section["id"] == "sync")[
            "artifacts"
        ] = [
            {
                "path": "sync_quality_report.json",
                "exists": True,
                "status": "ok",
            }
        ]
        next(section for section in overview["sidebar"] if section["id"] == "bop")[
            "artifacts"
        ] = [
            {
                "path": "camera_rectification_report.json",
                "exists": True,
                "status": "complete",
            },
            {
                "path": "bop/bop_export_manifest.json",
                "exists": True,
                "status": "complete",
            },
        ]
        fulfill_json(
            route,
            {"job_id": "dataset-processing-1", "status": "queued"},
            status=202,
        )

    page.route("**/pipeline/run-config", processing_handler)
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=sync",
        wait_until="networkidle",
    )

    timing_policy = page.get_by_test_id("calibration-sync-policy")
    expect(timing_policy).to_contain_text("Automatic calibration timing")
    expect(timing_policy).to_contain_text("ready")
    expect(timing_policy).to_contain_text("Wrist RGB-D")
    expect(timing_policy).to_contain_text("profile-wrist-1")
    expect(timing_policy).to_contain_text("+70.000 ms")
    expect(timing_policy).to_contain_text("sensor")
    expect(timing_policy).to_contain_text("host_wall")
    expect(timing_policy).to_contain_text("domain global_time")
    expect(timing_policy).to_contain_text("fallback forbidden")
    expect(timing_policy).to_contain_text("20.000 ms")
    page.mouse.move(0, 0)
    timing_policy.get_by_role(
        "button", name="About robot-pose time offset"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Positive means pair the frame with a robot pose recorded later"
    )
    page.keyboard.press("Escape")
    timing_policy.get_by_role(
        "button", name="About calibration timestamp pair"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "exact frame and robot clock fields"
    )
    page.keyboard.press("Escape")
    timing_policy.get_by_role(
        "button", name="About maximum robot-pose gap"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "frame is excluded when its nearest robot pose is farther away"
    )
    page.keyboard.press("Escape")
    timing_contract = page.get_by_test_id("dataset-sync-timing-contract")
    expect(timing_contract).to_contain_text(
        "selected per-camera timing policy will be applied and verified automatically"
    )
    page.mouse.move(0, 0)
    timing_contract.get_by_role(
        "button", name="About automatic calibration timing"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Manual values and generic defaults cannot override them"
    )

    processing = page.get_by_test_id("dataset-processing")
    expect(processing).to_have_count(1)
    expect(processing).to_contain_text("One queued job runs the required derived-data stages in order")
    expect(processing).to_contain_text(
        "Calibration validation is automatic here; there is no second operator preflight."
    )
    process_action = page.get_by_role(
        "button", name="Process and export dataset", exact=True
    )
    expect(process_action).to_have_count(1)
    expect(process_action).to_be_enabled()
    for stale_action in (
        "Synchronize frames",
        "Verify synchronization",
        "Validate selected calibration",
        "Export BOP dataset",
    ):
        expect(
            page.get_by_role("button", name=stale_action, exact=True)
        ).to_have_count(0)

    export_outcome = page.locator('[data-workflow-step="export"]')
    expect(export_outcome).to_contain_text("BOP export has not completed")
    expect(export_outcome).to_contain_text(
        "Use Process and export dataset in step 5"
    )

    process_action.click()
    expect(page.get_by_text("Dataset processing queued")).to_be_visible()
    assert processing_requests == [{"run_root": RUN_ROOT}]
    expect(page.get_by_text("BOP dataset is ready")).to_be_visible(timeout=5_000)


def test_run_setup_disables_camera_without_deleting_identity_or_profile(
    console_server,
    page,
) -> None:
    requests: list[dict] = []
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
                "calibration_profile_id": "profile-wrist-1",
            },
            {
                "sensor_type": "realsense_d435",
                "device_id": "static-1",
                "display_name": "Static RGB-D",
                "mounting_mode": "static",
                "enabled": True,
                "inverted": True,
                "calibration_profile_id": "profile-static-1",
            },
            {
                "sensor_type": "realsense_d435",
                "device_id": "offline-1",
                "display_name": "Offline wrist camera",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
                "calibration_profile_id": "profile-offline-1",
            },
        ]
    )
    status = selected_sensor_status()
    status["families"][0]["devices"].append(
        {
            "sensor_type": "realsense_d435",
            "device_id": "offline-1",
            "display_name": "Offline wrist camera",
            "effective_display_name": "Offline wrist camera",
            "connected": True,
            "capture_ready": False,
            "capture_readiness_reason": "USB SuperSpeed unavailable",
            "mounting_mode": "eye_in_hand",
            "inverted": False,
        }
    )
    status["total_connected"] = 3
    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route("**/sensors/status", lambda route: fulfill_json(route, status))

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    rows = page.locator('[data-testid="run-camera-row"]')
    expect(rows).to_have_count(3)
    offline = page.locator(
        '[data-testid="run-camera-row"][data-sensor-key="realsense_d435:offline-1"]'
    )
    expect(offline).to_have_attribute("data-camera-state", "enabled")
    expect(offline).to_contain_text("not ready")
    page.get_by_label("Enable Offline wrist camera for this run").click()
    expect(offline).to_have_attribute("data-camera-state", "disabled")
    expect(offline).to_have_css("opacity", "0.6")

    page.get_by_role("button", name="Save setup").click()
    expect(page.get_by_text("Calibration recording setup saved")).to_be_visible()
    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    assert len(written["sensors"]) == 3
    disabled = next(
        sensor for sensor in written["sensors"] if sensor["device_id"] == "offline-1"
    )
    assert disabled["enabled"] is False
    assert disabled["calibration_profile_id"] == "profile-offline-1"
    assert {
        sensor["device_id"] for sensor in written["sensors"] if sensor["enabled"]
    } == {"wrist-1", "static-1"}


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
    expect(page.get_by_role("dialog")).to_contain_text(
        "Manual test request: 0.1 m/s (100 mm/s)"
    )
    expect(page.get_by_role("button", name="Queue start")).to_be_disabled()
    expect(page.get_by_role("dialog").get_by_role("checkbox")).to_have_count(1)
    page.get_by_text("I confirm this is the intended lab IIWA target.").click()
    expect(page.get_by_text("I authorize motion of the real lab IIWA for this start.")).to_be_visible()
    expect(page.get_by_text("I confirm the capture cameras and pose receiver are ready.")).to_be_visible()
    expect(page.get_by_role("button", name="Queue start")).to_be_enabled()
    page.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    page.get_by_role("button", name="Stop IIWA").click()
    stop_warning = page.get_by_test_id("iiwa-stop-warning")
    expect(stop_warning).to_contain_text("IIWA STOP is not a safety stop")
    expect(stop_warning).to_contain_text("cannot interrupt active motion")
    expect(stop_warning).to_contain_text("Sunrise must be restarted manually before another START")
    expect(page.get_by_role("button", name="Queue stop")).to_be_disabled()
    expect(page.get_by_role("dialog").get_by_role("checkbox")).to_have_count(1)
    assert [item["command"] for item in commands] == ["start_iiwa"]
    page.get_by_text("I confirm this is the intended lab IIWA target.").click()
    page.get_by_role("button", name="Queue stop").click()
    expect(page.get_by_text("IIWA stop queued")).to_be_visible()

    assert commands == [
        {
            "command": "start_iiwa",
            "robot_ip": "172.31.1.200",
            "robot_port": 30301,
            "allow_real_robot": True,
            "allow_cameras": True,
        },
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
    expect(dialog).to_contain_text("Manual test request: 0.1 m/s (100 mm/s)")
    expect(dialog.get_by_role("button", name="Queue start")).to_be_disabled()
    expect(dialog.get_by_role("checkbox")).to_have_count(1)
    dialog.get_by_text("I confirm this is the intended lab IIWA target.").click()
    expect(dialog.get_by_text("I authorize motion of the real lab IIWA for this start.")).to_be_visible()
    expect(dialog.get_by_text("I confirm the capture cameras and pose receiver are ready.")).to_be_visible()
    expect(dialog.get_by_role("button", name="Queue start")).to_be_enabled()
    dialog.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    controls.get_by_role("button", name="Stop IIWA").click()
    stop_warning = dialog.get_by_test_id("iiwa-stop-warning")
    expect(stop_warning).to_contain_text("IIWA STOP is not a safety stop")
    expect(stop_warning).to_contain_text("cannot interrupt active motion")
    expect(stop_warning).to_contain_text("Sunrise must be restarted manually before another START")
    expect(dialog.get_by_role("button", name="Queue stop")).to_be_disabled()
    expect(dialog.get_by_role("checkbox")).to_have_count(1)
    dialog.get_by_text("I confirm this is the intended lab IIWA target.").click()
    dialog.get_by_role("button", name="Queue stop").click()
    expect(page.get_by_text("IIWA stop queued")).to_be_visible()

    assert commands == [
        {
            "command": "start_iiwa",
            "robot_ip": "172.31.1.147",
            "robot_port": 30300,
            "allow_real_robot": True,
            "allow_cameras": True,
        },
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


def test_jobs_filters_and_progressively_reveals_history(console_server, page) -> None:
    install_common_mocks(page)
    history = []
    for index in range(25):
        failed = index == 24
        history.append({
            "id": f"history-{index:02d}",
            "name": "failed_calibration" if failed else f"completed_job_{index:02d}",
            "command": ["uv"],
            "cwd": "/repo",
            "status": "failed" if failed else "succeeded",
            "created_at": f"2026-07-{index + 1:02d}T12:00:00Z",
            "log_path": f"/tmp/history-{index:02d}.log",
            "started_at": f"2026-07-{index + 1:02d}T12:00:01Z",
            "ended_at": f"2026-07-{index + 1:02d}T12:00:02Z",
            "returncode": 1 if failed else 0,
            "message": "solver evidence failed" if failed else "complete",
            "tail": [],
            "resources": ["cpu"],
            "parameters": {"run_root": RUN_ROOT},
        })
    page.route(
        "**/jobs",
        lambda route: fulfill_json(route, {"jobs": history, "resources": {}}),
    )

    page.goto(f"{console_server.url}/#/jobs", wait_until="networkidle")

    expect(page.get_by_role("button", name="Log")).to_have_count(20)
    page.get_by_role("button", name="Show 5 older jobs").click()
    expect(page.get_by_role("button", name="Log")).to_have_count(25)
    page.get_by_role("combobox", name="Filter jobs by status").click()
    page.get_by_role("option", name="Failed (1)").click()
    expect(page.get_by_text("failed_calibration", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Log")).to_have_count(1)
    page.get_by_label("Search jobs").fill("no such job")
    expect(page.get_by_role("heading", name="No matching jobs")).to_be_visible()
    page.get_by_role("button", name="Clear filters").click()
    expect(page.get_by_role("button", name="Log")).to_have_count(20)


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


def test_calibration_workflow_explains_intrinsics_and_saves_complete_bundle(
    console_server, page
) -> None:
    requests: list[dict] = []
    promoted = {"value": False}
    install_common_mocks(page)
    setup = {
        "schema_version": "calibration_setup.v1",
        "run_root": RUN_ROOT,
        "cameras": [
            {"sensor_key": "realsense_d435:wrist-1", "sensor_name": "realsense_wrist-1", "display_name": "Wrist RGB-D", "sensor_type": "realsense_d435", "device_id": "wrist-1", "current_mounting_mode": "eye_in_hand"},
            {"sensor_key": "oak_d_pro:static-1", "sensor_name": "luxonis_static-1", "display_name": "Auxiliary OAK-D", "sensor_type": "oak_d_pro", "device_id": "static-1", "current_mounting_mode": "eye_in_hand"},
        ],
        "unavailable_cameras": [],
        "saved_targets": [
            {"target_id": "5f09f41c-dd91-44ef-a048-1f43fc990e17", "display_name": "Lab board", "valid": True, "selected": True},
            {"target_id": "9ab5ff1c-60f6-46b1-823d-2a912d5d4e3f", "display_name": "Alternate board", "valid": True},
        ],
        "modes": [
            {"id": "eye_in_hand", "label": "Robot-mounted camera (eye-in-hand)", "primary_transform": "camera → robot_flange", "target_mounting": "stationary relative to template_base"},
            {"id": "eye_to_hand", "label": "Static camera (eye-to-hand)", "primary_transform": "camera → template_base", "target_mounting": "rigidly attached to robot_flange"},
        ],
        "solver": {
            "default_pnp_methods": ["IPPE", "ITERATIVE", "SQPNP"],
            "default_extrinsic_methods": ["tsai", "park", "horaud", "andreff", "daniilidis", "shah", "li"],
            "intrinsics_policy": "compare_factory_opencv",
            "intrinsics_policies": [
                {"id": "compare_factory_opencv", "label": "Compare captured factory intrinsics with a gated OpenCV calibration"},
                {"id": "reuse_compatible_or_factory", "label": "Reuse an exact compatible profile, otherwise captured factory intrinsics"},
            ],
            "synchronization": {
                "default_policy": "auto_offset",
                "policies": [
                    {"id": "auto_offset", "label": "Auto-estimate robot-pose offset — recommended", "description": "Estimate effective per-camera latency."},
                    {"id": "fixed_zero", "label": "Use captured timestamps (0 ms)", "description": "Use the recorded pairing."},
                ],
                "search": {
                    "minimum_robot_pose_time_offset_ms": -150.0,
                    "maximum_robot_pose_time_offset_ms": 150.0,
                    "step_ms": 5.0,
                },
            },
            "thresholds": {
                "min_pnp_common_inliers": 12,
                "min_pnp_common_inlier_ratio": 0.5,
                "max_pnp_all_point_mean_reprojection_error_px": 3.0,
                "min_pnp_supported_markers": 4,
                "min_pnp_grid_rows": 2,
                "min_pnp_grid_columns": 2,
                "min_accepted_views": 15,
                "min_coverage_cells": 6,
                "max_per_view_reprojection_error_px": 3.0,
                "max_intrinsic_rms_reprojection_error_px": 1.5,
                "min_motion_poses": 4,
                "min_translation_span_mm": 20.0,
                "min_rotation_span_deg": 5.0,
                "min_rotation_axis_second_to_first_ratio": 0.15,
                "max_nearest_pose_delta_ms": 20.0,
            },
        },
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
    static_transform = {
        **transform,
        "to": "robot_flange",
        "translation_mm": [110, 20, 530],
        "matrix": [[1, 0, 0, 110], [0, 1, 0, 20], [0, 0, 1, 530], [0, 0, 0, 1]],
    }
    static_recommended = {
        **recommended,
        "candidate_id": "oak_d_pro:static-1|IPPE|park",
        "profile_id": "static_ippe_park",
        "recommended": True,
        "score": 0.16,
        "primary_transform": static_transform,
        "companion_transform": {
            **static_transform,
            "from": "aruco_grid",
            "to": "template_base",
        },
    }
    failed = {
        "candidate_id": "oak_d_pro:static-1|ITERATIVE|li", "pnp_method": "ITERATIVE", "extrinsic_method": "li", "algorithms": ["ITERATIVE", "li"], "status": "error", "validation_state": "failed", "score": None, "observation_count": 3, "inlier_count": 0, "outlier_count": 3, "outlier_ratio": 1, "error": "leave-one-pose-out validation requires at least four poses",
    }

    def attempt_payload() -> dict:
        return {
            "schema_version": "calibration_attempt.v1",
            "attempt_id": "a" * 32,
            "request": {"mode": "eye_in_hand", "sensor_keys": ["realsense_d435:wrist-1", "oak_d_pro:static-1"], "target_id": setup["saved_targets"][0]["target_id"], "solver_policy": "auto_compare", "intrinsics_policy": "compare_factory_opencv", "synchronization_policy": "auto_offset"},
            "progress": {"status": "complete", "message": "Calibration calculations are complete and awaiting review.", "phases": [
                {"id": "prepare_data", "label": "Prepare data", "status": "complete"},
                {"id": "estimate_target_poses", "label": "Estimate target poses", "status": "complete"},
                {"id": "estimate_time_offsets", "label": "Estimate time alignment", "status": "complete"},
                {"id": "compare_robot_camera_solutions", "label": "Compare robot-camera solutions", "status": "complete"},
                {"id": "validate_and_rank", "label": "Validate and rank", "status": "complete"},
            ]},
            "results": {"status": "complete", "recommended_camera_count": 2, "failed_camera_count": 0, "results": [
                {**setup["cameras"][0], "status": "passing", "recommended_candidate_id": recommended["candidate_id"], "recommendation": recommended, "candidates": [recommended, override]},
                {**setup["cameras"][1], "status": "passing", "recommended_candidate_id": static_recommended["candidate_id"], "recommendation": static_recommended, "candidates": [static_recommended, failed]},
            ]},
            "intrinsic_comparison": {
                "policy": "compare_factory_opencv",
                "sensors": [
                    {
                        "sensor_key": "realsense_d435:wrist-1",
                        "status": "manual_selected",
                        "selected_profile_id": "wrist_manual",
                        "selection_reason": "manual_opencv_passed_all_intrinsic_quality_gates",
                        "factory_profile_id": "wrist_factory",
                        "manual_profile_id": "wrist_manual",
                        "manual_failure": None,
                        "deltas": {
                            "focal_length_delta_px": [1.25, -0.75],
                            "principal_point_delta_px": [0.5, -0.25],
                            "max_abs_distortion_delta": 0.002,
                        },
                        "candidates": [
                            {
                                "profile_id": "wrist_manual",
                                "source": {"mode": "calibrate"},
                                "quality": {
                                    "status": "accepted",
                                    "accepted_view_count": 18,
                                    "coverage_cells": [0, 1, 2, 3, 4, 5],
                                    "rms_reprojection_error_px": 0.82,
                                },
                            }
                        ],
                    },
                    {
                        "sensor_key": "oak_d_pro:static-1",
                        "status": "factory_selected",
                        "selected_profile_id": "static_factory",
                        "selection_reason": "manual_opencv_not_available_or_failed_quality_gates",
                        "factory_profile_id": "static_factory",
                        "manual_profile_id": None,
                        "manual_failure": {
                            "message": "coverage failed",
                            "quality": {"reason": "coverage 2/9 is below 6/9"},
                        },
                        "deltas": None,
                        "candidates": [],
                    },
                ],
            },
            "time_offset_search": {
                "policy": "auto_offset",
                "status": "complete",
                "sign_convention": {
                    "operator_equation": "robot_pose_query_time = frame_time + offset",
                    "positive_operator_value": "pair the frame with a robot pose recorded later",
                    "conversion": "sync_delta_ms = -robot_pose_time_offset_ms",
                },
                "search": {
                    "minimum_robot_pose_time_offset_ms": -150.0,
                    "maximum_robot_pose_time_offset_ms": 150.0,
                    "step_ms": 5.0,
                },
                "sensors": [
                    {
                        "sensor_key": "realsense_d435:wrist-1",
                        "sensor_name": "realsense_wrist-1",
                        "display_name": "Wrist RGB-D",
                        "status": "applied",
                        "decision_reason": "motion_disjoint_cross_validation_passed",
                        "selected_robot_pose_time_offset_ms": 65.0,
                        "selected_sync_delta_ms": -65.0,
                        "candidate_robot_pose_time_offset_ms": 65.0,
                        "evidence_strength": "strong",
                        "boundary_hit": False,
                        "split": {"motion_count": 17, "selected_observation_count": 102, "fold_motion_counts": {"0": 6, "1": 6, "2": 5}},
                        "cross_validation": {
                            "zero_offset": {"residuals": {"mean_translation_mm": 3.91, "median_translation_mm": 3.8, "max_translation_mm": 6.0, "mean_rotation_deg": 0.42, "median_rotation_deg": 0.4, "max_rotation_deg": 0.8}},
                            "candidate": {"residuals": {"mean_translation_mm": 2.77, "median_translation_mm": 2.6, "max_translation_mm": 4.8, "mean_rotation_deg": 0.39, "median_rotation_deg": 0.37, "max_rotation_deg": 0.7}},
                            "improvement": {"absolute_translation_mm": 1.14, "relative_translation": 0.29156, "rotation_change_deg": -0.03},
                        },
                        "checks": [
                            {"name": "cross_validation_offset_stability", "status": "ok", "actual": 10.0, "threshold": 22.0},
                        ],
                        "curve": [
                            {"robot_pose_time_offset_ms": 0.0, "residuals": {"mean_translation_mm": 3.91, "median_translation_mm": 3.8, "max_translation_mm": 6.0, "mean_rotation_deg": 0.42, "median_rotation_deg": 0.4, "max_rotation_deg": 0.8}},
                            {"robot_pose_time_offset_ms": 65.0, "residuals": {"mean_translation_mm": 2.77, "median_translation_mm": 2.6, "max_translation_mm": 4.8, "mean_rotation_deg": 0.39, "median_rotation_deg": 0.37, "max_rotation_deg": 0.7}},
                        ],
                    },
                    {
                        "sensor_key": "oak_d_pro:static-1",
                        "sensor_name": "luxonis_static-1",
                        "display_name": "Auxiliary OAK-D",
                        "status": "applied",
                        "decision_reason": "motion_disjoint_cross_validation_passed",
                        "selected_robot_pose_time_offset_ms": 85.0,
                        "selected_sync_delta_ms": -85.0,
                        "candidate_robot_pose_time_offset_ms": 85.0,
                        "evidence_strength": "consistent",
                        "boundary_hit": False,
                        "split": {"motion_count": 15, "selected_observation_count": 90, "fold_motion_counts": {"0": 5, "1": 5, "2": 5}},
                        "cross_validation": {
                            "zero_offset": {"residuals": {"mean_translation_mm": 4.7, "median_translation_mm": 4.5, "max_translation_mm": 7.0, "mean_rotation_deg": 0.5, "median_rotation_deg": 0.45, "max_rotation_deg": 0.9}},
                            "candidate": {"residuals": {"mean_translation_mm": 3.0, "median_translation_mm": 2.8, "max_translation_mm": 5.0, "mean_rotation_deg": 0.4, "median_rotation_deg": 0.38, "max_rotation_deg": 0.8}},
                            "improvement": {"absolute_translation_mm": 1.7, "relative_translation": 0.3617, "rotation_change_deg": -0.1},
                        },
                        "checks": [{"name": "reference_method_sensitivity", "status": "warning", "actual": 28.0, "warning_threshold": 22.0, "failure_threshold": 44.0}],
                        "curve": [],
                    },
                ],
            },
            "promotion": ({"status": "promoted", "promoted_profile_ids": ["wrist_sqpnp_tsai", "static_ippe_park"]} if promoted["value"] else None),
        }

    page.route("**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?**", lambda route: fulfill_json(route, attempt_payload()))

    def promote_handler(route) -> None:
        requests.append({"path": "/calibration/promote", "body": route.request.post_data_json})
        promoted["value"] = True
        fulfill_json(route, {"attempt_id": "a" * 32, "job_id": "promotion-1", "status": "queued", "selections": route.request.post_data_json["candidate_ids"]}, status=202)

    page.route("**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/promote", promote_handler)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{console_server.url}/#/workflow/calibration", wait_until="networkidle")

    expect(page.get_by_test_id("calibration-workflow")).to_be_visible()
    expect(page.locator('input[name="calibration-mode"]')).to_have_count(2)
    expect(page.get_by_text("Robot-mounted camera (eye-in-hand)")).to_be_visible()
    expect(page.get_by_text("Static camera (eye-to-hand)")).to_be_visible()
    expect(page.locator("[data-stage-id]")).to_have_count(0)
    intrinsics_guidance = page.locator('[data-workflow-step="calculate"]')
    expect(intrinsics_guidance).to_contain_text("Factory and OpenCV intrinsics")
    expect(intrinsics_guidance).to_contain_text(
        "A lower RMS alone does not make it the preferred model"
    )
    intrinsics_guidance.get_by_role(
        "button", name="About Factory and OpenCV intrinsics"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Factory is the per-camera projection supplied by the camera SDK"
    )
    page.keyboard.press("Escape")
    expect(page.get_by_text("Automatic solution comparison")).to_be_visible()
    expect(page.locator('input[value="auto_offset"]')).to_be_checked()
    expect(page.get_by_test_id("calibration-synchronization-policy")).to_contain_text(
        "It does not synchronize hardware clocks or rewrite raw frame or robot timestamps"
    )
    page.get_by_test_id("calibration-synchronization-policy").get_by_role(
        "button", name="About robot-pose time-offset sign"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "positive robot-pose time offset pairs a frame"
    )
    page.locator('input[value="fixed_zero"]').check()
    expect(page.locator('input[value="fixed_zero"]')).to_be_checked()
    page.locator('input[value="auto_offset"]').check()
    expect(page.locator('input[value="eye_in_hand"]')).to_be_checked()
    page.locator('input[value="eye_to_hand"]').check()
    expect(page.locator('input[value="eye_to_hand"]')).to_be_checked()
    page.locator('input[value="eye_in_hand"]').check()
    camera_choices = page.get_by_test_id("calibration-workflow").get_by_role("checkbox")
    expect(camera_choices).to_have_count(2)
    camera_choices.nth(1).click()
    expect(camera_choices.nth(1)).not_to_be_checked()
    camera_choices.nth(1).click()
    expect(page.get_by_test_id("calibration-workflow").get_by_text("Lab board", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Analyze recording")).to_be_enabled()
    page.get_by_role("button", name="Analyze recording").click()
    expect(page.get_by_text("Calibration queued")).to_be_visible()
    assert requests[0]["body"]["mode"] == "eye_in_hand"
    assert requests[0]["body"]["sensor_keys"] == ["realsense_d435:wrist-1", "oak_d_pro:static-1"]
    assert requests[0]["body"]["target_id"] == "5f09f41c-dd91-44ef-a048-1f43fc990e17"
    assert requests[0]["body"]["intrinsics_policy"] == "compare_factory_opencv"
    assert requests[0]["body"]["synchronization_policy"] == "auto_offset"

    expect(page.get_by_text("Prepare data")).to_be_visible()
    expect(page.locator('[data-phase-id="estimate_time_offsets"]')).to_contain_text(
        "Estimate time alignment"
    )
    expect(page.get_by_test_id("calibration-duration-guidance")).to_contain_text(
        "three-camera comparison usually takes 10–20 minutes"
    )
    expect(page.get_by_test_id("calibration-duration-guidance")).to_contain_text(
        "background job continues"
    )
    expect(page.get_by_test_id("calibration-results")).to_be_visible()
    alignment = page.get_by_test_id("calibration-time-alignment")
    expect(alignment).to_contain_text("not evidence that the hardware clocks are synchronized")
    page.mouse.move(0, 0)
    alignment.get_by_role(
        "button", name="About robot-pose time-offset evidence"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "positive offset uses a later robot pose"
    )
    expect(alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')).to_contain_text("+65.0 ms")
    expect(alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')).to_contain_text("3.910 → 2.770 mm")
    expect(alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')).to_contain_text("29.2%")
    oak_alignment = alignment.locator(
        '[data-time-offset-sensor="oak_d_pro:static-1"]'
    )
    expect(oak_alignment).to_contain_text("Applied with 1 warning")
    expect(oak_alignment).to_contain_text("reference method sensitivity")
    alignment.get_by_text("Advanced offset evidence · Wrist RGB-D").click()
    expect(alignment).to_contain_text("cross validation offset stability")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    expect(page.get_by_test_id("calibration-acceptance-thresholds")).to_contain_text("≥15 accepted views")
    wrist_intrinsics = page.get_by_test_id(
        "intrinsic-comparison-realsense_d435:wrist-1"
    )
    expect(wrist_intrinsics).to_contain_text("Using OpenCV estimate")
    expect(wrist_intrinsics).to_contain_text(
        "Factory values come from the camera SDK. The OpenCV estimate is fitted from this recording's grid views."
    )
    expect(wrist_intrinsics).to_contain_text(
        "OpenCV training views / image coverage"
    )
    expect(wrist_intrinsics).to_contain_text("18 views · 6 of 9 regions")
    static_intrinsics = page.get_by_test_id(
        "intrinsic-comparison-oak_d_pro:static-1"
    )
    expect(static_intrinsics).to_contain_text("Using factory SDK values")
    expect(static_intrinsics).to_contain_text(
        "The factory SDK values are compatible"
    )
    expect(static_intrinsics).to_contain_text("coverage 2/9 is below 6/9")
    expect(page.get_by_text("camera → robot_flange").last).to_be_visible()
    expect(page.get_by_text("All attempted solutions and failures").first).to_be_visible()
    wrist_result = page.locator(
        '[data-camera-key="realsense_d435:wrist-1"]'
    )
    wrist_result.get_by_text("Alternative solution (advanced)", exact=True).click()
    wrist_result.get_by_label("Alternative solution", exact=True).click()
    page.get_by_role("option", name="SQPNP + tsai · score 0.2000").click()
    page.get_by_role("button", name="Save selected calibrations").click()
    expect(page.get_by_text("Calibration acceptance queued")).to_be_visible()
    assert requests[-1]["body"]["candidate_ids"] == {
        "realsense_d435:wrist-1": override["candidate_id"],
        "oak_d_pro:static-1": static_recommended["candidate_id"],
    }
    expect(page.get_by_role("button", name="Calibrations saved")).to_be_visible()
    expect(page.get_by_text("Saved 2 camera profile(s).")).to_be_visible()


def calibration_time_alignment_setup(
    *,
    latest_attempt_id: str | None,
    latest_status: str = "complete",
) -> dict:
    latest_attempt = (
        {"attempt_id": latest_attempt_id, "status": latest_status}
        if latest_attempt_id
        else None
    )
    return {
        "schema_version": "calibration_setup.v1",
        "run_root": RUN_ROOT,
        "cameras": [
            {
                "sensor_key": "realsense_d435:wrist-1",
                "sensor_name": "realsense_wrist-1",
                "display_name": "Wrist RGB-D",
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "current_mounting_mode": "eye_in_hand",
            }
        ],
        "unavailable_cameras": [],
        "saved_targets": [
            {
                "target_id": "5f09f41c-dd91-44ef-a048-1f43fc990e17",
                "display_name": "Lab board",
                "valid": True,
                "selected": True,
            }
        ],
        "modes": [
            {
                "id": "eye_in_hand",
                "label": "Robot-mounted camera (eye-in-hand)",
                "primary_transform": "camera → robot_flange",
                "target_mounting": "stationary relative to template_base",
            },
            {
                "id": "eye_to_hand",
                "label": "Static camera (eye-to-hand)",
                "primary_transform": "camera → template_base",
                "target_mounting": "rigidly attached to robot_flange",
            },
        ],
        "solver": {
            "default_pnp_methods": ["IPPE", "ITERATIVE", "SQPNP"],
            "default_extrinsic_methods": ["shah", "li"],
            "intrinsics_policy": "compare_factory_opencv",
            "intrinsics_policies": [],
            "synchronization": {
                "default_policy": "auto_offset",
                "policies": [
                    {
                        "id": "auto_offset",
                        "label": "Auto-estimate robot-pose offset — recommended",
                        "description": "Estimate effective per-camera latency.",
                    },
                    {
                        "id": "fixed_zero",
                        "label": "Use captured timestamps (0 ms)",
                        "description": "Use the recorded pairing.",
                    },
                ],
                "search": {
                    "minimum_robot_pose_time_offset_ms": -150.0,
                    "maximum_robot_pose_time_offset_ms": 150.0,
                    "step_ms": 5.0,
                },
            },
            "thresholds": {
                "min_pnp_common_inliers": 12,
                "min_pnp_common_inlier_ratio": 0.5,
                "max_pnp_all_point_mean_reprojection_error_px": 3.0,
                "min_pnp_supported_markers": 4,
                "min_pnp_grid_rows": 2,
                "min_pnp_grid_columns": 2,
                "min_accepted_views": 15,
                "min_coverage_cells": 6,
                "max_per_view_reprojection_error_px": 3.0,
                "max_intrinsic_rms_reprojection_error_px": 1.5,
                "min_motion_poses": 4,
                "min_translation_span_mm": 20.0,
                "min_rotation_span_deg": 5.0,
                "min_rotation_axis_second_to_first_ratio": 0.15,
                "max_nearest_pose_delta_ms": 20.0,
            },
        },
        "latest_attempt": latest_attempt,
    }


def calibration_attempt_progress(
    *,
    status: str,
    time_alignment_status: str,
    message: str,
) -> dict:
    later_status = "pending" if status == "failed" else "complete"
    return {
        "status": status,
        "message": message,
        "phases": [
            {"id": "prepare_data", "label": "Prepare data", "status": "complete"},
            {
                "id": "estimate_target_poses",
                "label": "Estimate target poses",
                "status": "complete",
            },
            {
                "id": "estimate_time_offsets",
                "label": "Estimate time alignment",
                "status": time_alignment_status,
            },
            {
                "id": "compare_robot_camera_solutions",
                "label": "Compare robot-camera solutions",
                "status": later_status,
            },
            {
                "id": "validate_and_rank",
                "label": "Validate and rank",
                "status": later_status,
            },
        ],
    }


def calibration_failed_results(camera: dict) -> dict:
    return {
        "status": "failed",
        "recommended_camera_count": 0,
        "failed_camera_count": 1,
        "results": [
            {
                **camera,
                "status": "failed",
                "recommended_candidate_id": None,
                "recommendation": None,
                "candidates": [],
            }
        ],
    }


def test_failed_auto_sync_evidence_remains_visible_without_solver_results(
    console_server,
    page,
) -> None:
    attempt_id = "f" * 32
    setup = calibration_time_alignment_setup(
        latest_attempt_id=attempt_id,
        latest_status="failed",
    )
    install_common_mocks(page)
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))
    attempt = {
        "schema_version": "calibration_attempt.v1",
        "attempt_id": attempt_id,
        "request": {
            "mode": "eye_in_hand",
            "sensor_keys": ["realsense_d435:wrist-1"],
            "target_id": setup["saved_targets"][0]["target_id"],
            "solver_policy": "auto_compare",
            "intrinsics_policy": "compare_factory_opencv",
            "synchronization_policy": "auto_offset",
        },
        "progress": calibration_attempt_progress(
            status="failed",
            time_alignment_status="failed",
            message=(
                "ValueError: Auto-sync evidence failed closed for: "
                "realsense_d435:wrist-1"
            ),
        ),
        "results": None,
        "intrinsic_comparison": None,
        "time_offset_search": {
            "policy": "auto_offset",
            "status": "failed",
            "sign_convention": {
                "operator_equation": "robot_pose_query_time = frame_time + offset",
                "positive_operator_value": (
                    "pair the frame with a robot pose recorded later"
                ),
                "conversion": "sync_delta_ms = -robot_pose_time_offset_ms",
            },
            "search": {
                "minimum_robot_pose_time_offset_ms": -150.0,
                "maximum_robot_pose_time_offset_ms": 150.0,
                "step_ms": 5.0,
            },
            "sensors": [
                {
                    "sensor_key": "realsense_d435:wrist-1",
                    "sensor_name": "realsense_wrist-1",
                    "display_name": "Wrist RGB-D",
                    "status": "failed",
                    "decision_reason": (
                        "candidate_failed_safety_or_stability_checks"
                    ),
                    "selected_robot_pose_time_offset_ms": 0.0,
                    "selected_sync_delta_ms": 0.0,
                    "candidate_robot_pose_time_offset_ms": 150.0,
                    "evidence_strength": "insufficient",
                    "boundary_hit": True,
                    "split": {
                        "motion_count": 15,
                        "selected_observation_count": 90,
                        "fold_motion_counts": {"0": 5, "1": 5, "2": 5},
                    },
                    "cross_validation": {
                        "zero_offset": {
                            "residuals": {
                                "mean_translation_mm": 4.0,
                                "median_translation_mm": 3.9,
                                "max_translation_mm": 6.0,
                                "mean_rotation_deg": 0.5,
                                "median_rotation_deg": 0.45,
                                "max_rotation_deg": 0.9,
                            }
                        },
                        "candidate": {
                            "residuals": {
                                "mean_translation_mm": 2.8,
                                "median_translation_mm": 2.7,
                                "max_translation_mm": 4.5,
                                "mean_rotation_deg": 0.47,
                                "median_rotation_deg": 0.43,
                                "max_rotation_deg": 0.8,
                            }
                        },
                        "improvement": {
                            "absolute_translation_mm": 1.2,
                            "relative_translation": 0.3,
                            "rotation_change_deg": -0.03,
                        },
                    },
                    "checks": [
                        {
                            "name": "reference_method_sensitivity",
                            "status": "warning",
                            "actual": 30.0,
                            "warning_threshold": 22.0,
                            "failure_threshold": 44.0,
                        },
                        {
                            "name": "search_optimum_not_at_boundary",
                            "status": "error",
                            "actual": 150.0,
                            "threshold": [-150.0, 150.0],
                        },
                    ],
                    "curve": [
                        {
                            "robot_pose_time_offset_ms": 0.0,
                            "residuals": {
                                "mean_translation_mm": 4.0,
                                "median_translation_mm": 3.9,
                                "max_translation_mm": 6.0,
                                "mean_rotation_deg": 0.5,
                                "median_rotation_deg": 0.45,
                                "max_rotation_deg": 0.9,
                            },
                        },
                        {
                            "robot_pose_time_offset_ms": 150.0,
                            "residuals": {
                                "mean_translation_mm": 2.8,
                                "median_translation_mm": 2.7,
                                "max_translation_mm": 4.5,
                                "mean_rotation_deg": 0.47,
                                "median_rotation_deg": 0.43,
                                "max_rotation_deg": 0.8,
                            },
                        },
                    ],
                }
            ],
        },
        "promotion": None,
    }
    page.route(
        f"**/calibration/attempts/{attempt_id}?**",
        lambda route: fulfill_json(route, attempt),
    )

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=calculate",
        wait_until="networkidle",
    )

    expect(page.get_by_test_id("calibration-results")).to_have_count(0)
    alignment = page.get_by_test_id("calibration-time-alignment")
    expect(alignment).to_be_visible()
    expect(page.get_by_test_id("calibration-time-alignment-failed")).to_contain_text(
        "Auto time alignment stopped this calibration"
    )
    row = alignment.locator(
        '[data-time-offset-sensor="realsense_d435:wrist-1"]'
    )
    expect(row).to_contain_text("Time alignment rejected")
    expect(row).to_contain_text("Applied +0.0 ms")
    expect(row).to_contain_text("Rejected candidate +150.0 ms")
    expect(row).to_contain_text("0 ms → rejected +150.0 ms candidate")
    expect(row).to_contain_text("reference method sensitivity")
    expect(row).to_contain_text("search optimum not at boundary")
    expect(page.get_by_role("button", name="Save selected calibrations")).to_have_count(
        0
    )
    assert page.evaluate(
        "document.documentElement.scrollWidth <= "
        "document.documentElement.clientWidth"
    )


def test_fixed_zero_policy_is_submitted_and_reported(
    console_server,
    page,
) -> None:
    attempt_id = "0" * 32
    setup = calibration_time_alignment_setup(latest_attempt_id=None)
    requests: list[dict] = []
    install_common_mocks(page)
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))

    def create_handler(route) -> None:
        requests.append(route.request.post_data_json)
        fulfill_json(
            route,
            {"attempt_id": attempt_id, "job_id": "fixed-zero-job", "status": "queued"},
            status=202,
        )

    page.route("**/calibration/attempts", create_handler)
    attempt = {
        "schema_version": "calibration_attempt.v1",
        "attempt_id": attempt_id,
        "request": {
            "mode": "eye_in_hand",
            "sensor_keys": ["realsense_d435:wrist-1"],
            "target_id": setup["saved_targets"][0]["target_id"],
            "solver_policy": "auto_compare",
            "intrinsics_policy": "compare_factory_opencv",
            "synchronization_policy": "fixed_zero",
        },
        "progress": calibration_attempt_progress(
            status="complete",
            time_alignment_status="complete",
            message="Calibration calculations are complete and awaiting review.",
        ),
        "results": calibration_failed_results(setup["cameras"][0]),
        "intrinsic_comparison": None,
        "time_offset_search": {
            "policy": "fixed_zero",
            "status": "complete",
            "sign_convention": {
                "operator_equation": "robot_pose_query_time = frame_time + offset",
                "positive_operator_value": (
                    "pair the frame with a robot pose recorded later"
                ),
                "conversion": "sync_delta_ms = -robot_pose_time_offset_ms",
            },
            "search": {
                "minimum_robot_pose_time_offset_ms": -150.0,
                "maximum_robot_pose_time_offset_ms": 150.0,
                "step_ms": 5.0,
            },
            "sensors": [
                {
                    "sensor_key": "realsense_d435:wrist-1",
                    "sensor_name": "realsense_wrist-1",
                    "display_name": "Wrist RGB-D",
                    "status": "fixed_zero",
                    "decision_reason": "fixed_zero_policy_selected",
                    "selected_robot_pose_time_offset_ms": 0.0,
                    "selected_sync_delta_ms": 0.0,
                    "candidate_robot_pose_time_offset_ms": 0.0,
                    "evidence_strength": "not_applicable",
                    "boundary_hit": False,
                    "checks": [],
                    "curve": [],
                }
            ],
        },
        "promotion": None,
    }
    page.route(
        f"**/calibration/attempts/{attempt_id}?**",
        lambda route: fulfill_json(route, attempt),
    )
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=calculate",
        wait_until="networkidle",
    )

    page.locator('input[value="fixed_zero"]').check()
    page.get_by_role("button", name="Analyze recording").click()

    expect(page.get_by_text("Calibration queued")).to_be_visible()
    assert requests[-1]["synchronization_policy"] == "fixed_zero"
    row = page.get_by_test_id("calibration-time-alignment").locator(
        '[data-time-offset-sensor="realsense_d435:wrist-1"]'
    )
    expect(row).to_contain_text("Recorded timing kept")
    expect(row).to_contain_text("Applied +0.0 ms")
    expect(row).to_contain_text("not applicable")
    expect(page.get_by_test_id("calibration-time-alignment-failed")).to_have_count(0)


def test_historical_calibration_attempt_explains_missing_auto_sync_evidence(
    console_server,
    page,
) -> None:
    attempt_id = "1" * 32
    setup = calibration_time_alignment_setup(latest_attempt_id=attempt_id)
    install_common_mocks(page)
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))
    attempt = {
        "schema_version": "calibration_attempt.v1",
        "attempt_id": attempt_id,
        "request": {
            "mode": "eye_in_hand",
            "sensor_keys": ["realsense_d435:wrist-1"],
            "target_id": setup["saved_targets"][0]["target_id"],
            "solver_policy": "auto_compare",
            "intrinsics_policy": "compare_factory_opencv",
        },
        "progress": calibration_attempt_progress(
            status="complete",
            time_alignment_status="complete",
            message="Calibration calculations are complete and awaiting review.",
        ),
        "results": calibration_failed_results(setup["cameras"][0]),
        "intrinsic_comparison": None,
        "promotion": None,
    }
    page.route(
        f"**/calibration/attempts/{attempt_id}?**",
        lambda route: fulfill_json(route, attempt),
    )

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=calculate",
        wait_until="networkidle",
    )

    alignment = page.get_by_test_id("calibration-time-alignment")
    expect(alignment).to_contain_text("Legacy timing evidence unavailable")
    expect(alignment).to_contain_text(
        "not reusable for a new dataset"
    )
    expect(alignment.locator("table")).to_have_count(0)


def test_calibration_target_preview_fit_generate_download_select_and_run_switch(
    console_server, page
) -> None:
    requests: list[dict] = []
    library_urls: list[str] = []
    deleted = {"value": False}
    selected_runs: set[str] = set()
    locked_runs: set[str] = set()
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
            "replacement_blockers": (
                ["calibration_observations.json"] if run_root in locked_runs else []
            ),
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
    library_preview_urls: list[str] = []

    def library_preview_handler(route) -> None:
        library_preview_urls.append(route.request.url)
        route.fulfill(status=200, content_type="image/png", body=png)

    page.route(
        f"**/calibration-targets/bundles/{target_id}/preview.png",
        library_preview_handler,
    )

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
        selected_runs.add(body["run_root"])
        locked_runs.add(body["run_root"])
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
    expect(page.get_by_role("img", name="Calibration target preview", exact=True)).to_be_visible()
    library_preview = page.get_by_role(
        "img",
        name="Anisotropic calibration board calibration target preview",
        exact=True,
    )
    expect(library_preview).to_be_visible()
    expect(library_preview).to_have_js_property("complete", True)
    assert library_preview.evaluate("image => image.naturalWidth") > 0
    assert library_preview.evaluate("image => image.naturalHeight") > 0
    preview_box = library_preview.bounding_box()
    preview_container_box = page.get_by_test_id(
        "calibration-target-library-preview"
    ).bounding_box()
    assert preview_box is not None
    assert preview_container_box is not None
    assert preview_box["width"] <= preview_container_box["width"]
    assert preview_box["height"] <= preview_container_box["height"]
    natural_ratio = library_preview.evaluate(
        "image => image.naturalWidth / image.naturalHeight"
    )
    rendered_ratio = preview_box["width"] / preview_box["height"]
    assert rendered_ratio == pytest.approx(natural_ratio, abs=0.01)
    expect(page.get_by_test_id("calibration-target-library-preview")).to_be_visible()
    expect(page.get_by_text("297 × 210 mm", exact=True)).to_be_visible()
    preview_page_ratio = page.get_by_test_id("calibration-preview-page").evaluate(
        "element => { const box = element.getBoundingClientRect(); return box.width / box.height }"
    )
    assert preview_page_ratio == pytest.approx(297 / 210, abs=0.002)
    assert any(item["path"] == "/calibration-targets/preview" for item in requests)
    assert library_preview_urls

    page.get_by_role("button", name="Fit to page").click()
    expect(page.get_by_text("Board fitted to the selected page")).to_be_visible()
    page.get_by_label("Target display name").fill("Printed target 01")
    page.get_by_role("button", name="Generate bundle").click()
    expect(page.get_by_text("Calibration target generated")).to_be_visible()
    generated = next(item["body"] for item in requests if item["path"] == "/calibration-targets/generate")
    assert generated["display_name"] == "Printed target 01"
    assert generated["configuration"]["print_compensation"] == {"x_percent": 101, "y_percent": 99}
    expect(page.get_by_text("Active for this run", exact=True)).to_have_count(0)

    pdf_link = page.get_by_role("link", name="PDF")
    expect(pdf_link).to_have_attribute(
        "href", f"/calibration-targets/bundles/{target_id}/download/pdf"
    )
    expect(pdf_link).to_have_attribute("download", "")

    page.get_by_role("button", name="Select for run").click()
    page.get_by_role("combobox", name="Target placement").click()
    page.get_by_role("option", name="Use PoseGridGen board pose").click()
    page.get_by_role("button", name="Select target").click()
    expect(page.get_by_text("Calibration target selected")).to_be_visible()
    selection = [item["body"] for item in requests if item["path"] == "/calibration-targets/select"][-1]
    assert selection == {"run_root": RUN_ROOT, "placement": "posegridgen_board_to_base"}
    expect(page.get_by_text("Active for this run", exact=True)).to_be_visible()
    reuse_notice = page.get_by_test_id("calibration-target-reuse-notice")
    expect(reuse_notice).to_contain_text("Target selection fixed for this run")
    expect(reuse_notice).to_contain_text(
        "after moving cameras, choose a fresh run folder"
    )
    select_request_count = len(
        [item for item in requests if item["path"] == "/calibration-targets/select"]
    )
    page.get_by_role("button", name="Review active target").click()
    expect(page.get_by_role("combobox", name="Target placement")).to_be_disabled()
    expect(page.get_by_text("Placement is fixed only for this completed run")).to_be_visible()
    page.get_by_role("dialog").get_by_role(
        "button", name="Close", exact=True
    ).first.click()
    assert len(
        [item for item in requests if item["path"] == "/calibration-targets/select"]
    ) == select_request_count
    expect(page.get_by_role("button", name="Delete Anisotropic calibration board")).to_be_disabled()

    page.get_by_role("combobox", name="Selected run").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_text("Active for this run", exact=True)).to_have_count(0)
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
    target_frame = {
        "name": "aruco_grid",
        "origin": "compensated_outer_board_top_left",
        "axes": {"x": "right", "y": "down", "z": "into_board"},
    }
    return {
        "schema_version": "cell_scene.v1",
        "coordinate_system": {
            "units": "millimetres",
            "handedness": "right",
            "up_axis": "-Z",
            "reference_frame": "template_base",
            "transform_semantics": "entity_to_parent",
            "presentation": {
                "mode": "calibration_target_front",
                "presentation_only": True,
                "source_frame": "template_base",
                "anchor_frame": "calibration_target",
                "display_up_axis": "+Z",
                "source_front_axis": "-Z",
                "matrix": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                "transform": {
                    "semantics": "entity_to_parent",
                    "parent_frame": "display",
                    "translation_mm": [0, 0, 0],
                    "rotation_quaternion_wxyz": [0, 1, 0, 0],
                },
                "target_frame": target_frame,
            },
        },
        "run_root": RUN_ROOT,
        "entities": [
            {"id": "template_base", "type": "reference_frame", "label": "Template base", "status": "planned", "transform": {**identity, "parent_frame": None}, "unresolved_reason": None, "geometry": {"kind": "axes", "size_mm": 100}, "provenance": {"source": "config"}},
            {"id": "robot_flange", "type": "robot_flange", "label": "Robot flange", "status": "recorded", "transform": identity, "unresolved_reason": None, "geometry": {"kind": "flange_proxy"}, "provenance": {"source": "match_robot_ee_poses.json"}},
            {
                "id": "camera:realsense_123",
                "type": "camera",
                "label": "Wrist D435",
                "status": "planned",
                "transform": {**identity, "parent_frame": "robot_flange", "translation_mm": [10, 20, -500]},
                "unresolved_reason": None,
                "geometry": {"kind": "camera_frustum", "width": 1280, "height": 720, "fx": 900, "fy": 900, "cx": 640, "cy": 360},
                "provenance": {"source": "calibration_profiles.json", "profile_id": "wrist-profile"},
                "calibration": {
                    "profile_id": "wrist-profile",
                    "schema_version": "calibration.v2",
                    "status": "valid",
                    "mounting_mode": "eye_in_hand",
                    "rig_position": "wrist",
                    "extrinsics": {
                        "from": "camera",
                        "to": "robot_flange",
                        "matrix": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
                        "rotation_quaternion_wxyz": [1, 0, 0, 0],
                        "translation_mm": [10, 20, 30],
                    },
                    "companion_transform": {
                        "from": "aruco_grid",
                        "to": "template_base",
                        "matrix": [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]],
                        "rotation_quaternion_wxyz": [1, 0, 0, 0],
                        "translation_mm": [1, 2, 3],
                    },
                    "quality": {
                        "num_observations": 12,
                        "num_inliers": 10,
                        "mean_reprojection_error_px": 0.321,
                        "max_reprojection_error_px": 0.8,
                        "residual_translation_mm": 0.75,
                        "residual_rotation_deg": 0.4,
                        "outlier_count": 2,
                        "outlier_ratio": 0.1667,
                        "held_out_residuals": {
                            "translation_mean_mm": 0.75,
                            "rotation_mean_deg": 0.4,
                            "fold_count": 3,
                        },
                        "notes": None,
                    },
                    "evidence": {
                        "profile_source": "/tmp/run/calibration_profiles.json",
                        "method": "auto_compare:IPPE+park",
                        "calibration_dataset_id": "attempt-dataset",
                        "target_type": "aruco_grid",
                        "target_id": "target-1",
                        "calibrated_at": "2026-07-21T12:00:00+00:00",
                        "operator": "operator",
                        "sync_delta_ms": 1.2,
                        "promotion_attempt_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "promotion_candidate_id": "realsense_d435:123|IPPE|park",
                        "promotion_multi_camera_bundle_id": "joint:IPPE:park",
                        "promotion_solver_provenance": {"solver_policy": "auto_compare", "pnp_method": "IPPE", "extrinsic_method": "park"},
                        "promoted_at": "2026-07-21T12:00:00+00:00",
                        "promoted_by": "operator",
                        "intrinsic_profile_id": "123_1280x720_normal_factory",
                    },
                },
            },
            {"id": "camera:missing", "type": "camera", "label": "Uncalibrated camera", "status": "unresolved", "transform": None, "unresolved_reason": "No valid calibration profile", "geometry": {"kind": "camera_frustum"}, "provenance": {"source": "calibration_profiles"}},
            {
                "id": "calibration_target",
                "type": "calibration_target",
                "label": "calib00 (reference placement)",
                "status": "reference",
                "transform": identity,
                "unresolved_reason": None,
                "geometry": {
                    "kind": "calibration_target",
                    "placement_known": False,
                    "frame": target_frame,
                    "target_bounds": {"x_mm": 0, "y_mm": 0, "width_mm": 90, "height_mm": 40},
                    "markers": [{"id": 0, "corners_mm": [[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0]]}],
                    "pdf_url": "/ui/cell-calibration-target-pdf?run_root=test",
                },
                "provenance": {"source": "processed/calibration/attempt/target_bundle/calibration_target.json", "placement_known": False},
            },
            *([] if objectless else [{
                "id": "pose_template_footprint",
                "type": "template",
                "label": "Exact object footprint",
                "status": "planned",
                "transform": identity,
                "unresolved_reason": None,
                "geometry": {
                    "kind": "pose_template_footprint",
                    "page": {"width_mm": 420, "height_mm": 297},
                    "page_configuration": {"origin_from_lower_left_mm": [15, 15]},
                    "contours": [{"instance_uuid": "object-1", "contours": [[{"x_mm": 20, "y_mm": 20}, {"x_mm": 50, "y_mm": 20}, {"x_mm": 35, "y_mm": 50}]]}],
                },
                "provenance": {"source": "pose_template_preview.json"},
            }]),
        ],
        "warnings": [{"code": "missing_calibration_profiles", "message": "No calibration profile collection is available"}],
        "timelines": [{"id": "sensor:realsense_123", "label": "realsense_123", "kind": "synchronized", "frame_count": 2, "default": True, "exact": True, "interpolation": "none", "page_limit": 2000, "source": "match_robot_ee_poses.json"}],
        "default_timeline_id": "sensor:realsense_123",
        "trajectory_preview": [
            {"index": 0, "frame_index": 0, "frame_id": "000000.png", "timestamp_ns": 1, "motion": "arc", "transform": identity},
            {"index": 1, "frame_index": 1, "frame_id": "000001.png", "timestamp_ns": 2, "motion": "arc", "transform": {**identity, "translation_mm": [10, 20, 30]}},
        ],
        "object_selection": {
            "objectless": objectless,
            "dataset_mode": "objectless" if objectless else "pose_template",
            "instance_count": 0 if objectless else 1,
            "pose_template": None if objectless else {"template_uuid": "test-template"},
            "bop_export": {"status": "not_exported"},
        },
    }


def test_cell_canvas_layers_inspection_and_exact_seeking(console_server, page) -> None:
    install_common_mocks(page)
    scene = cell_scene_payload()
    calibration = scene["entities"][2]["calibration"]
    calibration["extrinsics"]["matrix"][0][3] = "10"
    calibration["extrinsics"]["rotation_quaternion_wxyz"] = ["1", "0", "0", "0"]
    calibration["extrinsics"]["translation_mm"] = ["10", "20", "30"]
    calibration["quality"]["mean_reprojection_error_px"] = "0.321"
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, scene))
    page.route("**/ui/cell-scene/timeline?**", lambda route: fulfill_json(route, {"schema_version": "cell_timeline.v1", "timeline": scene["timelines"][0], "offset": 0, "limit": 2000, "total": 2, "next_offset": None, "previous_offset": None, "poses": scene["trajectory_preview"]}))

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_test_id("cell-webgl-canvas")).to_be_visible()
    expect(page.get_by_test_id("cell-webgl-canvas")).to_have_attribute(
        "data-presentation-mode", "calibration_target_front"
    )
    expect(page.get_by_test_id("cell-webgl-canvas")).to_have_attribute(
        "data-presentation-quaternion", "0,1,0,0"
    )
    expect(page.get_by_test_id("cell-coordinate-convention")).to_contain_text(
        "origin top-left · +X right · +Y down · +Z into grid"
    )
    expect(page.get_by_text("Partial cell scene")).to_be_visible()
    expect(page.get_by_text("1 camera is hidden", exact=False)).to_be_visible()
    expect(page.get_by_text("Exact object footprint", exact=True)).to_be_visible()
    page.get_by_text("calib00 (reference placement)", exact=True).click()
    expect(page.get_by_text("Shown at the reference origin", exact=False)).to_be_visible()
    expect(page.get_by_role("link", name="Open exact calibration-target PDF")).to_be_visible()
    page.get_by_text("Wrist D435", exact=True).click()
    evidence = page.get_by_test_id("cell-calibration-evidence")
    expect(evidence.get_by_text("Calibration extrinsic", exact=True)).to_be_visible()
    expect(page.get_by_test_id("cell-calibration-transform-frames")).to_have_text("camera → robot_flange")
    expect(
        evidence.get_by_text(
            "1.0000000, 0.0000000, 0.0000000, 0.0000000",
            exact=True,
        ).first
    ).to_be_visible()
    expect(evidence.get_by_text("10.0000, 20.0000, 30.0000", exact=True)).to_be_visible()
    expect(evidence.get_by_text("12 / 10", exact=True)).to_be_visible()
    expect(evidence.get_by_text("0.321 px", exact=True)).to_be_visible()
    expect(evidence.get_by_text("0.800 px", exact=True)).to_be_visible()
    expect(evidence.get_by_text("0.1667", exact=True)).to_be_visible()
    expect(evidence.get_by_text("1.200 ms", exact=True)).to_be_visible()
    expect(evidence.get_by_text("operator / operator", exact=True)).to_be_visible()
    expect(evidence.get_by_text("attempt-dataset", exact=True)).to_be_visible()
    expect(evidence.get_by_text('"fold_count":3', exact=False)).to_be_visible()
    expect(evidence.get_by_text("IPPE + park", exact=True)).to_be_visible()
    expect(evidence.get_by_text("wrist-profile", exact=True)).to_be_visible()
    expect(page.get_by_test_id("cell-calibration-matrix")).to_contain_text("10.000000")
    expect(page.get_by_test_id("cell-calibration-companion-frames")).to_have_text("aruco_grid → template_base")
    expect(page.get_by_test_id("cell-calibration-companion-matrix")).to_contain_text("3.000000")
    expect(evidence.get_by_text("joint:IPPE:park", exact=True)).to_be_visible()
    page.get_by_text("Raw provenance", exact=True).click()
    raw_provenance = page.get_by_test_id("cell-raw-provenance")
    expect(raw_provenance).to_contain_text('"calibration_dataset_id": "attempt-dataset"')
    expect(raw_provenance).to_contain_text('"outlier_ratio": 0.1667')
    expect(raw_provenance).to_contain_text('"sync_delta_ms": 1.2')
    page.get_by_text("Robot flange", exact=True).click()
    expect(page.get_by_text("10.00, 20.00, 30.00")).not_to_be_visible()
    page.get_by_role("slider", name="Frame scrubber").fill("1")
    expect(page.get_by_text("Exact frame 000001.png · arc")).to_be_visible()
    page.get_by_text("Recorded trajectory", exact=True).click()
    expect(page.get_by_role("checkbox", name="Recorded trajectory")).not_to_be_checked()


def test_cell_webgl_fallback_and_objectless_state(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script("HTMLCanvasElement.prototype.getContext = () => null")
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, cell_scene_payload(objectless=True)))
    page.route("**/ui/cell-scene/timeline?**", lambda route: fulfill_json(route, {"schema_version": "cell_timeline.v1", "timeline": cell_scene_payload()["timelines"][0], "offset": 0, "limit": 2000, "total": 0, "next_offset": None, "previous_offset": None, "poses": []}))

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_test_id("cell-webgl-fallback")).to_be_visible()
    expect(page.get_by_text("WebGL is unavailable")).to_be_visible()
    expect(page.get_by_text("Objectless RGB-D run")).to_be_visible()
    expect(page.get_by_text("Robot flange", exact=True)).to_be_visible()
