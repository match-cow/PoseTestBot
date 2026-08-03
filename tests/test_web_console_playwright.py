from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse
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
    selected_bundle = (resolved_config.get("calibration_profile_selection") or {}).get(
        "bundle_sha256", "a" * 64
    )
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
    preflight_state = (
        preflight_state if preflight_state is not None else {"blocker": None}
    )
    config_payload = config_payload if config_payload is not None else run_config()

    page.route(
        "**/ui/bootstrap",
        lambda route: fulfill_json(
            route,
            {
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
            },
        ),
    )
    page.route(
        "**/ui/runs",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "web_run_index.v1",
                "runs": [
                    {
                        "path": RUN_ROOT,
                        "name": "new-run",
                        "sequence": "real_full_capture_validation",
                        "plan_only": True,
                        "config_valid": True,
                        "config_error": None,
                        "modified_at": "2026-07-10T12:00:00Z",
                    },
                    {
                        "path": "/tmp/posetestbot-console/old-run",
                        "name": "old-run",
                        "sequence": "sync_aruco",
                        "plan_only": True,
                        "config_valid": True,
                        "config_error": None,
                        "modified_at": "2026-07-09T12:00:00Z",
                    },
                ],
            },
        ),
    )
    page.route(
        "**/calibration-targets/status",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "calibration_target_generator_status.v1",
                "generation_available": generator_available,
                "generator": {
                    "checkout": "/repo/third_party/PoseGridGen",
                    "required_revision": "9e6975901fe096bf65f7b7b599d7b82461d2e67c",
                    "reason": None
                    if generator_available
                    else "Pinned source checkout is unavailable",
                },
            },
        ),
    )
    page.route(
        "**/ui/overview**",
        lambda route: fulfill_json(route, overview_payload(config_payload)),
    )
    page.route(
        "**/ui/storage**",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "run_storage.v1",
                "run_root": RUN_ROOT,
                "filesystem_path": "/tmp",
                "status": "ready",
                "total_bytes": 3 * 1024**4,
                "used_bytes": 7 * 1024**4 // 4,
                "free_bytes": 5 * 1024**4 // 4,
                "free_fraction": 5 / 12,
                "thresholds": {
                    "critical_free_bytes": 100 * 1024**3,
                    "warning_free_bytes": int(3 * 1024**4 * 0.15),
                    "critical_free_bytes_cap": 100 * 1024**3,
                    "warning_free_bytes_cap": 500 * 1024**3,
                    "critical_free_fraction": 0.05,
                    "warning_free_fraction": 0.15,
                },
                "error": None,
            },
        ),
    )
    page.route(
        "**/sensors/status",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "sensor_status.v1",
                "families": [],
                "total_connected": 0,
                "all_expected_connected": True,
            },
        ),
    )
    page.route(
        "**/robot/status",
        lambda route: fulfill_json(
            route,
            {"schema_version": "robot_status.v2", "selected_profile": {"mode": "real"}},
        ),
    )
    page.route(
        "**/runtime/status",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "runtime_status.v1",
                "runtimes": [{"runtime_id": "blenderproc", "available": True}],
            },
        ),
    )
    page.route(
        "**/bop/annotations/setup?**",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "bop_annotation_setup.v1",
                "run_root": RUN_ROOT,
                "runtime": {
                    "available": True,
                    "required_version": "2.8.0",
                    "detected_version": "2.8.0",
                    "install_command": None,
                    "reason": None,
                },
                "toolkit": {
                    "available": True,
                    "status": "ready",
                    "revision": "renderer-revision",
                    "required_revision": "renderer-revision",
                    "environment_ready": True,
                    "renderer": "vispy",
                    "install_command": None,
                    "reason": None,
                },
                "readiness": {
                    "ready": False,
                    "blockers": [
                        {
                            "code": "bop_export_missing",
                            "message": "Complete the base BOP export first.",
                        }
                    ],
                    "warnings": [],
                },
                "readiness_by_mode": {
                    "pose": {
                        "ready": False,
                        "blockers": [
                            {
                                "code": "bop_export_missing",
                                "message": "Complete the base BOP export first.",
                            }
                        ],
                        "warnings": [],
                    },
                    "pose_and_masks": {
                        "ready": False,
                        "blockers": [
                            {
                                "code": "bop_export_missing",
                                "message": "Complete the base BOP export first.",
                            }
                        ],
                        "warnings": [],
                    },
                },
                "current_output": None,
                "counts": {"sensors": 0, "frames": 0, "instances": 0},
            },
        ),
    )
    page.route(
        "**/capture/jobs**",
        lambda route: fulfill_json(
            route,
            {"jobs": [], "active_count": 0, "resources": {}, "status_artifact": None},
        ),
    )
    page.route(
        "**/jobs", lambda route: fulfill_json(route, {"jobs": [], "resources": {}})
    )
    page.route(
        "**/jobs?**",
        lambda route: fulfill_json(
            route,
            {
                "jobs": [],
                "resources": {},
                "total": 0,
                "status_counts": {},
                "next_cursor": None,
                "limit": 20,
            },
        ),
    )
    page.route(
        "**/monitoring/webcam",
        lambda route: fulfill_json(
            route,
            {
                "job": {"id": "monitor-1", "status": "failed"},
                "webrtc_status": {
                    "schema_version": "monitor_webrtc.v1",
                    "transport": "webrtc",
                    "status": "failed",
                    "signaling_ready": False,
                    "peer_count": 0,
                    "frame_count": 0,
                    "selected_node": None,
                    "error": "mock camera offline",
                },
            },
        ),
    )
    page.route(
        "**/pipeline/sequences",
        lambda route: fulfill_json(
            route,
            {
                "sequences": [
                    {
                        "id": "real_full_capture_validation",
                        "label": "Real Full Capture Validation",
                        "description": "Safe plan",
                        "steps": [],
                    }
                ]
            },
        ),
    )
    page.route(
        "**/pipeline/stages",
        lambda route: fulfill_json(
            route,
            {
                "stages": [
                    {
                        "id": "capture_plan",
                        "label": "Capture Plan",
                        "description": "Write a command plan without hardware.",
                        "resources": ["disk_io"],
                        "parameters": [],
                    }
                ]
            },
        ),
    )

    def config_handler(route) -> None:
        if route.request.method == "POST":
            requests.append(
                {"path": "/run-config", "body": route.request.post_data_json}
            )
            fulfill_json(
                route, {"config": config_payload, "output": "written"}, status=201
            )
        else:
            fulfill_json(
                route,
                {
                    "config": config_payload,
                    "preflight": {"queue_blocker": preflight_state["blocker"]},
                },
            )

    page.route("**/run-config**", config_handler)

    def pipeline_handler(route) -> None:
        requests.append({"path": "/pipeline/run", "body": route.request.post_data_json})
        fulfill_json(
            route, {"job_id": f"job-{len(requests)}", "status": "queued"}, status=202
        )

    page.route("**/pipeline/run", pipeline_handler)
    page.route(
        "**/sensors/previews/stop",
        lambda route: (
            requests.append({"path": "/sensors/previews/stop", "body": {}}),
            fulfill_json(route, {"jobs": []}),
        )[1],
    )


def test_navigation_run_fallback_persistence_and_both_themes(
    console_server, page
) -> None:
    install_common_mocks(page)
    page.emulate_media(color_scheme="dark")
    page.add_init_script(
        "if (!localStorage.getItem('posetestbot.selectedRun')) localStorage.setItem('posetestbot.selectedRun', '/tmp/posetestbot-console/deleted-run'); localStorage.removeItem('posetestbot.theme')"
    )

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
        "Run folders",
        "BOP Evaluation",
        "Jobs",
    ]
    expect(primary_navigation.get_by_role("link", name="Workflow")).to_have_attribute(
        "href", "#/workflow/setup"
    )
    expect(page.get_by_text("Recommended next action", exact=True)).to_have_count(0)
    workflow_overview = page.get_by_test_id("dashboard-workflow-overview")
    expect(workflow_overview).to_have_attribute("data-workflow-journey", "calibration")
    expect(
        workflow_overview.get_by_role("heading", name="Camera calibration workflow")
    ).to_be_visible()
    expect(workflow_overview.locator("[data-workflow-step]")).to_have_count(5)
    expect(
        page.get_by_role(
            "link",
            name="Open camera calibration step 1: Configure the run and cameras",
        )
    ).to_have_attribute("href", "#/workflow/calibration?step=configure")
    assert page.evaluate("localStorage.getItem('posetestbot.theme')") is None
    assert page.evaluate("localStorage.getItem('posetestbot.selectedRun')") is None
    active_run_context = page.get_by_test_id("active-run-context")
    expect(active_run_context).to_contain_text("Active run folder")
    expect(active_run_context).to_contain_text(
        "All run-owned pages and actions use this folder"
    )
    expect(
        page.get_by_role("combobox", name="Active run folder").get_by_text("Change")
    ).to_be_visible()
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        "new-run"
    )
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        "old-run"
    )
    page.get_by_role("complementary", name="Application sidebar").get_by_role(
        "link", name="Devices"
    ).click()
    expect(page).to_have_url(f"{console_server.url}/#/devices")
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        "old-run"
    )
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="Create or open a run folder…").click()
    expect(
        page.get_by_role("heading", name="Create or open a run folder")
    ).to_be_visible()
    expect(
        page.get_by_text(
            "Each acquisition run is a separate folder",
            exact=False,
        )
    ).to_be_visible()
    expect(page.get_by_text("Choose one folder per acquisition run.")).to_be_visible()
    expect(page.get_by_role("combobox", name="Run storage root")).to_contain_text(
        "/tmp/posetestbot-console"
    )
    expect(page.locator("#new-run-name")).to_have_value("")
    custom_run = "/tmp/posetestbot-console/unlisted-run"
    page.locator("#new-run-name").fill("unlisted-run")
    expect(page.get_by_test_id("new-run-path-preview")).to_have_text(custom_run)
    page.get_by_role("button", name="Use run folder", exact=True).click()
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        custom_run
    )
    assert (
        page.evaluate("localStorage.getItem('posetestbot.selectedRun')") == custom_run
    )
    page.reload(wait_until="networkidle")
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        custom_run
    )
    assert (
        page.evaluate("localStorage.getItem('posetestbot.selectedRun')") == custom_run
    )
    page.get_by_role("button", name="Open operator console guide").click()
    expect(page.get_by_role("heading", name="Operator console guide")).to_be_visible()
    expect(
        page.get_by_text("Choose an outcome in Workflow", exact=True)
    ).to_be_visible()
    expect(
        page.get_by_text("IIWA STOP is not a safety stop", exact=False)
    ).to_be_visible()
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
    expect(
        page.get_by_role("link", name="Open PoseTestBot on GitHub")
    ).to_have_attribute("href", "https://github.com/match-cow/PoseTestBot")
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
    expect(
        page.get_by_text(
            "Physical capture always requires fresh operator acknowledgement.",
            exact=True,
        )
    ).to_have_count(0)
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_css(
        "background-color", "rgba(0, 0, 0, 0)"
    )
    expect(page.get_by_role("img", name="PoseTestBot")).to_have_css("padding", "0px")


def test_run_folders_inventory_move_delete_and_local_overflow(
    console_server, page
) -> None:
    install_common_mocks(page)
    primary_root = "/tmp/posetestbot-console"
    archive_root = "/mnt/posetestbot-archive"
    movable_run = f"{primary_root}/archive-run"
    disposable_run = f"{primary_root}/disposable-run"
    active_run_alias = f"{primary_root}/active-run-alias"
    move_identity = {"device": 101, "inode": 201}
    delete_identity = {"device": 101, "inode": 202}
    move_requests: list[dict] = []
    delete_requests: list[dict] = []
    refresh_requests: list[dict] = []
    job_status = {"move-run-folder": "queued", "delete-run-folder": "queued"}
    move_submitted = {"value": False}
    inventory_stale = {"value": False}
    inventory_cache_missing = {"value": True}

    def storage(root: str) -> dict:
        return {
            "schema_version": "run_storage.v1",
            "run_root": root,
            "filesystem_path": root,
            "status": "ready",
            "total_bytes": 2 * 1024**4,
            "used_bytes": 512 * 1024**3,
            "free_bytes": 1536 * 1024**3,
            "free_fraction": 0.75,
            "thresholds": {
                "critical_free_bytes": 100 * 1024**3,
                "warning_free_bytes": 300 * 1024**3,
                "critical_free_bytes_cap": 100 * 1024**3,
                "warning_free_bytes_cap": 500 * 1024**3,
                "critical_free_fraction": 0.05,
                "warning_free_fraction": 0.15,
            },
            "error": None,
        }

    def run_folder(
        *,
        path: str,
        size_bytes: int,
        identity: dict,
        sensor_name: str,
        mounting_mode: str,
        object_names: list[str],
        evidence: dict,
    ) -> dict:
        return {
            "path": path,
            "name": Path(path).name,
            "root": primary_root,
            "modified_at": "2026-07-29T08:30:00Z",
            "size_bytes": size_bytes,
            "allocated_bytes": size_bytes + 4096,
            "file_count": 24,
            "directory_count": 8,
            "symlink_count": 0,
            "scan_complete": True,
            "scan_error_count": 0,
            "scan_errors": [],
            "identity": identity,
            "config": {
                "valid": True,
                "error": None,
                "run_name": Path(path).name,
                "sequence": "calibrated_capture_to_bop_dataset_dry_run",
                "plan_only": True,
            },
            "contents": {
                "dataset_mode": "pose_template" if object_names else "objectless",
                "resolution": "720p",
                "fps": 6,
                "synchronization_mode": "timestamp_aligned",
                "sensor_count": 1,
                "enabled_sensor_count": 1,
                "sensors": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "wrist-1",
                        "name": sensor_name,
                        "mounting_mode": mounting_mode,
                        "enabled": True,
                    }
                ],
                "object_count": len(object_names),
                "object_names": object_names,
                "template_uuid": (
                    "22222222-2222-4222-8222-222222222222" if object_names else None
                ),
                "evidence": evidence,
            },
            "breakdown": {
                "raw_capture": {
                    "size_bytes": size_bytes // 2,
                    "allocated_bytes": size_bytes // 2,
                    "file_count": 12,
                },
                "processed": {
                    "size_bytes": size_bytes // 4,
                    "allocated_bytes": size_bytes // 4,
                    "file_count": 6,
                },
            },
            "relocation": None,
        }

    inventory_runs = [
        run_folder(
            path=RUN_ROOT,
            size_bytes=3 * 1024**3,
            identity={"device": 101, "inode": 200},
            sensor_name="Active wrist camera",
            mounting_mode="eye_in_hand",
            object_names=[],
            evidence={
                "raw_capture": True,
                "synchronized": False,
                "calibration": True,
                "bop_export": False,
                "bop_evaluation": False,
            },
        ),
        run_folder(
            path=movable_run,
            size_bytes=8 * 1024**3,
            identity=move_identity,
            sensor_name="Wrist RGB-D",
            mounting_mode="eye_in_hand",
            object_names=["Clamp", "Gauge block"],
            evidence={
                "raw_capture": True,
                "synchronized": True,
                "calibration": True,
                "bop_export": True,
                "bop_evaluation": True,
            },
        ),
        run_folder(
            path=disposable_run,
            size_bytes=512 * 1024**2,
            identity=delete_identity,
            sensor_name="Static RGB-D",
            mounting_mode="static",
            object_names=["Disposable cube"],
            evidence={
                "raw_capture": True,
                "synchronized": True,
                "calibration": False,
                "bop_export": False,
                "bop_evaluation": False,
            },
        ),
    ]
    inventory_runs[2]["config"] = {
        "valid": False,
        "error": "legacy configuration requires repair",
        "run_name": None,
        "sequence": None,
        "plan_only": None,
    }
    inventory_runs[0]["relocation"] = {
        "original_path": RUN_ROOT,
        "aliases": [active_run_alias],
        "history_count": 1,
    }

    page.add_init_script(
        f"""
        localStorage.setItem("posetestbot.selectedRun", {json.dumps(active_run_alias)});
        localStorage.setItem(
          "posetestbot.customRunFolders.v1",
          JSON.stringify([{json.dumps(active_run_alias)}])
        );
        """
    )

    page.unroute("**/ui/bootstrap")
    page.route(
        "**/ui/bootstrap",
        lambda route: fulfill_json(
            route,
            {
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
                "default_run_root": RUN_ROOT,
                "allowed_run_roots": [primary_root, archive_root],
            },
        ),
    )

    def operation_job(job_id: str) -> dict:
        status = job_status[job_id]
        source = movable_run if job_id == "move-run-folder" else disposable_run
        parameters = {
            "run_folder_operation": (
                "move" if job_id == "move-run-folder" else "delete"
            ),
            "source_run_root": source,
            "cancelable": False,
        }
        if job_id == "move-run-folder":
            parameters["destination_run_root"] = f"{archive_root}/archive-run"
        return {
            "id": job_id,
            "name": job_id.replace("-", " "),
            "command": ["uv", "run", "python", "scripts/manage_run_folder.py"],
            "cwd": "/repo",
            "status": status,
            "created_at": "2026-07-29T08:31:00Z",
            "started_at": ("2026-07-29T08:31:01Z" if status != "queued" else None),
            "ended_at": ("2026-07-29T08:31:02Z" if status == "succeeded" else None),
            "returncode": 0 if status == "succeeded" else None,
            "message": None,
            "tail": [],
            "resources": ["disk_io"],
            "parameters": parameters,
            "log_path": f"/tmp/jobs/{job_id}/log.txt",
            "visibility": "operator",
            "scope_kind": "run",
            "run_root": source,
        }

    def refresh_job() -> dict:
        return {
            "id": "refresh-run-folders",
            "name": "run folder inventory",
            "command": ["uv", "run", "python", "scripts/manage_run_folders.py"],
            "cwd": "/repo",
            "status": "succeeded",
            "created_at": "2026-07-29T08:29:00Z",
            "started_at": "2026-07-29T08:29:01Z",
            "ended_at": "2026-07-29T08:29:02Z",
            "returncode": 0,
            "message": None,
            "tail": [],
            "resources": ["disk_io", "run_folder_storage"],
            "parameters": {
                "run_folder_inventory": True,
                "cancelable": False,
            },
            "log_path": "/tmp/jobs/refresh-run-folders/log.txt",
            "visibility": "operator",
            "scope_kind": "global",
            "run_root": None,
        }

    def run_folders_handler(route) -> None:
        path = urlparse(route.request.url).path
        method = route.request.method
        if path == "/ui/run-folders" and method == "GET":
            move_active = move_submitted["value"] and job_status["move-run-folder"] in {
                "queued",
                "running",
                "canceling",
            }
            visible_runs = (
                inventory_runs
                if job_status["move-run-folder"] != "succeeded"
                else [run for run in inventory_runs if run["path"] != movable_run]
            )
            fulfill_json(
                route,
                {
                    "schema_version": "run_folder_inventory.v1",
                    "generated_at": (
                        None
                        if inventory_cache_missing["value"]
                        else "2026-07-29T08:30:00Z"
                    ),
                    "inventory_state": (
                        "missing"
                        if inventory_cache_missing["value"]
                        else "stale"
                        if inventory_stale["value"]
                        else "ready"
                    ),
                    "stale": (
                        inventory_cache_missing["value"] or inventory_stale["value"]
                    ),
                    "roots": [
                        {
                            "path": primary_root,
                            "exists": True,
                            "identity": {"device": 101, "inode": 1001},
                            "storage": storage(primary_root),
                        },
                        {
                            "path": archive_root,
                            "exists": True,
                            "identity": {"device": 202, "inode": 2002},
                            "storage": storage(archive_root),
                        },
                    ],
                    "runs": [] if inventory_cache_missing["value"] else visible_runs,
                    "refresh_job": None,
                    "operation_job": (
                        operation_job("move-run-folder") if move_active else None
                    ),
                    "maintenance": {
                        "schema_version": "run_folder_maintenance.v1",
                        "recovered_count": 1,
                        "transactions": [
                            {
                                "transaction_id": "a" * 32,
                                "operation": "move",
                                "action": "rolled_back_move",
                            }
                        ],
                        "unresolved_count": 0,
                        "journal_fingerprint": "no-pending-transactions",
                        "unresolved": [],
                    },
                },
            )
            return
        if path == "/ui/run-folders/refresh" and method == "POST":
            refresh_requests.append({})
            inventory_cache_missing["value"] = False
            job = refresh_job()
            fulfill_json(
                route,
                {"job_id": job["id"], "status": job["status"], "job": job},
                status=202,
            )
            return
        if path == "/ui/run-folders/move" and method == "POST":
            move_requests.append(route.request.post_data_json)
            move_submitted["value"] = True
            job = operation_job("move-run-folder")
            fulfill_json(
                route,
                {
                    "job_id": job["id"],
                    "status": job["status"],
                    "job": job,
                    "source_run_root": movable_run,
                    "destination_run_root": f"{archive_root}/archive-run",
                    "compatibility_alias": movable_run,
                },
                status=202,
            )
            return
        if path == "/ui/run-folders" and method == "DELETE":
            delete_requests.append(route.request.post_data_json)
            job = operation_job("delete-run-folder")
            fulfill_json(
                route,
                {
                    "job_id": job["id"],
                    "status": job["status"],
                    "job": job,
                    "source_run_root": disposable_run,
                },
                status=202,
            )
            return
        fulfill_json(route, {"output": "Unexpected run-folder request"}, status=404)

    page.route("**/ui/run-folders**", run_folders_handler)

    def operation_job_handler(route) -> None:
        job_id = urlparse(route.request.url).path.rsplit("/", 1)[-1]
        fulfill_json(route, {"job": operation_job(job_id)})

    page.route("**/jobs/move-run-folder", operation_job_handler)
    page.route("**/jobs/delete-run-folder", operation_job_handler)
    page.route(
        "**/jobs/refresh-run-folders",
        lambda route: fulfill_json(route, {"job": refresh_job()}),
    )

    page.goto(f"{console_server.url}/#/run-folders", wait_until="networkidle")

    expect(page).to_have_url(f"{console_server.url}/#/run-folders")
    expect(page.get_by_role("heading", name="Run folders", exact=True)).to_be_visible()
    expect(page.get_by_role("link", name="Run folders")).to_have_attribute(
        "href", "#/run-folders"
    )
    expect(page.get_by_role("button", name="Refresh inventory")).to_be_visible()
    expect(page.get_by_test_id("run-folder-root")).to_have_count(2)
    assert len(refresh_requests) == 1
    maintenance = page.get_by_test_id("run-folder-maintenance")
    expect(maintenance).to_contain_text("Interrupted storage work recovered")
    expect(maintenance).to_contain_text("Rolled Back Move")

    rows = page.get_by_test_id("run-folder-row")
    expect(rows).to_have_count(3)
    active_row = page.locator(
        f'[data-testid="run-folder-row"][data-run-path="{RUN_ROOT}"]'
    )
    expect(active_row).to_contain_text("Active run")
    expect(active_row.get_by_test_id("run-folder-active-action-reason")).to_have_text(
        "Switch the active run folder before moving or deleting this folder."
    )
    expect(active_row.get_by_role("button", name="Move new-run")).to_be_disabled()
    expect(active_row.get_by_role("button", name="Delete new-run")).to_be_disabled()

    movable_row = page.locator(
        f'[data-testid="run-folder-row"][data-run-path="{movable_run}"]'
    )
    size = movable_row.get_by_test_id("run-folder-size")
    expect(size).to_contain_text(re.compile(r"8(?:\.0+)? GiB"))
    contents = movable_row.get_by_test_id("run-folder-contents")
    expect(contents).to_contain_text("Wrist RGB-D")
    expect(contents).to_contain_text(re.compile(r"eye[ _-]in[ _-]hand", re.I))
    expect(contents).to_contain_text("Clamp")
    expect(contents).to_contain_text("Gauge block")
    expect(movable_row).to_contain_text("BOP export")
    expect(movable_row).to_contain_text("BOP evaluation")

    page.set_viewport_size({"width": 900, "height": 900})
    table = page.get_by_test_id("run-folders-table")
    expect(table).to_be_visible()
    overflow = table.evaluate(
        """element => ({
            clientWidth: element.clientWidth,
            scrollWidth: element.scrollWidth,
            overflowX: getComputedStyle(element).overflowX,
        })"""
    )
    assert overflow["overflowX"] in {"auto", "scroll"}
    assert overflow["scrollWidth"] > overflow["clientWidth"]
    table.evaluate("element => { element.scrollLeft = element.scrollWidth }")
    move_button = movable_row.get_by_role("button", name="Move archive-run")
    move_button.scroll_into_view_if_needed()
    expect(move_button).to_be_in_viewport()
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )
    page.set_viewport_size({"width": 1440, "height": 1000})

    movable_row.get_by_role("button", name="Move archive-run").click()
    move_dialog = page.get_by_test_id("run-folder-move-dialog")
    expect(move_dialog.get_by_role("heading", name="Move archive-run?")).to_be_visible()
    expect(move_dialog).to_contain_text(
        "After the move, a compatibility link at the original path keeps "
        "existing references working."
    )
    move_dialog.get_by_role("combobox", name="Destination root").click()
    page.get_by_role("option", name=archive_root).click()
    with page.expect_response(
        lambda response: urlparse(response.url).path == "/jobs/move-run-folder"
    ):
        move_dialog.get_by_role("button", name="Queue move").click()
    assert move_requests == [
        {
            "run_root": movable_run,
            "destination_root": archive_root,
            "expected_identity": move_identity,
            "expected_destination_root_identity": {
                "device": 202,
                "inode": 2002,
            },
        }
    ]
    move_status = page.get_by_test_id("run-folder-operation-status")
    expect(move_status).to_contain_text("Moving archive-run")
    expect(move_status).to_contain_text(
        "This background storage job continues after navigation and cannot be "
        "canceled safely after submission."
    )
    expect(move_status.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )

    page.reload(wait_until="networkidle")
    recovered_move_status = page.get_by_test_id("run-folder-operation-status")
    expect(recovered_move_status).to_contain_text("Moving archive-run")

    disposable_row = page.locator(
        f'[data-testid="run-folder-row"][data-run-path="{disposable_run}"]'
    )
    expect(disposable_row).to_contain_text("Invalid run configuration")
    expect(disposable_row).to_contain_text("Disposable cube")
    expect(
        disposable_row.get_by_role("button", name="Delete disposable-run")
    ).to_be_disabled()
    inventory_cache_missing["value"] = True
    job_status["move-run-folder"] = "succeeded"
    expect(
        page.locator(f'[data-testid="run-folder-row"][data-run-path="{movable_run}"]')
    ).to_have_count(0, timeout=5_000)
    assert len(refresh_requests) == 2
    expect(
        disposable_row.get_by_role("button", name="Delete disposable-run")
    ).to_be_enabled()
    disposable_row.get_by_role("button", name="Delete disposable-run").click()
    delete_dialog = page.get_by_test_id("run-folder-delete-dialog")
    expect(delete_dialog).to_have_count(1)
    expect(
        delete_dialog.get_by_role("heading", name="Delete disposable-run?")
    ).to_be_visible()
    expect(delete_dialog).to_contain_text(
        "This permanently deletes the entire run folder, including raw capture "
        "data and all derived evidence. This action cannot be undone or canceled "
        "after submission."
    )
    page.set_viewport_size({"width": 900, "height": 480})
    expect(delete_dialog).to_have_css("overflow-y", "auto")
    confirm_delete = delete_dialog.get_by_role("button", name="Confirm delete")
    confirm_delete.scroll_into_view_if_needed()
    expect(confirm_delete).to_be_in_viewport()
    assert delete_requests == []
    with page.expect_response(
        lambda response: urlparse(response.url).path == "/jobs/delete-run-folder"
    ):
        confirm_delete.click()
    assert delete_requests == [
        {
            "run_root": disposable_run,
            "confirm": True,
            "expected_identity": delete_identity,
        }
    ]
    expect(page.get_by_test_id("run-folder-delete-dialog")).to_have_count(0)
    delete_status = page.get_by_test_id("run-folder-operation-status")
    expect(delete_status).to_contain_text("Deleting disposable-run")
    expect(delete_status.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )

    inventory_stale["value"] = True
    page.reload(wait_until="networkidle")
    stale_disposable_row = page.locator(
        f'[data-testid="run-folder-row"][data-run-path="{disposable_run}"]'
    )
    expect(
        stale_disposable_row.get_by_test_id("run-folder-inventory-action-reason")
    ).to_contain_text("Wait for a current inventory")
    expect(
        stale_disposable_row.get_by_role("button", name="Move disposable-run")
    ).to_be_disabled()
    expect(
        stale_disposable_row.get_by_role("button", name="Delete disposable-run")
    ).to_be_disabled()


def test_primary_navigation_resets_document_scroll_position(
    console_server, page
) -> None:
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


def test_bop_evaluation_queues_result_recovers_job_and_shows_metrics(
    console_server,
    page,
) -> None:
    install_common_mocks(page)
    requests: list[dict] = []
    job_status: dict[str, str | None] = {"value": None}
    result = {
        "result_id": "result-aaaaaaaaaaaa",
        "method": "foundationpose",
        "display_name": "FoundationPose · trial 1",
        "filename": "foundationpose_ptb123456789abc-test.csv",
        "source_kind": "registered_result",
        "created_at": "2026-07-26T12:00:00Z",
        "sha256": "a" * 64,
        "estimate_count": 1621,
        "target_estimate_count": 1621,
        "target_coverage": 1.0,
        "compatible": True,
        "blockers": [],
        "simulation": None,
    }
    setup = {
        "toolkit": {
            "status": "ready",
            "available": True,
            "revision": "cea62d651c7e395b2e1962b9749e4e89693c6ac4",
            "required_revision": "cea62d651c7e395b2e1962b9749e4e89693c6ac4",
            "environment_ready": True,
            "renderer": "vispy",
            "install_command": None,
            "reason": None,
        },
        "dataset": {
            "status": "ready",
            "evaluation_ready": True,
            "simulation_ready": True,
            "dataset_id": "ptb123456789abc",
            "name": "PoseTestBot",
            "split": "test",
            "export_manifest_sha256": "b" * 64,
            "manifest_schema_version": "bop_export_manifest.v5",
            "scene_count": 2,
            "frame_count": 1621,
            "target_count": 1621,
            "model_count": 1,
            "annotation_count": 1621,
            "annotation_source": "blenderproc",
            "image_size": [1280, 720],
            "result_registration_ready": True,
            "result_filename_template": "{method}_ptb123456789abc-test.csv",
            "blockers": [],
            "warnings": [
                {
                    "code": "format_audit",
                    "message": "Dataset passed the structural format audit.",
                }
            ],
        },
        "results": [result],
        "evaluations": [],
    }
    page.route(
        "**/bop/evaluation/setup?**",
        lambda route: fulfill_json(route, setup),
    )

    def job_payload(status: str) -> dict:
        return {
            "id": "bopeval1",
            "name": "bop_evaluation",
            "command": ["uv", "run", "python", "scripts/run_bop_evaluation.py"],
            "cwd": "/repo",
            "status": status,
            "created_at": "2026-07-26T12:01:00Z",
            "started_at": ("2026-07-26T12:01:01Z" if status != "queued" else None),
            "ended_at": "2026-07-26T12:02:00Z" if status == "succeeded" else None,
            "returncode": 0 if status == "succeeded" else None,
            "message": "Command completed successfully."
            if status == "succeeded"
            else None,
            "tail": ["evaluating"],
            "resources": ["cpu", "disk_io"],
            "parameters": {
                "run_root": RUN_ROOT,
                "evaluation_id": "evaluation-bbbbbbbbbbbb",
                "result_id": result["result_id"],
            },
            "scope_kind": "run",
            "run_root": RUN_ROOT,
            "log_path": "/tmp/bopeval1.log",
            "visibility": "operator",
        }

    def evaluation_handler(route) -> None:
        requests.append(route.request.post_data_json)
        job_status["value"] = "queued"
        fulfill_json(
            route,
            {
                "job": job_payload("queued"),
                "job_id": "bopeval1",
                "evaluation_id": "evaluation-bbbbbbbbbbbb",
            },
            status=202,
        )

    def jobs_handler(route) -> None:
        status = job_status["value"]
        fulfill_json(
            route,
            {
                "jobs": [] if status is None else [job_payload(status)],
                "resources": {},
            },
        )

    page.route("**/bop/evaluations", evaluation_handler)
    page.route("**/jobs", jobs_handler)
    page.goto(f"{console_server.url}/#/bop-evaluation", wait_until="networkidle")

    expect(page.get_by_role("heading", name="BOP evaluation")).to_be_visible()
    dataset = page.get_by_test_id("bop-evaluation-dataset")
    expect(dataset).to_contain_text("Selected-run dataset")
    expect(dataset).to_contain_text(RUN_ROOT)
    expect(dataset).to_contain_text("1,621")
    expect(dataset).to_contain_text("Dataset passed the structural format audit.")
    csv_contract = page.get_by_test_id("bop-result-csv-contract")
    expect(csv_contract).to_contain_text("scene_id,im_id,obj_id,score,R,t,time")
    expect(csv_contract).to_contain_text("model-to-camera translation in millimetres")
    expect(csv_contract).to_contain_text("processing time per image in seconds")
    expect(page.get_by_test_id("bop-result-details")).to_contain_text(
        "FoundationPose · trial 1"
    )
    page.get_by_role("button", name="Queue BOP evaluation").click()
    expect(page.get_by_text("BOP evaluation queued")).to_be_visible()
    assert requests == [
        {
            "run_root": RUN_ROOT,
            "source": {
                "kind": "registered_result",
                "result_id": "result-aaaaaaaaaaaa",
            },
        }
    ]
    job = page.get_by_test_id("bop-evaluation-job-status")
    expect(job).to_contain_text("BOP evaluation is queued")
    expect(job).to_contain_text("continues after navigation")
    expect(job.get_by_role("link", name="Open live log in Jobs")).to_have_attribute(
        "href", "#/jobs"
    )

    job_status["value"] = "running"
    expect(job).to_contain_text("BOP evaluation is running", timeout=5_000)
    setup["evaluations"] = [
        {
            "evaluation_id": "evaluation-bbbbbbbbbbbb",
            "created_at": "2026-07-26T12:01:00Z",
            "completed_at": "2026-07-26T12:02:00Z",
            "result_id": result["result_id"],
            "result": result,
            "source_kind": "registered_result",
            "simulation": None,
            "protocol": "BOP19 localization",
            "status": "succeeded",
            "metrics": [
                {
                    "id": "bop19_average_recall",
                    "label": "Average Recall",
                    "value": 0.9876,
                    "display": "0.9876",
                },
                {
                    "id": "bop19_average_recall_mssd",
                    "label": "AR MSSD",
                    "value": 0.9821,
                    "display": "0.9821",
                },
            ],
            "provenance": {
                "toolkit_revision": "cea62d651c7e395b2e1962b9749e4e89693c6ac4"
            },
            "report_available": True,
        }
    ]
    job_status["value"] = "succeeded"
    report = page.get_by_test_id("bop-evaluation-report")
    expect(report).to_contain_text("Official BOP metrics", timeout=5_000)
    expect(report).to_contain_text("0.9876")
    expect(report).to_contain_text("AR MSSD")
    expect(page.get_by_test_id("bop-evaluation-history")).to_contain_text(
        "FoundationPose · trial 1"
    )

    page.get_by_role("tab", name="Simulated from GT · Test only").click()
    expect(
        page.get_by_text(
            "Test only: estimates are derived from ground truth",
            exact=True,
        )
    ).to_be_visible()
    expect(page.locator("#translation-sigma")).to_have_value("1")
    expect(page.locator("#rotation-sigma")).to_have_value("0.25")
    expect(page.locator("#simulation-seed")).to_have_value("42")
    expect(page.locator("#simulation-score")).to_have_value("1")


def test_workflow_chooser_distinguishes_numbered_guided_journeys(
    console_server,
    page,
) -> None:
    install_common_mocks(page)

    page.goto(f"{console_server.url}/#/workflow/setup", wait_until="networkidle")

    expect(page.get_by_role("heading", name="What do you want to do?")).to_be_visible()
    expect(page.get_by_role("heading", name="Calibrate cameras")).to_be_visible()
    expect(page.get_by_role("heading", name="Record an object dataset")).to_be_visible()
    expect(
        page.get_by_text("Each guided workflow shows the required order", exact=False)
    ).to_be_visible()

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
        "02Choose the pose template and placement",
        "03Check readiness",
        "04Record the object dataset",
        "05Process frames and create the base BOP export",
        "06Add optional BOP ground-truth evidence",
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

    stepper = page.get_by_role("navigation", name="Workflow steps")
    expect(stepper).to_be_visible()
    expect(page).to_have_url(f"{console_server.url}/#/workflow/calibration?step=target")
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
        number_box = (
            steps.nth(index).locator("[data-workflow-step-number]").bounding_box()
        )
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


def test_sidebar_preserves_current_workflow_step_and_fast_return(
    console_server,
    page,
) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    capture_runtime: dict[str, str | None] = {"status": None}
    bundle_sha256 = "d" * 64
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
            }
        ]
    )
    configured["dataset_mode"] = "pose_template"
    configured["calibration_profiles"] = (
        "processed/calibration_inputs/current/calibration_profiles.json"
    )
    configured["intrinsic_calibration_profiles"] = (
        "processed/calibration_inputs/current/intrinsic_calibration_profiles.json"
    )
    configured["calibration_profile_selection"] = {
        "selection_artifact": "calibration_profile_selection.json",
        "bundle_sha256": bundle_sha256,
        "selected_at": "2026-07-22T12:00:00+00:00",
    }
    configured["pose_template"] = {
        "template_uuid": "22222222-2222-4222-8222-222222222222",
        "placement_confirmed": True,
    }

    install_common_mocks(page, config_payload=configured)
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {
                "selected": valid_library_selection(
                    bundle_sha256=bundle_sha256,
                    calibration_profiles=configured["calibration_profiles"],
                    intrinsic_calibration_profiles=configured[
                        "intrinsic_calibration_profiles"
                    ],
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
            {"selection": None, "replacement_blockers": [], "ready": True},
        ),
    )

    def capture_jobs_handler(route) -> None:
        status = capture_runtime["status"]
        fulfill_json(
            route,
            {
                "run_root": RUN_ROOT,
                "jobs": (
                    []
                    if status is None
                    else [
                        {
                            "id": "dataset-capture-1",
                            "name": "Dataset capture",
                            "status": status,
                            "kind": "pipeline_sequence",
                            "stage": None,
                            "sequence": "real_full_capture_validation",
                            "run_root": RUN_ROOT,
                            "resources": ["cameras", "robot", "disk_io"],
                            "message": None,
                            "created_at": "2026-07-26T12:00:00Z",
                            "started_at": "2026-07-26T12:00:01Z",
                            "ended_at": None,
                            "active": True,
                            "tail": ["recording"],
                            "log_endpoint": "/capture/jobs/dataset-capture-1/log",
                            "stop_endpoint": "/capture/jobs/dataset-capture-1/stop",
                        }
                    ]
                ),
                "active_count": 0 if status is None else 1,
                "resources": {},
                "status_artifact": None,
            },
        )

    page.route("**/capture/jobs**", capture_jobs_handler)

    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=capture",
        wait_until="networkidle",
    )

    sidebar = page.get_by_role("complementary", name="Application sidebar")
    current = sidebar.get_by_test_id("current-workflow-card")
    expect(current).to_contain_text("Resume position")
    expect(current).to_contain_text("Object dataset")
    expect(current).to_contain_text("Active run · Viewed step 4 of 6")
    expect(current).to_contain_text("Record the object dataset")
    expect(current).to_contain_text("Current step")
    resume = current.get_by_role(
        "link",
        name="Resume object dataset at step 4: Record the object dataset",
    )
    expect(resume).to_be_in_viewport()
    expect(resume).to_have_attribute("href", "#/workflow/dataset?step=capture")
    expect(
        sidebar.get_by_role("navigation", name="Primary navigation").get_by_role(
            "link", name="Workflow"
        )
    ).to_have_attribute("href", "#/workflow/dataset?step=capture")

    stored = json.loads(
        page.evaluate("localStorage.getItem('posetestbot.workflowSessions.v1')")
    )
    assert stored[RUN_ROOT]["journey"] == "dataset"
    assert stored[RUN_ROOT]["stepId"] == "capture"
    assert stored[RUN_ROOT]["status"] == "current"

    sidebar.get_by_role("link", name="Devices").click()
    expect(page).to_have_url(f"{console_server.url}/#/devices")
    device_handoff = page.get_by_role(
        "complementary", name="Where this page fits in the operator workflow"
    )
    expect(
        device_handoff.get_by_role("link", name="Open workflow step 1")
    ).to_have_attribute("href", "#/workflow/dataset?step=configure")

    sidebar.get_by_role("link", name="Jobs").click()
    expect(page).to_have_url(f"{console_server.url}/#/jobs")
    jobs_handoff = page.get_by_role(
        "complementary", name="Where this page fits in the operator workflow"
    )
    expect(jobs_handoff.get_by_role("link", name="Open workflow")).to_have_attribute(
        "href", "#/workflow/dataset?step=capture"
    )

    sidebar.get_by_role("link", name="Dashboard").click()
    expect(page).to_have_url(f"{console_server.url}/#/dashboard")
    capture_runtime["status"] = "running"
    page.reload(wait_until="networkidle")
    current = page.get_by_role(
        "complementary", name="Application sidebar"
    ).get_by_test_id("current-workflow-card")
    expect(current).to_contain_text("Active run · Viewed step 4 of 6")
    expect(current.get_by_role("status")).to_have_text("Recording running")
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    sidebar = page.get_by_role("complementary", name="Application sidebar")
    expect(sidebar.get_by_test_id("current-workflow-card")).to_have_count(0)
    expect(
        sidebar.get_by_role("navigation", name="Primary navigation").get_by_role(
            "link", name="Workflow"
        )
    ).to_have_attribute("href", "#/workflow/setup")
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="new-run · real_full_capture_validation").click()
    current = sidebar.get_by_test_id("current-workflow-card")
    expect(current).to_contain_text("Active run · Viewed step 4 of 6")
    current.get_by_role(
        "link",
        name="Resume object dataset at step 4: Record the object dataset",
    ).click()
    expect(page).to_have_url(f"{console_server.url}/#/workflow/dataset?step=capture")
    expect(
        page.get_by_role("navigation", name="Workflow steps").locator(
            '[aria-current="step"]'
        )
    ).to_contain_text("Record the object dataset")
    active_capture = page.get_by_test_id("capture-active-job")
    expect(active_capture).to_contain_text("Dataset capture is running")
    expect(active_capture).to_contain_text("continues after navigation")
    expect(active_capture).to_contain_text(
        "Another capture cannot be submitted while this job is active."
    )
    expect(
        active_capture.get_by_role("link", name="Open capture in Jobs")
    ).to_have_attribute("href", "#/jobs")
    expect(
        page.get_by_role("button", name="Review and start capture", exact=True)
    ).to_have_count(0)


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
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="Create or open a run folder…").click()
    page.locator("#new-run-name").fill("fresh-calibration-run")
    page.get_by_role("button", name="Use run folder", exact=True).click()
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    expect(page.get_by_role("heading", name="Calibrate cameras")).to_be_visible()
    setup = page.get_by_test_id("camera_calibration-run-setup")
    expect(setup).to_be_visible()
    expect(setup.get_by_test_id("run-camera-row")).to_have_count(2)
    expect(setup.get_by_role("button", name="Save setup")).to_be_enabled()
    expect(page.get_by_role("navigation", name="Workflow steps")).to_be_visible()
    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        fresh_run
    )


def test_switching_active_run_clears_setup_draft_before_unconfigured_save(
    console_server,
    page,
) -> None:
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Wrist RGB-D",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
            }
        ]
    )
    install_common_mocks(page, config_payload=configured)
    page.route(
        "**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status())
    )
    fresh_run = "/tmp/posetestbot-console/empty-new-run"
    unavailable_run = "/tmp/posetestbot-console/unavailable-run"
    writes: list[dict] = []

    def config_handler(route) -> None:
        run_root = parse_qs(urlparse(route.request.url).query).get(
            "run_root", [RUN_ROOT]
        )[0]
        if route.request.method == "POST":
            writes.append(route.request.post_data_json)
            fulfill_json(route, {"config": configured}, status=201)
        elif run_root == fresh_run:
            fulfill_json(
                route,
                {"output": "run_config.json does not exist"},
                status=404,
            )
        elif run_root == unavailable_run:
            fulfill_json(
                route,
                {"output": "run configuration storage is temporarily unavailable"},
                status=503,
            )
        else:
            fulfill_json(
                route,
                {
                    "config": configured,
                    "preflight": {"queue_blocker": "missing_preflight"},
                },
            )

    page.unroute("**/run-config**")
    page.route("**/run-config**", config_handler)
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    page.get_by_label("Run name").fill("unsaved values from the previous run")
    page.locator("#fps").fill("41")
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="Create or open a run folder…").click()
    page.locator("#new-run-name").fill("empty-new-run")
    page.get_by_role("button", name="Use run folder", exact=True).click()

    expect(page.get_by_role("combobox", name="Active run folder")).to_contain_text(
        fresh_run
    )
    setup = page.get_by_test_id("camera_calibration-run-setup")
    expect(setup.get_by_label("Run name")).to_have_value("")
    expect(setup.locator("#fps")).to_have_value("6")
    expect(setup.locator("#velocity")).to_have_value("0.01")
    expect(setup.get_by_role("button", name="Save setup")).to_be_enabled()

    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="Create or open a run folder…").click()
    page.locator("#new-run-name").fill("unavailable-run")
    page.get_by_role("button", name="Use run folder", exact=True).click()
    setup = page.get_by_test_id("camera_calibration-run-setup")
    expect(
        setup.get_by_text("The active run’s setup could not be loaded")
    ).to_be_visible()
    expect(setup).to_contain_text(
        "Existing setup may still be present, so saving remains disabled."
    )
    expect(setup.get_by_role("button", name="Retry setup lookup")).to_be_visible()
    expect(setup.get_by_role("button", name="Save setup")).to_be_disabled()
    expect(setup.get_by_label("Run name")).to_be_disabled()
    assert writes == []


def test_responsive_shell_and_dataset_workflow_links(console_server, page) -> None:
    config = run_config(plan_only=False)
    config["dataset_mode"] = "pose_template"
    config["calibration_profiles"] = (
        "processed/calibration_inputs/current/calibration_profiles.json"
    )
    config["intrinsic_calibration_profiles"] = (
        "processed/calibration_inputs/current/intrinsic_calibration_profiles.json"
    )
    config["calibration_profile_selection"] = {
        "selection_artifact": "calibration_profile_selection.json",
        "bundle_sha256": "a" * 64,
        "selected_at": "2026-07-27T12:00:00+00:00",
    }
    config["pose_template"] = {
        "template_uuid": "22222222-2222-4222-8222-222222222222",
        "placement_confirmed": True,
    }
    overview = overview_payload(config)
    for section_id in ("preflight", "capture", "sync"):
        next(section for section in overview["sidebar"] if section["id"] == section_id)[
            "status"
        ] = "complete"
    next(section for section in overview["sidebar"] if section["id"] == "bop")[
        "status"
    ] = "blocked"
    install_common_mocks(page, config_payload=config)
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview))
    page.set_viewport_size({"width": 900, "height": 900})

    page.goto(f"{console_server.url}/#/dashboard", wait_until="networkidle")

    expect(page.locator("aside")).to_be_hidden()
    primary_navigation = page.get_by_role("navigation", name="Primary navigation")
    expect(primary_navigation).to_have_count(1)
    expect(primary_navigation).to_be_visible()
    expect(
        primary_navigation.get_by_role("link", name="Calibration Targets")
    ).to_be_visible()
    expect(
        primary_navigation.get_by_role("link", name="Pose Templates")
    ).to_be_visible()
    expect(
        page.get_by_role(
            "link",
            name="Open object dataset step 1: Configure cameras and select calibration",
        )
    ).to_have_attribute("href", "#/workflow/dataset?step=configure")
    workflow_overview = page.get_by_test_id("dashboard-workflow-overview")
    expect(workflow_overview).to_have_attribute("data-workflow-journey", "dataset")
    expect(
        workflow_overview.get_by_role("heading", name="Object dataset workflow")
    ).to_be_visible()
    expect(workflow_overview).to_contain_text(
        "5 required steps plus 1 optional ground-truth step"
    )
    expect(workflow_overview).to_contain_text(
        "a saved camera calibration is an input to step 1"
    )
    expect(workflow_overview.locator("[data-workflow-step]")).to_have_count(6)
    expect(
        workflow_overview.locator('[data-workflow-step="template"]')
    ).to_contain_text("Choose the pose template and placement")
    expect(workflow_overview.locator('[data-workflow-step="export"]')).to_contain_text(
        "Add optional BOP ground-truth evidence"
    )
    expect(workflow_overview.locator('[data-workflow-step="export"]')).to_contain_text(
        "Optional"
    )
    expect(workflow_overview.locator('[data-workflow-step="sync"]')).to_contain_text(
        "Blocked"
    )
    expect(
        workflow_overview.locator('[data-workflow-step="export"]')
    ).not_to_contain_text("Blocked")
    expect(workflow_overview.get_by_text("Needs attention")).to_have_count(0)
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
            "limits": {
                "cad_bytes": 52428800,
                "batch_bytes": 104857600,
                "faces": 1000000,
                "contour_vertices": 10000,
                "instances": 200,
            },
        },
    }


def pose_template_catalog() -> dict:
    return {
        "schema_version": "object_catalog.v1",
        "objects": [
            {
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
                "extraction": {
                    "vertices": 8,
                    "faces": 12,
                    "bounds_mm": [[-5, -5, -5], [5, 5, 5]],
                    "watertight": True,
                },
                "assets": {
                    "source": {
                        "path": "objects/1/source/clamp.stl",
                        "sha256": "a" * 64,
                        "size_bytes": 100,
                        "media_type": "application/octet-stream",
                    },
                    "canonical_ply": {
                        "path": "objects/1/derived/canonical.ply",
                        "sha256": "b" * 64,
                        "size_bytes": 80,
                        "media_type": "application/octet-stream",
                    },
                    "texture": {
                        "path": "objects/1/texture/texture.png",
                        "sha256": "c" * 64,
                        "size_bytes": 40,
                        "media_type": "image/png",
                    },
                },
                "usage": {"template_count": 0, "templates": []},
            }
        ],
    }


def workpiece_catalog() -> dict:
    value = pose_template_catalog()
    value.update(
        {
            "version": 4,
            "created_at": "2026-07-20T09:00:00Z",
            "updated_at": "2026-07-21T11:00:00Z",
            "next_obj_id": 9,
            "tombstones": [],
        }
    )
    value["objects"].append(
        {
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
            "extraction": {
                "vertices": 8,
                "faces": 12,
                "bounds_mm": [[-12.5, -5, -2.5], [12.5, 5, 2.5]],
                "watertight": True,
            },
            "assets": {
                "source": {
                    "path": "objects/8/source/gauge.ply",
                    "sha256": "d" * 64,
                    "size_bytes": 120,
                    "media_type": "application/octet-stream",
                },
                "canonical_ply": {
                    "path": "objects/8/derived/canonical.ply",
                    "sha256": "e" * 64,
                    "size_bytes": 90,
                    "media_type": "application/octet-stream",
                },
            },
            "usage": {"template_count": 0, "templates": []},
        }
    )
    return value


def pose_template_library() -> dict:
    return {
        "schema_version": "pose_template_library.v1",
        "templates": [
            {
                "template_uuid": "22222222-2222-4222-8222-222222222222",
                "display_name": "Clamp pair",
                "description": "fixture",
                "created_at": "2026-07-20T10:00:00Z",
                "bundle_sha256": "d" * 64,
                "archive": {"state": "active"},
                "page": {
                    "size": "A3",
                    "orientation": "landscape",
                    "width_mm": 420,
                    "height_mm": 297,
                },
                "instances": [
                    {
                        "instance_uuid": "33333333-3333-4333-8333-333333333333",
                        "catalog_uuid": "11111111-1111-4111-8111-111111111111",
                        "catalog": {
                            "catalog_uuid": "11111111-1111-4111-8111-111111111111",
                            "name": "Clamp",
                            "obj_id": 7,
                        },
                        "pose_template_from_object": {
                            "matrix": [
                                [1, 0, 0, 45],
                                [0, 1, 0, 55],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                        },
                    }
                ],
            }
        ],
    }


def pose_template_orientation_analysis(
    catalog_uuid: str = "11111111-1111-4111-8111-111111111111",
) -> dict:
    return {
        "schema_version": "pose_template_orientation_analysis.v1",
        "catalog_uuid": catalog_uuid,
        "preview_mesh": {
            "vertices": [
                [-10, -5, 0],
                [10, -5, 0],
                [10, 5, 0],
                [-10, 5, 0],
                [0, 0, 12],
            ],
            "faces": [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]],
        },
        "recognition_mesh": {
            "vertices": [
                [-10, -5, 0],
                [10, -5, 0],
                [10, 5, 0],
                [-10, 5, 0],
                [-10, -5, 12],
                [10, -5, 12],
                [10, 5, 12],
                [-10, 5, 12],
            ],
            "faces": [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
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
                "source_to_placed": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "slice_z_mm": 0.1,
                "contours": [
                    {
                        "points": [
                            {"x_mm": -10, "y_mm": -5},
                            {"x_mm": 10, "y_mm": -5},
                            {"x_mm": 7, "y_mm": 5},
                            {"x_mm": -10, "y_mm": 3},
                        ]
                    }
                ],
            },
            {
                "orientation_id": "stable-side",
                "label": "Side base",
                "probability": 0.18,
                "source_to_placed": [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 5],
                    [0, 0, 0, 1],
                ],
                "slice_z_mm": 0.1,
                "contours": [
                    {
                        "points": [
                            {"x_mm": -10, "y_mm": -6},
                            {"x_mm": 10, "y_mm": -6},
                            {"x_mm": 10, "y_mm": 6},
                            {"x_mm": -10, "y_mm": 6},
                        ]
                    }
                ],
            },
        ],
    }


def pose_template_orientation_thumbnail(
    catalog_uuid: str = "11111111-1111-4111-8111-111111111111",
) -> dict:
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
        "instances": [
            {
                "instance_uuid": "33333333-3333-4333-8333-333333333333",
                "catalog_uuid": "11111111-1111-4111-8111-111111111111",
                "catalog": {"name": "Clamp", "obj_id": 7},
                "pose_template_from_object": {
                    "matrix": [[1, 0, 0, 45], [0, 1, 0, 55], [0, 0, 1, 0], [0, 0, 0, 1]]
                },
                "preview_mesh_sha256": "b" * 64,
                "compensated_contours": [
                    [
                        {"x_mm": 30, "y_mm": 30},
                        {"x_mm": 50, "y_mm": 30},
                        {"x_mm": 47, "y_mm": 42},
                        {"x_mm": 30, "y_mm": 40},
                    ],
                    [
                        {"x_mm": 36, "y_mm": 34},
                        {"x_mm": 40, "y_mm": 34},
                        {"x_mm": 40, "y_mm": 37},
                        {"x_mm": 36, "y_mm": 37},
                    ],
                ],
            }
        ],
        "preview_meshes": {
            "b" * 64: {
                "vertices": [
                    [-10, -5, 0],
                    [10, -5, 0],
                    [10, 5, 0],
                    [-10, 5, 0],
                    [0, 0, 12],
                ],
                "faces": [
                    [0, 1, 4],
                    [1, 2, 4],
                    [2, 3, 4],
                    [3, 0, 4],
                    [0, 3, 2],
                    [0, 2, 1],
                ],
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
        "instances": [
            {
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
            }
        ],
        "approximation": {
            "approximate": False,
            "truncated": False,
            "strategy": "largest-primary-then-round-robin-even-decimation",
            "source_contours": len(contours),
            "included_contours": len(contours),
            "source_points": point_count,
            "included_points": point_count,
            "limits": {
                "instances": 200,
                "contours": 400,
                "points": 4096,
                "points_per_contour": 48,
            },
        },
    }


def test_pose_templates_editor_catalog_generation_and_unavailable_browse(
    console_server, page
) -> None:
    install_common_mocks(page)
    page.add_init_script(
        "Object.defineProperty(Crypto.prototype, 'randomUUID', { value: undefined, configurable: true })"
    )
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    requests: list[dict] = []
    preview_posts = {"count": 0}
    availability = {"available": True}
    orientation_ready = {"value": False}
    library_job_status = {"generate": "queued", "clone": "queued"}
    library_payload = pose_template_library()
    page.route(
        "**/pose-templates/status",
        lambda route: fulfill_json(
            route, pose_template_source(available=availability["available"])
        ),
    )
    page.route(
        "**/workpieces/catalog",
        lambda route: fulfill_json(route, pose_template_catalog()),
    )
    page.route(
        "**/pose-templates/library", lambda route: fulfill_json(route, library_payload)
    )

    def orientation_handler(route) -> None:
        if route.request.method == "POST":
            orientation_ready["value"] = True
            fulfill_json(
                route, {"job_id": "orientation-job", "request_id": "d" * 32}, status=202
            )
        elif orientation_ready["value"]:
            fulfill_json(route, pose_template_orientation_analysis())
        else:
            fulfill_json(
                route,
                {
                    "output": "Cached orientation analysis is stale",
                    "analysis_required": True,
                },
                status=409,
            )

    page.route("**/pose-templates/workpieces/*/orientations", orientation_handler)
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: (
            fulfill_json(route, pose_template_orientation_thumbnail())
            if orientation_ready["value"]
            else fulfill_json(
                route,
                {
                    "output": "Orientation thumbnail unavailable",
                    "analysis_required": True,
                },
                status=404,
            )
        ),
    )
    page.route(
        "**/pose-templates/library/*/preview",
        lambda route: fulfill_json(route, immutable_template_preview()),
    )
    page.route(
        "**/pose-templates/library/*/thumbnail",
        lambda route: fulfill_json(route, immutable_template_thumbnail()),
    )
    page.route(
        "**/jobs/generate-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "generate-job",
                    "status": library_job_status["generate"],
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )
    page.route(
        "**/jobs/clone-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "clone-job",
                    "status": library_job_status["clone"],
                    "message": (
                        "Command exited with status 1"
                        if library_job_status["clone"] == "failed"
                        else None
                    ),
                    "tail": (
                        [
                            "Canonical geometry changed; analyze stable orientations again.",
                            "Command exited with code 1",
                        ]
                        if library_job_status["clone"] == "failed"
                        else []
                    ),
                }
            },
        ),
    )
    page.route(
        "**/jobs/orientation-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "orientation-job",
                    "status": "succeeded",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )
    page.route(
        "**/jobs/delete-cleanup-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "delete-cleanup-job",
                    "status": "running",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )

    def preview_handler(route) -> None:
        if route.request.method == "POST":
            preview_posts["count"] += 1
            if preview_posts["count"] == 1:
                fulfill_json(
                    route, {"output": "Resources busy: cpu, disk_io"}, status=409
                )
                return
            requests.append(
                {
                    "path": "/pose-templates/preview",
                    "body": route.request.post_data_json,
                }
            )
            fulfill_json(
                route, {"job_id": "preview-job", "request_id": "a" * 32}, status=202
            )
        else:
            fulfill_json(route, immutable_template_preview())

    page.route("**/pose-templates/preview**", preview_handler)
    page.route(
        "**/pose-templates/generate",
        lambda route: (
            requests.append(
                {
                    "path": "/pose-templates/generate",
                    "body": route.request.post_data_json,
                }
            ),
            fulfill_json(route, {"job_id": "generate-job"}, status=202),
        )[1],
    )
    page.route(
        "**/pose-templates/library/*/clone",
        lambda route: (
            requests.append({"path": "/library/clone", "body": {}}),
            fulfill_json(route, {"job_id": "clone-job"}, status=202),
        )[1],
    )

    def template_delete_handler(route) -> None:
        requests.append(
            {
                "path": "/library/delete",
                "method": route.request.method,
                "body": route.request.post_data_json,
            }
        )
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
    for authoring_step in (
        "Template authoring · 1 of 3",
        "Template authoring · 2 of 3",
        "Template authoring · 3 of 3",
    ):
        expect(page.get_by_text(authoring_step, exact=True)).to_be_visible()
    expect(page.get_by_role("link", name="PDF")).to_have_class(
        re.compile(r"\bborder\b.*\bbg-card\b")
    )
    expect(page.get_by_role("button", name="Upload CAD")).to_have_count(0)
    library_thumbnail = page.get_by_test_id(
        "template-thumbnail-22222222-2222-4222-8222-222222222222"
    )
    expect(library_thumbnail.locator("path")).to_have_count(1)
    expect(library_thumbnail.locator("path")).to_have_attribute("fill-rule", "evenodd")
    expect(
        library_thumbnail.locator('g[transform="translate(0 297) scale(1 -1)"]')
    ).to_have_count(1)
    page.get_by_role("textbox", name="Filter template workpieces").fill(
        "no such object"
    )
    expect(
        page.get_by_text("No active workpieces match these filters.")
    ).to_be_visible()
    page.get_by_role("textbox", name="Filter template workpieces").fill("small clamp")
    expect(page.get_by_text("Clamp", exact=True)).to_be_visible()
    page.get_by_role("textbox", name="Filter template workpieces").fill("")
    expect(page.get_by_label("X print %")).to_have_value("100")
    expect(page.get_by_label("Y print %")).to_have_value("100")
    assert (
        page.get_by_test_id("pose-template-preview-canvas").evaluate(
            "element => getComputedStyle(element).backgroundColor"
        )
        != "rgb(255, 255, 255)"
    )
    page.get_by_role("button", name="Choose orientation for Clamp").click()
    chooser = page.get_by_test_id("orientation-chooser")
    expect(chooser).to_contain_text("same-scale high-detail recognition surface")
    expect(chooser).to_contain_text("tiny printable-layout proxy is not used")
    wide_slice = chooser.get_by_role(
        "img", name="Wide base exact selected slice contour"
    )
    expect(wide_slice.locator("path")).to_have_attribute("fill-rule", "evenodd")
    expect(wide_slice.locator("path")).to_have_attribute(
        "transform", "translate(0 0) scale(1 -1)"
    )
    expect(
        page.get_by_test_id(
            "workpiece-isometric-11111111-1111-4111-8111-111111111111"
        ).locator("polygon")
    ).to_have_count(12)
    wide_preview = page.get_by_test_id("orientation-isometric-stable-wide")
    expect(wide_preview.locator("polygon")).to_have_count(12)
    expect(wide_preview.locator("xpath=..")).to_have_attribute(
        "data-preview-quality", "recognition"
    )
    expect(
        page.get_by_test_id("orientation-preview-quality-stable-wide")
    ).to_contain_text("Full recognition surface · 12 of 12 source faces")
    wide_points = wide_preview.locator("polygon").first.get_attribute("points")
    side_points = (
        page.get_by_test_id("orientation-isometric-stable-side")
        .locator("polygon")
        .first.get_attribute("points")
    )
    assert wide_points != side_points
    chooser.get_by_role("radio").filter(has_text="Side base").click()
    chooser.get_by_role("button", name="Add selected orientation").click()
    expect(page.get_by_label("Clamp X mm")).to_be_visible()
    expect(
        page.get_by_test_id("selected-instance-isometric").locator("polygon")
    ).to_have_count(12)
    assert page_errors == []
    page.get_by_label("Clamp Rotation °").fill("27.5")
    page.get_by_label("X print %").fill("101")
    expect(page.get_by_role("button", name="Generate immutable version")).to_be_enabled(
        timeout=15_000
    )
    assert preview_posts["count"] >= 2
    page.get_by_role("button", name="Generate immutable version").click()
    expect(page.get_by_text("Immutable template generation queued")).to_be_visible()
    library_job = page.get_by_test_id("pose-template-library-job")
    expect(library_job).to_contain_text("continues after navigation")
    expect(library_job.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    assert (
        requests[-1]["body"]["configuration"]["instances"][0]["orientation_id"]
        == "stable-side"
    )
    assert (
        requests[-1]["body"]["configuration"]["instances"][0]["pose"]["rotation_deg"]
        == 27.5
    )
    library_job_status["generate"] = "succeeded"
    expect(library_job).to_have_count(0, timeout=5_000)
    expect(page.get_by_role("button", name="Clone")).to_be_enabled(timeout=15_000)
    page.get_by_role("button", name="Clone").click()
    library_job = page.get_by_test_id("pose-template-library-job")
    expect(library_job).to_contain_text("continues after navigation")
    expect(library_job.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    library_job_status["clone"] = "failed"
    expect(
        page.get_by_text(
            "Canonical geometry changed; analyze stable orientations again."
        )
    ).to_be_visible(timeout=15_000)
    assert {item["path"] for item in requests} >= {
        "/pose-templates/generate",
        "/library/clone",
    }
    generation = next(
        item
        for item in reversed(requests)
        if item["path"] == "/pose-templates/generate"
    )
    assert generation["body"]["configuration"]["print_compensation"]["x_scale"] == 1.01
    page.get_by_role("button", name="Delete Clamp pair").click()
    deletion = page.get_by_test_id("pose-template-delete-confirmation")
    expect(deletion).to_contain_text("library entry is removed immediately")
    expect(deletion).to_contain_text("continues after navigation")
    expect(deletion).to_contain_text("Existing run-owned snapshots remain intact")
    deletion.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Pose template deleted")).to_be_visible()
    expect(
        page.get_by_test_id(
            "pose-template-library-card-22222222-2222-4222-8222-222222222222"
        )
    ).to_have_count(0)
    cleanup = page.get_by_test_id("pose-template-cleanup-job")
    expect(cleanup).to_contain_text("cleanup continues after navigation")
    expect(cleanup.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    delete_request = next(
        item for item in requests if item["path"] == "/library/delete"
    )
    assert delete_request == {
        "path": "/library/delete",
        "method": "DELETE",
        "body": {"confirm": True},
    }

    availability["available"] = False
    page.reload(wait_until="networkidle")
    expect(page.get_by_test_id("pose-template-disabled-action-reason")).to_have_text(
        "PoseTemplateCreator checkout is missing"
    )
    expect(
        page.get_by_test_id("pose-template-generation-disabled-reason")
    ).to_have_text("PoseTemplateCreator checkout is missing")
    expect(
        page.get_by_text("bash scripts/install.sh --with-posetemplatecreator")
    ).to_be_visible()
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
    background_job_status = {"upload": "queued", "correction": "queued"}
    requests: list[dict] = []

    def item(catalog_uuid: str) -> dict:
        return next(
            value
            for value in catalogue["objects"]
            if value["catalog_uuid"] == catalog_uuid
        )

    def status_handler(route) -> None:
        active = sum(value["state"] == "active" for value in catalogue["objects"])
        fulfill_json(
            route,
            {
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
            },
        )

    def catalog_handler(route) -> None:
        request = route.request
        path = urlparse(request.url).path
        if path == "/workpieces/catalog" and request.method == "GET":
            catalog_requests["count"] += 1
            fulfill_json(route, catalogue)
            return
        if path == "/workpieces/catalog/import" and request.method == "POST":
            requests.append(
                {
                    "path": path,
                    "method": request.method,
                    "body": request.post_data or "",
                }
            )
            fulfill_json(
                route,
                {
                    "schema_version": "workpiece_catalog_import.v1",
                    "updated": [catalogue["objects"][0]["catalog_uuid"]],
                    "unchanged": [catalogue["objects"][1]["catalog_uuid"]],
                    "skipped_missing_assets": [],
                },
            )
            return
        if path == "/workpieces/catalog/upload" and request.method == "POST":
            requests.append(
                {
                    "path": path,
                    "method": request.method,
                    "body": request.post_data or "",
                }
            )
            catalogue["objects"].append(
                {
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
                    "extraction": {
                        "vertices": 8,
                        "faces": 12,
                        "bounds_mm": [[-4, -4, -4], [4, 4, 4]],
                        "watertight": True,
                    },
                    "assets": {
                        "source": {
                            "path": "objects/9/source/new-clamp.stl",
                            "sha256": "f" * 64,
                        },
                        "canonical_ply": {
                            "path": "objects/9/derived/canonical.ply",
                            "sha256": "1" * 64,
                        },
                    },
                    "usage": {"template_count": 0, "templates": []},
                }
            )
            fulfill_json(
                route,
                {"job_id": "workpiece-upload-job", "request_id": "a" * 32},
                status=202,
            )
            return
        parts = path.removeprefix("/workpieces/catalog/").split("/")
        catalog_uuid = parts[0]
        current = item(catalog_uuid)
        if (
            len(parts) == 2
            and parts[1] == "unit-corrections"
            and request.method == "POST"
        ):
            body = request.post_data_json
            requests.append({"path": path, "method": request.method, "body": body})
            current["geometry_revision"] = 2
            current["source_to_mm_scale"] = (
                0.001 if body["conversion"] == "millimeter_to_meter" else 1000.0
            )
            current["canonical_ply_sha256"] = "2" * 64
            factor = 0.001 if body["conversion"] == "millimeter_to_meter" else 1000.0
            current["extraction"]["bounds_mm"] = [
                [coordinate * factor for coordinate in corner]
                for corner in current["extraction"]["bounds_mm"]
            ]
            fulfill_json(
                route,
                {"job_id": "unit-correction-job", "request_id": "b" * 32},
                status=202,
            )
            return
        if len(parts) == 1 and request.method == "PATCH":
            body = request.post_data_json
            requests.append({"path": path, "method": request.method, "body": body})
            current.update(body)
            current["updated_at"] = "2026-07-22T12:30:00Z"
            fulfill_json(route, current)
            return
        if (
            len(parts) == 2
            and parts[1] in {"archive", "restore"}
            and request.method == "POST"
        ):
            requests.append({"path": path, "method": request.method, "body": None})
            current["state"] = "archived" if parts[1] == "archive" else "active"
            current["archived_at"] = (
                "2026-07-22T12:45:00Z" if parts[1] == "archive" else None
            )
            fulfill_json(route, current)
            return
        if len(parts) == 1 and request.method == "DELETE":
            requests.append(
                {"path": path, "method": request.method, "body": request.post_data_json}
            )
            delete_requests["count"] += 1
            if delete_requests["count"] == 1:
                fulfill_json(
                    route,
                    {
                        "output": "Workpiece is referenced by or cannot be checked against pose-template bundles",
                        "blockers": [
                            {
                                "template_uuid": "22222222-2222-4222-8222-222222222222",
                                "display_name": "Clamp pair",
                                "state": "active",
                                "reason": "catalog_reference",
                            }
                        ],
                    },
                    status=409,
                )
                return
            catalogue["objects"].remove(current)
            fulfill_json(
                route,
                {"schema_version": "workpiece_catalog_delete.v1", "status": "deleted"},
            )
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
    page.route(
        "**/jobs/workpiece-upload-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "workpiece-upload-job",
                    "status": background_job_status["upload"],
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )
    page.route(
        "**/jobs/unit-correction-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "unit-correction-job",
                    "status": background_job_status["correction"],
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")

    expect(page.get_by_test_id("workpieces-page")).to_be_visible()
    expect(page.get_by_role("link", name="Workpiece Catalogue")).to_be_visible()
    expect(
        page.get_by_text("This is a global reusable library", exact=False)
    ).to_be_visible()
    expect(
        page.get_by_text("do not mutate the active run", exact=False)
    ).to_be_visible()
    expect(page.get_by_test_id("workpiece-preview-fallback")).to_be_visible()
    expect(page.get_by_text("3D preview is unavailable")).to_be_visible()
    expect(page.get_by_role("heading", name="3D preview")).to_be_visible()
    expect(
        page.get_by_text("Archive this workpiece to enable unit correction.")
    ).to_be_visible()
    expect(page.get_by_role("button", name="Correct model units")).to_be_disabled()
    expect(page.get_by_test_id("workpiece-previews")).to_have_count(0)
    expect(page.get_by_role("button", name="Select Clamp")).to_be_visible()
    expect(
        page.get_by_test_id(
            "workpiece-isometric-11111111-1111-4111-8111-111111111111"
        ).locator("polygon")
    ).to_have_count(6)
    expect(page.get_by_role("button", name="Select Gauge block")).to_have_count(0)
    expect(page.get_by_text("Wrong model scale?", exact=True)).to_have_count(0)
    dimensions = page.get_by_test_id("workpiece-dimensions")
    dimensions.get_by_role("button", name="About model dimensions").hover()
    expect(page.get_by_role("tooltip")).to_contain_text("Wrong model scale?")
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Archive this workpiece first, then use Correct model units."
    )
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Existing immutable templates keep their original geometry snapshot."
    )
    page.keyboard.press("Escape")

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
    expect(
        correction.get_by_text("File was authored in metres — enlarge ×1000")
    ).to_be_visible()
    expect(
        correction.get_by_text("Model is 1000× too large — shrink ÷1000")
    ).to_be_visible()
    expect(correction.get_by_text("Current dimensions")).to_be_visible()
    expect(correction.get_by_text("After correction")).to_be_visible()
    correction.get_by_role("radio").filter(has_text="shrink ÷1000").click()
    correction.get_by_label("Unit correction operator").fill("qa-operator")
    correction.get_by_label("Confirm unit correction").click()
    correction.get_by_role("button", name="Queue unit correction").click()
    unit_progress = page.get_by_test_id("workpiece-unit-correction-progress")
    expect(unit_progress).to_contain_text("continues after navigation")
    expect(unit_progress.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    background_job_status["correction"] = "succeeded"
    expect(page.get_by_text("Workpiece units corrected")).to_be_visible()
    unit_request = next(
        value for value in requests if value["path"].endswith("/unit-corrections")
    )
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
    page.get_by_test_id("workpiece-import-input").set_input_files(
        {
            "name": "object_catalog.json",
            "mimeType": "application/json",
            "buffer": json.dumps(workpiece_catalog()).encode(),
        }
    )
    page.get_by_role("button", name="Import metadata").click()
    expect(page.get_by_text("Catalogue metadata imported")).to_be_visible()
    import_request = next(
        value for value in requests if value["path"].endswith("/import")
    )
    assert "object_catalog.json" in import_request["body"]

    page.get_by_test_id("workpiece-upload-button").click()
    page.get_by_test_id("workpiece-cad-input").set_input_files(
        {
            "name": "new-clamp.stl",
            "mimeType": "application/octet-stream",
            "buffer": b"solid clamp",
        }
    )
    page.get_by_test_id("workpiece-upload-name").fill("New clamp")
    page.get_by_test_id("workpiece-upload-alias").fill("Queued workpiece")
    page.get_by_test_id("workpiece-upload-tags").fill("new, metal")
    page.get_by_test_id("workpiece-upload-groups").fill("incoming")
    page.get_by_role("button", name="Upload and inspect").click()
    expect(page.get_by_text("Workpiece inspection queued")).to_be_visible()
    upload_progress = page.get_by_test_id("workpiece-upload-progress")
    expect(upload_progress).to_contain_text("continues after navigation")
    expect(upload_progress.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    background_job_status["upload"] = "succeeded"
    expect(page.get_by_text("Workpiece added to the catalogue")).to_be_visible()
    upload_request = next(
        value for value in requests if value["path"].endswith("/upload")
    )
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
    preview_job_status = {"value": "queued"}
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "request",
        lambda request: (
            canonical_mesh_requests.append(request.url)
            if urlparse(request.url).path.endswith("/assets/canonical_ply")
            else None
        ),
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
                    "status": preview_job_status["value"],
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
    foreground = np.linalg.norm(screenshot.astype(np.int16) - background, axis=2) > 24
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
        and stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] < width - 2
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
    preview_progress = page.get_by_test_id("workpiece-preview-progress")
    expect(preview_progress).to_contain_text("continues after navigation")
    expect(preview_progress.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    preview_job_status["value"] = "succeeded"
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
        page.get_by_test_id(f"workpiece-thumbnail-error-{workpiece['catalog_uuid']}")
    ).to_contain_text("Preview/server revision mismatch")

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")
    expect(
        page.get_by_test_id(f"workpiece-thumbnail-error-{workpiece['catalog_uuid']}")
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
    vertices = [
        [0, 0, 1.5],
        *[
            [
                10 * np.cos(2 * np.pi * index / sections),
                7 * np.sin(2 * np.pi * index / sections),
                0.8 * np.sin(6 * np.pi * index / sections),
            ]
            for index in range(sections)
        ],
    ]
    faces = [[0, index + 1, (index + 1) % sections + 1] for index in range(sections)]
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
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(route, thumbnail),
    )

    page.goto(f"{console_server.url}/#/workpieces", wait_until="networkidle")

    rendered = page.get_by_test_id(f"workpiece-isometric-{workpiece['catalog_uuid']}")
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
    page.add_init_script(
        "Object.defineProperty(Crypto.prototype, 'randomUUID', { value: undefined, configurable: true })"
    )
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route(
        "**/pose-templates/status",
        lambda route: fulfill_json(route, pose_template_source(available=True)),
    )
    page.route(
        "**/pose-templates/library",
        lambda route: fulfill_json(
            route, {"schema_version": "pose_template_library.v1", "templates": []}
        ),
    )
    page.route(
        "**/pose-templates/workpieces/*/orientations",
        lambda route: fulfill_json(
            route, pose_template_orientation_analysis(record["catalog_uuid"])
        ),
    )
    page.route(
        "**/pose-templates/workpieces/*/orientation-thumbnail",
        lambda route: fulfill_json(
            route, pose_template_orientation_thumbnail(record["catalog_uuid"])
        ),
    )

    def preview_handler(route) -> None:
        if route.request.method == "POST":
            fulfill_json(
                route, {"job_id": "preview-job", "request_id": "c" * 32}, status=202
            )
        else:
            fulfill_json(route, immutable_template_preview())

    page.route("**/pose-templates/preview**", preview_handler)
    page.goto(f"{console_server.url}/#/pose-templates", wait_until="networkidle")

    expect(page.get_by_text("Browser box", exact=True)).to_be_visible()
    page.get_by_role("button", name="Choose orientation for Browser box").click()
    page.get_by_test_id("orientation-chooser").get_by_role(
        "button", name="Add selected orientation"
    ).click()
    expect(
        page.get_by_role("button", name="Select and move Browser box")
    ).to_have_count(1)
    page.get_by_role("button", name="Choose orientation for Browser box").click()
    page.get_by_test_id("orientation-chooser").get_by_role(
        "button", name="Add selected orientation"
    ).click()
    expect(
        page.get_by_role("button", name="Select and move Browser box")
    ).to_have_count(2)
    page.get_by_role("button", name="Remove Browser box instance").click()
    expect(
        page.get_by_role("button", name="Select and move Browser box")
    ).to_have_count(1)
    expect(page.get_by_role("button", name="Generate immutable version")).to_be_enabled(
        timeout=15_000
    )
    assert page_errors == []


def test_ground_truth_workflow_selection_and_full_placement(
    console_server, page
) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    install_common_mocks(page)
    submitted: list[dict] = []
    selection_saved = {"value": False}
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
    preview_payload["instances"].append(
        {
            **second_instance,
            "preview_mesh_sha256": "f" * 64,
            "orientation": {"label": "Flat face"},
            "compensated_contours": [
                [
                    {"x_mm": 80, "y_mm": 55},
                    {"x_mm": 105, "y_mm": 55},
                    {"x_mm": 105, "y_mm": 65},
                    {"x_mm": 80, "y_mm": 65},
                ]
            ],
        }
    )
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
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
    }
    page.route(
        "**/pose-templates/library", lambda route: fulfill_json(route, library_payload)
    )
    page.route(
        "**/pose-templates/library/*/preview",
        lambda route: fulfill_json(route, preview_payload),
    )
    page.route(
        "**/pose-templates/library/*/thumbnail",
        lambda route: fulfill_json(route, immutable_template_thumbnail()),
    )
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
    page.route(
        "**/jobs/selection-job",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "selection-job",
                    "status": "succeeded",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )

    def selection_handler(route) -> None:
        if route.request.method == "POST":
            submitted.append(route.request.post_data_json)
            selection_saved["value"] = True
            fulfill_json(route, {"job_id": "selection-job"}, status=202)
        else:
            selection = (
                {
                    "template_uuid": "22222222-2222-4222-8222-222222222222",
                    "placement_confirmed": True,
                    "instances": [
                        {
                            "instance_uuid": "33333333-3333-4333-8333-333333333333",
                            "name": "Clamp",
                            "obj_id": 7,
                            "template_base_from_object": {"translation_mm": [0, 0, 0]},
                        }
                    ],
                }
                if selection_saved["value"]
                else None
            )
            fulfill_json(
                route,
                {
                    "selection": selection,
                    "replacement_blockers": [],
                    "ready": selection is not None,
                },
            )

    page.route("**/pose-templates/runs/selection**", selection_handler)
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=template",
        wait_until="networkidle",
    )
    expect(page.get_by_test_id("ground-truth-workflow")).to_be_visible()
    placement_boxes = [
        page.get_by_label(f"Template placement {label}").bounding_box()
        for label in ("X mm", "Y mm", "Z mm", "Roll °", "Pitch °", "Yaw °")
    ]
    assert all(box is not None for box in placement_boxes)
    boxes = [box for box in placement_boxes if box is not None]
    assert max(box["y"] for box in boxes[:3]) - min(box["y"] for box in boxes[:3]) < 2
    assert max(box["y"] for box in boxes[3:]) - min(box["y"] for box in boxes[3:]) < 2
    assert boxes[3]["y"] > boxes[0]["y"] + boxes[0]["height"]
    assert min(box["width"] for box in boxes) >= 75
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
    expect(page.get_by_test_id("selected-template-scene")).to_have_attribute(
        "data-origin-offset-mm", "15,15"
    )
    expect(
        page.get_by_test_id("selected-template-scene").locator("canvas")
    ).to_have_count(1)
    expect(page.get_by_text("Exact immutable PLY detail")).to_be_visible()
    object_index = page.get_by_test_id("selected-template-object-index")
    expect(
        object_index.get_by_role("button", name="Focus Clamp, obj_000007, instance 1")
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
    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(confirmation).not_to_be_checked()
    page.get_by_role("combobox", name="Active run folder").click()
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
    workflow_steps = page.get_by_role("navigation", name="Workflow steps")
    workflow_steps.get_by_role("button").filter(has_text="Check readiness").click()
    expect(page.get_by_test_id("ground-truth-workflow")).not_to_be_visible()
    workflow_steps.get_by_role("button").filter(
        has_text="Choose the pose template and placement"
    ).click()
    expect(page.get_by_label("Template placement X mm")).to_have_value("12")
    expect(page.get_by_label("Template placement Z mm")).to_have_value("34")
    expect(page.get_by_label("Template placement Yaw °")).to_have_value("90")
    expect(confirmation).to_be_checked()
    page.get_by_role("button", name="Select for run").click()
    expect(page.get_by_text("Pose template selection queued")).to_be_visible()
    assert submitted[0]["confirmed"] is True
    assert submitted[0]["template_uuid"] == "22222222-2222-4222-8222-222222222222"
    assert submitted[0]["placement"]["matrix"][0][3] == 12
    assert submitted[0]["placement"]["matrix"][2][3] == 34
    saved_selection = page.get_by_test_id("saved-pose-template-selection")
    draft = page.get_by_test_id("pose-template-selection-draft")
    expect(saved_selection.get_by_test_id("saved-pose-template-ready")).to_be_visible()
    expect(
        draft.get_by_role("heading", name="Replacement selection draft")
    ).to_be_visible()
    expect(draft).to_contain_text("The saved pose template at left remains active")
    expect(draft.get_by_text("Required", exact=True)).to_have_count(0)
    expect(draft.get_by_role("button", name="Replace run selection")).to_be_visible()


def test_run_config_preflight_blocker_and_fresh_capture_gates(
    console_server, page
) -> None:
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
    page.route(
        "**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status())
    )
    capture_setup = {
        "queued": False,
        "post_queue_reads": 0,
        "job_id": None,
        "status": None,
    }
    readiness_job_status: dict[str, str | None] = {"value": None}

    def readiness_job_payload(status: str) -> dict:
        return {
            "id": "readiness-1",
            "name": "pipeline:run_preflight",
            "command": ["uv", "run", "python", "scripts/run_pipeline_stage.py"],
            "cwd": "/repo",
            "status": status,
            "created_at": "2026-07-27T11:00:00Z",
            "started_at": ("2026-07-27T11:00:01Z" if status != "queued" else None),
            "ended_at": ("2026-07-27T11:00:03Z" if status == "succeeded" else None),
            "returncode": 0 if status == "succeeded" else None,
            "message": None,
            "tail": [],
            "resources": ["disk_io"],
            "parameters": {
                "run_root": RUN_ROOT,
                "pipeline_stage": "run_preflight",
            },
            "scope_kind": "run",
            "run_root": RUN_ROOT,
            "log_path": "/tmp/readiness-1.log",
        }

    def readiness_submit_handler(route) -> None:
        requests.append({"path": "/pipeline/run", "body": route.request.post_data_json})
        readiness_job_status["value"] = "queued"
        fulfill_json(route, {"job_id": "readiness-1", "status": "queued"}, status=202)

    def jobs_handler(route) -> None:
        status = readiness_job_status["value"]
        fulfill_json(
            route,
            {
                "jobs": [] if status is None else [readiness_job_payload(status)],
                "resources": {},
            },
        )

    page.route("**/pipeline/run", readiness_submit_handler)
    page.route("**/jobs", jobs_handler)

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

    def capture_jobs_handler(route) -> None:
        status = capture_setup["status"]
        job_id = capture_setup["job_id"]
        fulfill_json(
            route,
            {
                "run_root": RUN_ROOT,
                "jobs": (
                    []
                    if status is None or job_id is None
                    else [
                        {
                            "id": job_id,
                            "name": "Calibration capture",
                            "status": status,
                            "kind": "pipeline_sequence",
                            "stage": None,
                            "sequence": "real_full_capture_validation",
                            "run_root": RUN_ROOT,
                            "resources": ["cameras", "robot", "disk_io"],
                            "message": None,
                            "created_at": "2026-07-27T12:00:00Z",
                            "started_at": None,
                            "ended_at": None,
                            "active": True,
                            "tail": [],
                            "log_endpoint": f"/capture/jobs/{job_id}/log",
                            "stop_endpoint": f"/capture/jobs/{job_id}/stop",
                        }
                    ]
                ),
                "active_count": 0 if status is None else 1,
                "resources": {},
                "status_artifact": None,
            },
        )

    def pipeline_sequence_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/pipeline/run-sequence", "body": body})
        capture_setup["queued"] = True
        capture_setup["job_id"] = f"job-{len(requests)}"
        capture_setup["status"] = "queued"
        fulfill_json(
            route,
            {"job_id": capture_setup["job_id"], "status": "queued"},
            status=202,
        )

    page.route("**/calibration/setup?**", calibration_setup_handler)
    page.route("**/capture/jobs**", capture_jobs_handler)
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
    expect(
        page.get_by_text("Full capture is an A1 joint PTP", exact=False)
    ).to_be_visible()
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
    preflight_request = next(
        item["body"] for item in requests if item["path"] == "/pipeline/run"
    )
    assert preflight_request["stage"] == "run_preflight"
    assert "allow_cameras" not in json.dumps(preflight_request)
    readiness_job = readiness.get_by_test_id("calibration-readiness-job-status")
    expect(readiness_job).to_contain_text("Readiness check is queued", timeout=5_000)
    expect(readiness_job).to_contain_text("continues after navigation")
    expect(
        readiness_job.get_by_role("link", name="Open live status in Jobs")
    ).to_have_attribute("href", "#/jobs")
    expect(readiness.get_by_role("button", name="Check in progress…")).to_be_disabled()

    readiness_job_status["value"] = "running"
    page.reload(wait_until="networkidle")
    readiness = page.get_by_test_id("calibration-readiness-check")
    expect(
        readiness.get_by_test_id("calibration-readiness-job-status")
    ).to_contain_text("Readiness check is running", timeout=5_000)
    expect(readiness.get_by_role("button", name="Check in progress…")).to_be_disabled()
    readiness_job_status["value"] = "succeeded"
    preflight_state["blocker"] = None
    page.reload(wait_until="networkidle")
    page.get_by_role("navigation", name="Workflow steps").get_by_role("button").filter(
        has_text="Record calibration images"
    ).click()
    page.get_by_role("button", name="Review and start capture", exact=True).click()
    expect(page.get_by_test_id("capture-timeout-envelope")).to_contain_text(
        "720 s total · 15 s sustained camera readiness (3 frames each) · 5 s maximum live camera-metadata pause · 120 s to first robot packet · 60 s between robot packets"
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
        item["body"] for item in requests if item["path"] == "/pipeline/run-sequence"
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
                "camera_metadata_idle_timeout_s": 5,
            },
        },
    }
    assert any(item["path"] == "/sensors/previews/stop" for item in requests)
    capture_job = page.get_by_test_id("capture-active-job")
    expect(capture_job).to_contain_text("Calibration capture is queued")
    expect(capture_job).to_contain_text("continues after navigation")
    expect(
        capture_job.get_by_role("link", name="Open capture in Jobs")
    ).to_have_attribute("href", "#/jobs")
    expect(
        page.get_by_role("button", name="Review and start capture", exact=True)
    ).to_have_count(0)
    assert (
        len([item for item in requests if item["path"] == "/pipeline/run-sequence"])
        == 1
    )
    page.get_by_role("navigation", name="Workflow steps").get_by_role("button").filter(
        has_text="Calculate, review, and publish"
    ).click()
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
            request.method == "GET" and urlparse(request.url).path == "/run-config"
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
    mobile_source_run_root = "/tmp/posetestbot-console/calibration-mobile-july-21"
    static_source_run_root = "/tmp/posetestbot-console/calibration-static-july-21"
    mobile_source_bundle_sha256 = "a" * 64
    static_source_bundle_sha256 = "d" * 64
    combined_bundle_sha256 = "e" * 64
    selected_calibration_path = (
        "processed/calibration_selection/calibration_profiles.json"
    )
    selected_intrinsics_path = (
        "processed/calibration_selection/intrinsic_calibration_profiles.json"
    )
    selection_artifact = calibration_selection_artifact(
        bundle_sha256=combined_bundle_sha256,
        calibration_profiles=selected_calibration_path,
        intrinsic_calibration_profiles=selected_intrinsics_path,
        source_run_root=mobile_source_run_root,
        source_run_name="Combined calibration from 2 source runs",
    )
    selection_artifact["schema_version"] = "calibration_profile_selection.v2"
    selection_artifact["source"] = {
        "kind": "composite",
        "run_name": "Combined calibration from 2 source runs",
        "bundle_sha256": combined_bundle_sha256,
    }
    selection_artifact["sources"] = [
        {
            "run_root": mobile_source_run_root,
            "run_name": "Mobile calibration July 21",
            "bundle_sha256": mobile_source_bundle_sha256,
            "selected_sensor_keys": ["realsense_d435:wrist-1"],
        },
        {
            "run_root": static_source_run_root,
            "run_name": "Static calibration July 21",
            "bundle_sha256": static_source_bundle_sha256,
            "selected_sensor_keys": ["realsense_d435:static-1"],
        },
    ]
    selection_artifact["sensor_profiles"] = {
        "realsense_d435:wrist-1": "profile-wrist-1",
        "realsense_d435:static-1": "profile-static-1",
    }
    mobile_source = {
        "source_run_root": mobile_source_run_root,
        "source_run_name": "Mobile calibration July 21",
        "bundle_sha256": mobile_source_bundle_sha256,
        "valid": True,
        "compatible": False,
        "issues": [
            {
                "code": "sensor_identity_not_calibrated",
                "message": "No calibration profile matches realsense_d435:static-1.",
                "sensor_key": "realsense_d435:static-1",
            }
        ],
        "calibration_profiles": {
            "sha256": "b" * 64,
            "valid_profile_count": 1,
            "profiles": [
                {
                    "profile_id": "profile-wrist-1",
                    "sensor_type": "realsense_d435",
                    "sensor_id": "wrist-1",
                    "mounting_mode": "eye_in_hand",
                    "status": "valid",
                    "resolution": [1280, 720],
                    "intrinsic_profile_id": "intrinsic-wrist-1",
                    "method": "IPPE + park",
                    "quality": {
                        "num_observations": 18,
                        "num_inliers": 17,
                        "mean_reprojection_error_px": 0.31,
                    },
                },
            ],
        },
        "intrinsic_calibration_profiles": {
            "sha256": "c" * 64,
            "profile_count": 1,
            "profiles": [
                {
                    "profile_id": "intrinsic-wrist-1",
                    "sensor_id": "wrist-1",
                    "resolution": [1280, 720],
                    "orientation": "normal",
                }
            ],
        },
    }
    static_source = {
        "source_run_root": static_source_run_root,
        "source_run_name": "Static calibration July 21",
        "bundle_sha256": static_source_bundle_sha256,
        "valid": True,
        "compatible": False,
        "issues": [
            {
                "code": "sensor_identity_not_calibrated",
                "message": "No calibration profile matches realsense_d435:wrist-1.",
                "sensor_key": "realsense_d435:wrist-1",
            }
        ],
        "calibration_profiles": {
            "sha256": "7" * 64,
            "valid_profile_count": 1,
            "profiles": [
                {
                    "profile_id": "profile-static-1",
                    "sensor_type": "realsense_d435",
                    "sensor_id": "static-1",
                    "mounting_mode": "static",
                    "status": "valid",
                    "resolution": [1280, 720],
                    "intrinsic_profile_id": "intrinsic-static-1",
                    "method": "IPPE + park",
                    "quality": {
                        "num_observations": 18,
                        "num_inliers": 17,
                        "mean_reprojection_error_px": 0.29,
                    },
                }
            ],
        },
        "intrinsic_calibration_profiles": {
            "sha256": "8" * 64,
            "profile_count": 1,
            "profiles": [
                {
                    "profile_id": "intrinsic-static-1",
                    "sensor_id": "static-1",
                    "resolution": [1280, 720],
                    "orientation": "inverted",
                }
            ],
        },
    }
    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route(
        "**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status())
    )
    page.route(
        "**/ui/calibrations?**",
        lambda route: fulfill_json(
            route,
            {"selected": None, "calibrations": [mobile_source, static_source]},
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

    expect(page.get_by_role("heading", name="Record an object dataset")).to_be_visible()
    speed = page.get_by_label("Requested robot capture speed (m/s)")
    expect(speed).to_have_value("0.2")
    expect(speed).to_have_attribute("max", "1")
    expect(
        page.get_by_text(
            "Requests above 0.03 m/s require the commissioned structured-command app",
            exact=False,
        )
    ).to_be_visible()
    expect(
        page.get_by_text("Speed alone cannot guarantee sharp frames", exact=False)
    ).to_be_visible()
    speed.fill("0.15")
    expect(
        page.get_by_role("heading", name="Saved camera calibration")
    ).to_contain_text("Required")
    expect(
        page.get_by_text(
            "Required: select and validate a previously published calibration below."
        )
    ).to_be_visible()
    save_setup = page.get_by_role("button", name="Save setup")
    expect(save_setup).to_be_disabled()
    readiness_action = page.get_by_role("button", name="Check readiness", exact=True)
    expect(readiness_action).to_have_count(0)

    assignments = page.get_by_test_id("calibration-source-assignments")
    expect(assignments.get_by_test_id("calibration-source-assignment")).to_have_count(2)
    mobile_source_choice = page.get_by_role(
        "combobox", name="Calibration source for Wrist RGB-D"
    )
    static_source_choice = page.get_by_role(
        "combobox", name="Calibration source for Static RGB-D"
    )
    mobile_source_choice.click()
    page.get_by_role("option").filter(has_text="Mobile calibration July 21").click()
    expect(mobile_source_choice).to_contain_text("Mobile calibration July 21")
    expect(
        page.get_by_role("button", name="Validate and save setup", exact=True)
    ).to_be_disabled()
    static_source_choice.click()
    page.get_by_role("option").filter(has_text="Static calibration July 21").click()
    expect(static_source_choice).to_contain_text("Static calibration July 21")
    expect(page.get_by_text("Ready to combine and validate")).to_be_visible()
    synchronization_mode = page.get_by_role("combobox", name="Synchronization mode")
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
    expect(
        page.get_by_text("Depth-only hardware synchronization boundary")
    ).to_be_visible()
    expect(
        page.get_by_text("not certified as hardware-synchronous across cameras")
    ).to_be_visible()
    expect(page.get_by_test_id("hardware-sync-contract-status")).to_contain_text(
        "Hardware trigger configuration is complete"
    )
    expect(
        page.get_by_test_id("hardware-sync-qualification-requirement")
    ).to_contain_text("Current physical qualification required before recording")
    expect(
        page.get_by_test_id("hardware-sync-qualification-requirement")
    ).to_contain_text("hardware_sync_qualification.json")
    page.get_by_label("Trigger group ID").fill("research-mixed-rig")
    workflow_steps = page.get_by_role("navigation", name="Workflow steps")
    workflow_steps.get_by_role("button").filter(
        has_text="Choose the pose template and placement"
    ).click()
    expect(page.get_by_test_id("object_dataset-run-setup")).not_to_be_visible()
    workflow_steps.get_by_role("button").filter(
        has_text="Configure cameras and select calibration"
    ).click()
    expect(speed).to_have_value("0.15")
    expect(mobile_source_choice).to_contain_text("Mobile calibration July 21")
    expect(static_source_choice).to_contain_text("Static calibration July 21")
    expect(page.get_by_label("Trigger group ID")).to_have_value("research-mixed-rig")
    validate_and_save = page.get_by_role(
        "button", name="Validate and save setup", exact=True
    )
    expect(validate_and_save).to_be_enabled()
    validate_and_save.click()

    expect(page.get_by_text("Object dataset setup saved")).to_be_visible()
    assert selection_requests == [
        {
            "run_root": RUN_ROOT,
            "source_selections": [
                {
                    "source_run_root": mobile_source_run_root,
                    "expected_bundle_sha256": mobile_source_bundle_sha256,
                    "sensor_keys": ["realsense_d435:wrist-1"],
                },
                {
                    "source_run_root": static_source_run_root,
                    "expected_bundle_sha256": static_source_bundle_sha256,
                    "sensor_keys": ["realsense_d435:static-1"],
                },
            ],
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
    assert written["velocity"] == 0.15
    assert written["sequence"] == "calibrated_capture_to_bop_dataset_dry_run"
    assert written["calibration_profiles"] == selected_calibration_path
    assert written["expected_calibration_bundle_sha256"] == combined_bundle_sha256
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

    workflow_steps.get_by_role("button").filter(has_text="Check readiness").click()
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
        "sensor_profile_mapping": [
            {
                "sensor_key": "realsense_d435:wrist-1",
                "profile_id": "profile-wrist-1",
                "intrinsic_profile_id": "intrinsic-wrist-1",
                "mounting_mode": "eye_in_hand",
            }
        ],
    }
    replacement_selection = calibration_selection_artifact(
        bundle_sha256=replacement_bundle_sha256,
        calibration_profiles=replacement_calibration_path,
        intrinsic_calibration_profiles=replacement_intrinsics_path,
        source_run_root=replacement_source_root,
        source_run_name="Replacement calibration",
    )

    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route(
        "**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status())
    )
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

    expect(
        page.get_by_text("A verified calibration snapshot is selected")
    ).to_be_visible()
    replacement_choice = page.get_by_role(
        "combobox", name="Calibration source for Wrist RGB-D"
    )
    replacement_choice.click()
    page.get_by_role("option").filter(has_text="Replacement calibration").click()
    expect(replacement_choice).to_contain_text("Replacement calibration")

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
            "source_selections": [
                {
                    "source_run_root": replacement_source_root,
                    "expected_bundle_sha256": replacement_bundle_sha256,
                    "sensor_keys": ["realsense_d435:wrist-1"],
                }
            ],
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
    assert written["expected_calibration_bundle_sha256"] == replacement_bundle_sha256


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
        "error": ("Profile profile-wrist-1 has no verified robot pose time offset."),
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
    workflow_steps = page.get_by_role("navigation", name="Workflow steps")
    workflow_steps.get_by_role("button").filter(has_text="Check readiness").click()
    readiness = page.get_by_test_id("dataset-readiness-check")
    expect(readiness).to_contain_text(
        "Calibration geometry and automatic timing verified"
    )
    expect(readiness).to_contain_text(
        "The selected calibration timing contract is invalid"
    )
    workflow_steps.get_by_role("button").filter(
        has_text="Process frames and create the base BOP export"
    ).click()
    expect(page.get_by_test_id("dataset-sync-timing-contract")).to_contain_text(
        "Return to Step 1 and select a calibration with valid timing"
    )


def test_dataset_processing_is_one_ordered_operator_action(
    console_server,
    page,
) -> None:
    processing_requests: list[dict] = []
    processing_job_status: dict[str, str | None] = {"value": None}
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
        }
    ]

    install_common_mocks(page, config_payload=configured)
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview))
    page.route(
        "**/sensors/status", lambda route: fulfill_json(route, selected_sensor_status())
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

    def processing_handler(route) -> None:
        processing_requests.append(route.request.post_data_json)
        processing_job_status["value"] = "queued"
        fulfill_json(
            route,
            {"job_id": "dataset-processing-1", "status": "queued"},
            status=202,
        )

    def jobs_handler(route) -> None:
        status = processing_job_status["value"]
        jobs = (
            []
            if status is None
            else [
                {
                    "id": "dataset-processing-1",
                    "name": "pipeline-run-config:calibrated_capture_to_bop_dataset_dry_run",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/run_pipeline_sequence.py",
                    ],
                    "cwd": "/repo",
                    "status": status,
                    "created_at": "2026-07-26T10:57:51Z",
                    "started_at": "2026-07-26T10:57:52Z",
                    "ended_at": (
                        "2026-07-26T11:00:48Z"
                        if status in {"succeeded", "failed", "canceled"}
                        else None
                    ),
                    "returncode": 0 if status == "succeeded" else None,
                    "message": (
                        "Command completed successfully."
                        if status == "succeeded"
                        else None
                    ),
                    "tail": ["processing"],
                    "resources": ["cpu", "disk_io"],
                    "parameters": {
                        "pipeline_sequence": "calibrated_capture_to_bop_dataset_dry_run",
                        "run_root": RUN_ROOT,
                    },
                    "scope_kind": "run",
                    "run_root": RUN_ROOT,
                    "log_path": "/tmp/dataset-processing-1.log",
                    "visibility": "operator",
                }
            ]
        )
        fulfill_json(route, {"jobs": jobs, "resources": {}})

    page.route("**/pipeline/run-config", processing_handler)
    page.route("**/jobs", jobs_handler)
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=sync",
        wait_until="networkidle",
    )

    stepper = page.get_by_role("navigation", name="Workflow steps")
    configure_step_button = stepper.get_by_role("button").filter(
        has_text="Configure cameras and select calibration"
    )
    sync_step_button = stepper.get_by_role("button").filter(
        has_text="Process frames and create the base BOP export"
    )
    export_step_button = stepper.get_by_role("button").filter(
        has_text="Add optional BOP ground-truth evidence"
    )
    configure_step_button.click()
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
    timing_policy.get_by_role("button", name="About robot-pose time offset").hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "Positive means pair the frame with a robot pose recorded later"
    )
    page.keyboard.press("Escape")
    timing_policy.get_by_role("button", name="About calibration timestamp pair").hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "exact frame and robot clock fields"
    )
    page.keyboard.press("Escape")
    timing_policy.get_by_role("button", name="About maximum robot-pose gap").hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "frame is excluded when its nearest robot pose is farther away"
    )
    page.keyboard.press("Escape")
    sync_step_button.click()
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
    sync_step = page.locator('[data-workflow-step="sync"]')
    expect(sync_step.get_by_text("Current step", exact=True)).to_be_visible()
    expect(
        sync_step.get_by_text(
            "Copy models and write the base BOP dataset",
            exact=True,
        )
    ).to_be_visible()
    expect(processing).to_contain_text(
        "One queued job runs five backend stages grouped into the four operator outcomes below"
    )
    expect(processing).to_contain_text(
        "Ground-truth generation is chosen separately in optional step 6"
    )
    expect(processing).to_contain_text(
        "Calibration validation is automatic here; there is no second operator preflight."
    )
    expect(processing).to_contain_text("Copy models and write the base BOP dataset")
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
        expect(page.get_by_role("button", name=stale_action, exact=True)).to_have_count(
            0
        )

    export_step_button.click()
    export_outcome = page.locator('[data-workflow-step="export"]')
    expect(export_outcome).to_contain_text("BOP export has not completed")
    expect(export_outcome).to_contain_text("Use the processing job in step 5")
    expect(export_outcome).to_contain_text("before optional ground-truth generation")

    sync_step_button.click()
    process_action.click()
    expect(page.get_by_text("Dataset processing queued")).to_be_visible()
    assert processing_requests == [{"run_root": RUN_ROOT}]
    processing_job_status["value"] = "running"
    job_status = processing.get_by_test_id("dataset-processing-job-status")
    expect(job_status).to_contain_text("Dataset processing is running", timeout=5_000)
    expect(job_status).to_contain_text("dataset-processing-1")
    expect(job_status).to_contain_text("continues after navigation")
    expect(
        job_status.get_by_role("link", name="Open live log in Jobs")
    ).to_have_attribute("href", "#/jobs")
    expect(page.get_by_role("button", name="Processing…")).to_be_disabled()
    expect(sync_step_button).to_contain_text("Running")

    processing_job_status["value"] = "succeeded"
    expect(job_status).to_contain_text(
        "Processing finished; export evidence is still being verified",
        timeout=5_000,
    )
    expect(job_status).to_contain_text(
        "has not yet accepted bop/bop_export_manifest.json"
    )

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
    job_status.get_by_role("button", name="Refresh evidence").click()
    expect(job_status).to_contain_text(
        "Dataset processing finished and BOP export is verified"
    )
    expect(sync_step_button).to_contain_text("Complete")
    export_step_button.click()
    export_outcome = page.locator('[data-workflow-step="export"]')
    expect(export_outcome.get_by_text("BOP image/model export is ready")).to_be_visible(
        timeout=5_000
    )
    expect(export_outcome).to_contain_text(
        "base export has populated calibrated scenes, models, and object targets"
    )


def test_dataset_export_queues_selected_gt_version_and_recovers_render_job(
    console_server,
    page,
) -> None:
    annotation_requests: list[dict] = []
    annotation_job_status: dict[str, str | None] = {"value": None}
    configured = run_config(plan_only=False)
    configured["dataset_mode"] = "pose_template"
    configured["pose_template"] = {
        "template_uuid": "22222222-2222-4222-8222-222222222222",
        "placement_confirmed": True,
    }
    overview = overview_payload(configured)
    next(section for section in overview["sidebar"] if section["id"] == "bop")[
        "artifacts"
    ] = [
        {
            "path": "bop/bop_export_manifest.json",
            "exists": True,
            "status": "complete",
        }
    ]
    setup = {
        "schema_version": "bop_annotation_setup.v1",
        "run_root": RUN_ROOT,
        "runtime": {
            "available": True,
            "required_version": "2.8.0",
            "detected_version": "2.8.0",
            "install_command": None,
            "reason": None,
        },
        "toolkit": {
            "available": False,
            "status": "unavailable",
            "revision": None,
            "required_revision": "f" * 64,
            "environment_ready": False,
            "renderer": "vispy",
            "install_command": "bash scripts/install.sh --with-bop-toolkit",
            "reason": "The pinned BOP Toolkit runtime is unavailable.",
        },
        "readiness": {"ready": True, "blockers": [], "warnings": []},
        "readiness_by_mode": {
            "pose": {
                "ready": True,
                "blockers": [],
                "warnings": [
                    {
                        "code": "pose_only",
                        "message": "Pose-only output cannot be evaluated with BOP metrics.",
                    }
                ],
            },
            "pose_and_masks": {
                "ready": False,
                "blockers": [
                    {
                        "code": "bop_toolkit_unavailable",
                        "message": "The pinned BOP Toolkit runtime is unavailable.",
                    }
                ],
                "warnings": [],
            },
        },
        "current_output": None,
        "counts": {"sensors": 3, "frames": 1621, "instances": 1621},
        "provenance": {
            "bop_export_manifest_sha256": "a" * 64,
            "calibration_bundle_sha256": "b" * 64,
            "pose_template_sha256": "c" * 64,
        },
    }

    def job_payload(status: str) -> dict:
        return {
            "id": "bop-annotations-1",
            "name": "bop_annotations:pose_and_masks",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/run_bop_annotations.py",
            ],
            "cwd": "/repo",
            "status": status,
            "created_at": "2026-07-26T13:00:00Z",
            "started_at": ("2026-07-26T13:00:01Z" if status != "queued" else None),
            "ended_at": "2026-07-26T13:04:00Z" if status == "succeeded" else None,
            "returncode": 0 if status == "succeeded" else None,
            "message": (
                "Command completed successfully." if status == "succeeded" else None
            ),
            "tail": ["rendering instance masks"],
            "resources": ["cpu", "disk_io"],
            "parameters": {
                "run_root": RUN_ROOT,
                "bop_annotations": True,
                "annotation_mode": "pose_and_masks",
            },
            "scope_kind": "run",
            "run_root": RUN_ROOT,
            "log_path": "/tmp/bop-annotations-1.log",
            "visibility": "operator",
        }

    def annotations_handler(route) -> None:
        annotation_requests.append(route.request.post_data_json)
        annotation_job_status["value"] = "queued"
        fulfill_json(
            route,
            {
                "job_id": "bop-annotations-1",
                "job": job_payload("queued"),
            },
            status=202,
        )

    def jobs_handler(route) -> None:
        status = annotation_job_status["value"]
        fulfill_json(
            route,
            {
                "jobs": [] if status is None else [job_payload(status)],
                "resources": {},
            },
        )

    install_common_mocks(page, config_payload=configured)
    page.route("**/ui/overview**", lambda route: fulfill_json(route, overview))
    page.route(
        "**/bop/annotations/setup?**",
        lambda route: fulfill_json(route, setup),
    )
    page.route("**/bop/annotations", annotations_handler)
    page.route("**/jobs", jobs_handler)
    page.goto(
        f"{console_server.url}/#/workflow/dataset?step=export",
        wait_until="networkidle",
    )

    export_step_button = (
        page.get_by_role("navigation", name="Workflow steps")
        .get_by_role("button")
        .filter(has_text="Add optional BOP ground-truth evidence")
    )
    expect(export_step_button).to_contain_text("Optional")
    expect(export_step_button).to_contain_text("Ready")
    generator = page.get_by_test_id("bop-ground-truth-generation")
    expect(generator).to_be_visible()
    expect(generator).to_contain_text("Choose the BOP ground-truth evidence")
    expect(generator).to_contain_text(
        "object model, pose template, measured placement, robot pose, and selected camera calibration"
    )
    expect(generator).to_contain_text("1,621")
    pose_choice = generator.get_by_role("radio").filter(
        has_text="Plain pose ground truth"
    )
    full_choice = generator.get_by_role("radio").filter(
        has_text="Pose + object masks and ROI"
    )
    expect(pose_choice).to_have_attribute("aria-checked", "false")
    expect(pose_choice).to_contain_text("scene_gt.json")
    expect(pose_choice).to_contain_text("not evaluation-ready")
    expect(full_choice).to_have_attribute("aria-checked", "true")
    expect(full_choice).to_contain_text("scene_gt_info.json")
    expect(full_choice).to_contain_text("mask/")
    expect(full_choice).to_contain_text("mask_visib/")
    expect(full_choice).to_contain_text("bbox_obj")
    expect(full_choice).to_contain_text("bbox_visib")
    expect(generator).to_contain_text(
        "official BOP Toolkit then renders full and visible masks against captured depth"
    )
    expect(
        generator.get_by_role("button", name="Generate pose + masks")
    ).to_be_disabled()
    expect(generator.get_by_test_id("bop-annotation-blockers")).to_contain_text(
        "The pinned BOP Toolkit runtime is unavailable."
    )
    pose_choice.click()
    expect(pose_choice).to_have_attribute("aria-checked", "true")
    expect(generator.get_by_role("button", name="Generate pose GT")).to_be_enabled()
    expect(generator).to_contain_text(
        "Pose-only output cannot be evaluated with BOP metrics."
    )
    expect(generator.get_by_role("link", name="Inspect BOP metrics")).to_have_count(0)
    setup["toolkit"].update(
        {
            "available": True,
            "status": "ready",
            "revision": "f" * 64,
            "environment_ready": True,
            "install_command": None,
            "reason": None,
        }
    )
    setup["readiness_by_mode"]["pose_and_masks"] = {
        "ready": True,
        "blockers": [],
        "warnings": [],
    }
    generator.get_by_role("button", name="Refresh readiness").click()
    full_choice.click()
    expect(full_choice).to_have_attribute("aria-checked", "true")
    expect(
        generator.get_by_role("button", name="Generate pose + masks")
    ).to_be_enabled()

    generator.get_by_role("button", name="Generate pose + masks").click()
    expect(page.get_by_text("Ground-truth generation queued")).to_be_visible()
    assert annotation_requests == [{"run_root": RUN_ROOT, "mode": "pose_and_masks"}]
    job_status = generator.get_by_test_id("bop-annotation-job-status")
    expect(job_status).to_contain_text("Ground-truth generation is queued")
    expect(job_status).to_contain_text("continues after navigation")
    expect(
        job_status.get_by_role("link", name="Open live log in Jobs")
    ).to_have_attribute("href", "#/jobs")

    annotation_job_status["value"] = "running"
    page.reload(wait_until="networkidle")
    generator = page.get_by_test_id("bop-ground-truth-generation")
    job_status = generator.get_by_test_id("bop-annotation-job-status")
    expect(job_status).to_contain_text(
        "Ground-truth generation is running",
        timeout=5_000,
    )
    expect(job_status).to_contain_text("bop-annotations-1")
    expect(generator.get_by_role("button", name="Generating…")).to_be_disabled()

    setup["current_output"] = {
        "mode": "pose_and_masks",
        "state": "complete",
        "annotation_count": 1621,
        "mask_count": 1621,
        "visible_mask_count": 1621,
        "evaluation_ready": True,
        "verified": True,
        "integrity_error": None,
        "manifest_sha256": "d" * 64,
        "blenderproc_version": "2.8.0",
        "toolkit_revision": "f" * 64,
    }
    annotation_job_status["value"] = "succeeded"
    evidence = generator.get_by_test_id("bop-annotation-evidence")
    expect(evidence).to_contain_text("verified for evaluation", timeout=5_000)
    expect(evidence).to_contain_text("1,621")
    expect(evidence).to_contain_text("1,621")
    expect(evidence).to_contain_text("full-frame instance masks")
    expect(evidence.get_by_role("link", name="Inspect BOP metrics")).to_have_attribute(
        "href", "#/bop-evaluation"
    )
    expect(export_step_button).to_contain_text("Complete", timeout=5_000)


def test_run_setup_keeps_and_edits_the_run_owned_camera_alias(
    console_server,
    page,
) -> None:
    requests: list[dict] = []
    configured = run_config(
        sensors=[
            {
                "sensor_type": "realsense_d435",
                "device_id": "wrist-1",
                "display_name": "Saved run wrist",
                "operator_alias": "Saved run wrist",
                "mounting_mode": "eye_in_hand",
                "enabled": True,
                "inverted": False,
            }
        ]
    )
    status = selected_sensor_status()
    status["families"][0]["devices"][0]["alias"] = "New lab-wide wrist"
    status["families"][0]["devices"][0]["effective_display_name"] = "New lab-wide wrist"
    install_common_mocks(page, requests=requests, config_payload=configured)
    page.route("**/sensors/status", lambda route: fulfill_json(route, status))

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=configure",
        wait_until="networkidle",
    )

    row = page.locator(
        '[data-testid="run-camera-row"][data-sensor-key="realsense_d435:wrist-1"]'
    )
    expect(row).to_contain_text("Saved run wrist")
    expect(row).not_to_contain_text("New lab-wide wrist")
    alias = row.get_by_label("Operator alias for realsense_d435:wrist-1")
    expect(alias).to_have_value("Saved run wrist")
    expect(row).to_contain_text("run_config.json")
    expect(row).to_contain_text("dataset_manifest.json")

    alias.fill("Dataset wrist view")
    expect(row).to_contain_text("Dataset wrist view")
    page.get_by_role("button", name="Save setup").click()
    expect(page.get_by_text("Calibration recording setup saved")).to_be_visible()

    written = next(item["body"] for item in requests if item["path"] == "/run-config")
    saved = written["sensors"][0]
    assert saved["operator_alias"] == "Dataset wrist view"
    assert saved["display_name"] == "Dataset wrist view"
    assert saved["device_id"] == "wrist-1"


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


def test_devices_show_typed_connection_state_and_visible_disabled_reasons(
    console_server,
    page,
) -> None:
    install_common_mocks(page)
    page.route(
        "**/sensors/status",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "sensor_status.v1",
                "families": [
                    {
                        "sensor_type": "zed_2i",
                        "display_name": "Stereolabs ZED 2i",
                        "devices": [
                            {
                                "sensor_type": "zed_2i",
                                "device_id": "zed-lab",
                                "display_name": "ZED lab",
                                "effective_display_name": "ZED lab",
                                "connected": True,
                                "capture_ready": True,
                                "live_rgb_preview_supported": False,
                                "mounting_mode": "static",
                                "inverted": False,
                            }
                        ],
                    }
                ],
                "total_connected": 1,
                "all_expected_connected": True,
            },
        ),
    )
    page.route(
        "**/sensors/aliases",
        lambda route: fulfill_json(route, {"aliases": {}}),
    )
    page.route(
        "**/sensors/previews?**",
        lambda route: fulfill_json(route, {"jobs": []}),
    )

    page.goto(f"{console_server.url}/#/devices", wait_until="networkidle")

    card = page.locator('[data-testid="sensor-card"][data-sensor-key="zed_2i:zed-lab"]')
    expect(card.locator('[data-status-tone="informational"]').first).to_contain_text(
        "Capture-ready"
    )
    preview = card.get_by_test_id("sensor-preview-toggle")
    expect(preview).to_be_disabled()
    reason = card.get_by_test_id("sensor-disabled-action-reason")
    expect(reason).to_contain_text(
        "Image-orientation override is available only for RealSense D435 cameras."
    )
    expect(reason).to_contain_text(
        "Live RGB preview is unavailable for this sensor family"
    )
    reason_id = reason.get_attribute("id")
    assert reason_id
    expect(preview).to_have_attribute("aria-describedby", reason_id)


def test_robot_controls_validate_and_confirm_start_and_stop(
    console_server, page
) -> None:
    commands: list[dict] = []
    install_common_mocks(page)

    def command_handler(route) -> None:
        commands.append(route.request.post_data_json)
        fulfill_json(
            route, {"job_id": f"robot-{len(commands)}", "status": "queued"}, status=202
        )

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
    expect(
        page.get_by_text("I authorize motion of the real lab IIWA for this start.")
    ).to_be_visible()
    expect(
        page.get_by_text("I confirm the capture cameras and pose receiver are ready.")
    ).to_be_visible()
    expect(page.get_by_role("button", name="Queue start")).to_be_enabled()
    page.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    page.get_by_role("button", name="Stop IIWA").click()
    stop_warning = page.get_by_test_id("iiwa-stop-warning")
    expect(stop_warning).to_contain_text("IIWA STOP is not a safety stop")
    expect(stop_warning).to_contain_text("cannot interrupt active motion")
    expect(stop_warning).to_contain_text(
        "Sunrise must be restarted manually before another START"
    )
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


def test_dashboard_quick_robot_controls_use_configured_target(
    console_server, page
) -> None:
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
    expect(
        dialog.get_by_text("I authorize motion of the real lab IIWA for this start.")
    ).to_be_visible()
    expect(
        dialog.get_by_text("I confirm the capture cameras and pose receiver are ready.")
    ).to_be_visible()
    expect(dialog.get_by_role("button", name="Queue start")).to_be_enabled()
    dialog.get_by_role("button", name="Queue start").click()
    expect(page.get_by_text("IIWA start queued")).to_be_visible()

    controls.get_by_role("button", name="Stop IIWA").click()
    stop_warning = dialog.get_by_test_id("iiwa-stop-warning")
    expect(stop_warning).to_contain_text("IIWA STOP is not a safety stop")
    expect(stop_warning).to_contain_text("cannot interrupt active motion")
    expect(stop_warning).to_contain_text(
        "Sunrise must be restarted manually before another START"
    )
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


def test_dashboard_prioritizes_monitor_storage_and_job_activity(
    console_server,
    page,
) -> None:
    install_common_mocks(page)
    monitor_start_requests: list[dict] = []

    def monitor_handler(route) -> None:
        if route.request.method == "POST":
            monitor_start_requests.append(route.request.post_data_json)
        route.fallback()

    page.route("**/monitoring/webcam", monitor_handler)

    def job(
        job_id: str,
        name: str,
        status: str,
        *,
        message: str | None = None,
        run_root: str | None = RUN_ROOT,
        scope_kind: str | None = None,
    ) -> dict:
        active = status in {"queued", "running", "canceling"}
        resolved_scope = scope_kind or ("unknown" if run_root is None else "run")
        return {
            "id": job_id,
            "name": name,
            "command": ["uv", "run", "python"],
            "cwd": "/repo",
            "status": status,
            "created_at": "2026-07-27T08:00:00Z",
            "started_at": "2026-07-27T08:00:01Z" if status != "queued" else None,
            "ended_at": None if active else "2026-07-27T08:01:00Z",
            "returncode": None if active else 1 if status == "failed" else 0,
            "message": message,
            "tail": [],
            "resources": ["disk_io"] if status != "queued" else ["cpu"],
            "parameters": {} if run_root is None else {"run_root": run_root},
            "scope_kind": resolved_scope,
            "run_root": run_root if resolved_scope == "run" else None,
            "log_path": f"/tmp/{job_id}.log",
            "visibility": "operator",
        }

    other_run = "/tmp/posetestbot-console/other-run"
    page.route(
        "**/jobs",
        lambda route: fulfill_json(
            route,
            {
                "jobs": [
                    job("running-1", "Synchronize selected run", "running"),
                    job(
                        "queued-1",
                        "Generate BOP annotations",
                        "queued",
                        run_root=other_run,
                    ),
                    job(
                        "failed-1",
                        "Calibration validation",
                        "failed",
                        message="Residual threshold exceeded for wrist camera.",
                        run_root=None,
                    ),
                    job("succeeded-1", "Old completed export", "succeeded"),
                ],
                "resources": {"disk_io": "running-1"},
            },
        ),
    )
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.goto(f"{console_server.url}/#/dashboard", wait_until="networkidle")

    expect(page.get_by_text("Recommended next action", exact=True)).to_have_count(0)
    expect(page.get_by_text("Run ArUco target detection", exact=True)).to_have_count(0)
    expect(
        page.get_by_text(
            "Generate target-pose observations for calibration support.",
            exact=True,
        )
    ).to_have_count(0)

    monitor = page.get_by_test_id("dashboard-room-monitor")
    activity = page.get_by_test_id("dashboard-job-activity")
    expect(monitor).to_be_visible()
    expect(activity).to_be_visible()
    assert monitor_start_requests == []
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and response.url.endswith("/monitoring/webcam")
        )
    ):
        monitor.get_by_role("button", name="Start monitor").click()
    assert monitor_start_requests == [{}]
    monitor_box = monitor.bounding_box()
    activity_box = activity.bounding_box()
    assert monitor_box is not None
    assert activity_box is not None
    assert monitor_box["width"] > activity_box["width"]

    storage = page.get_by_test_id("dashboard-storage")
    expect(storage).to_contain_text("1.25 TiB free")
    expect(storage).to_contain_text("42% free of 3.00 TiB on /tmp")

    expect(activity.get_by_role("heading", name="Active jobs")).to_be_visible()
    expect(activity.get_by_text("Synchronize selected run", exact=True)).to_be_visible()
    expect(activity.get_by_text("Generate BOP annotations", exact=True)).to_be_visible()
    expect(activity.get_by_role("heading", name="Recent failures")).to_be_visible()
    expect(activity.get_by_text("Calibration validation", exact=True)).to_be_visible()
    expect(activity).to_contain_text("Residual threshold exceeded for wrist camera.")
    selected_run_job = activity.get_by_role(
        "link", name="Open Synchronize selected run in Jobs"
    )
    expect(selected_run_job).to_contain_text("Active run")
    expect(selected_run_job).to_contain_text(RUN_ROOT)
    other_run_job = activity.get_by_role(
        "link", name="Open Generate BOP annotations in Jobs"
    )
    expect(other_run_job).to_contain_text("Other run")
    expect(other_run_job).to_contain_text(other_run)
    unscoped_job = activity.get_by_role(
        "link", name="Open Calibration validation in Jobs"
    )
    expect(unscoped_job).to_contain_text("Legacy unknown scope")
    expect(activity.get_by_text("Old completed export", exact=True)).to_have_count(0)
    expect(activity.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )


def test_jobs_log_cancel_and_removed_artifacts_route(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script(
        """
        window.__copiedDebugTexts = [];
        Object.defineProperty(navigator, "clipboard", {
          configurable: true,
          value: {
            writeText: async () => {
              throw new DOMException("Clipboard permission denied", "NotAllowedError");
            },
          },
        });
        document.execCommand = (command) => {
          if (command !== "copy") return false;
          const target = document.activeElement;
          if (!(target instanceof HTMLTextAreaElement)) return false;
          if (!target.closest('[role="dialog"]')) return false;
          window.__copiedDebugTexts.push(
            target.value.slice(target.selectionStart, target.selectionEnd)
          );
          return true;
        };
        """
    )
    canceled: list[str] = []
    job = {
        "id": "capture-1",
        "name": "pipeline:sync_run",
        "command": ["uv"],
        "cwd": "/repo",
        "status": "running",
        "created_at": "2026-07-10T12:00:00Z",
        "log_path": "/tmp/log",
        "started_at": "2026-07-10T12:00:01Z",
        "ended_at": None,
        "returncode": None,
        "message": None,
        "tail": ["working"],
        "resources": ["disk_io"],
        "parameters": {"pipeline_stage": "sync_run", "run_root": RUN_ROOT},
        "scope_kind": "run",
        "run_root": RUN_ROOT,
    }
    page.route(
        "**/jobs?**",
        lambda route: fulfill_json(
            route,
            {
                "jobs": [job],
                "resources": {"disk_io": "capture-1"},
                "total": 1,
                "status_counts": {"running": 1},
                "next_cursor": None,
                "limit": 20,
            },
        ),
    )
    page.route(
        "**/jobs/capture-1/log",
        lambda route: route.fulfill(
            status=200, content_type="text/plain", body="line one\nline two\n"
        ),
    )
    page.route(
        "**/jobs/capture-1/cancel",
        lambda route: (
            canceled.append("capture-1"),
            fulfill_json(route, {"job": {**job, "status": "canceling"}}),
        )[1],
    )
    page.goto(f"{console_server.url}/#/jobs", wait_until="networkidle")
    page.get_by_role("button", name="Log").click()
    expect(page.locator('[data-testid="job-log"]')).to_contain_text("line two")
    page.get_by_role("button", name="Copy output").click()
    expect(page.get_by_text("Job output copied")).to_be_visible()
    page.get_by_role("button", name="Copy context").click()
    expect(page.get_by_text("Job context copied")).to_be_visible()
    copied = page.evaluate("window.__copiedDebugTexts")
    assert copied[0] == "line one\nline two\n"
    context = json.loads(copied[1])
    assert context["schema_version"] == "posetestbot_job_debug_context.v1"
    assert context["job"]["id"] == "capture-1"
    assert context["job"]["parameters"] == {
        "pipeline_stage": "sync_run",
        "run_root": RUN_ROOT,
    }
    assert context["job"]["scope_kind"] == "run"
    assert context["job"]["run_root"] == RUN_ROOT
    assert "tail" not in context["job"]
    page.get_by_role("button", name="Cancel job").click()
    assert canceled == ["capture-1"]

    expect(page.get_by_role("link", name="Artifacts")).to_have_count(0)
    page.goto(f"{console_server.url}/#/artifacts", wait_until="networkidle")
    expect(page).to_have_url(f"{console_server.url}/#/dashboard")


def test_jobs_filters_and_progressively_reveals_history(console_server, page) -> None:
    install_common_mocks(page)
    other_run = "/tmp/posetestbot-console/other-run"
    history = [
        {
            "id": "active-1",
            "name": "active_capture_monitor",
            "command": ["uv"],
            "cwd": "/repo",
            "status": "running",
            "created_at": "2026-07-27T12:00:00Z",
            "log_path": "/tmp/active-1.log",
            "started_at": "2026-07-27T12:00:01Z",
            "ended_at": None,
            "returncode": None,
            "message": "monitoring",
            "tail": [],
            "resources": ["camera:test"],
            "parameters": {"run_root": RUN_ROOT, "cancelable": False},
            "scope_kind": "run",
            "run_root": RUN_ROOT,
        }
    ]
    for index in range(25):
        failed = index == 24
        status = "canceled" if index == 19 else "failed" if failed else "succeeded"
        if index == 24:
            scope_kind, run_root = "unknown", None
        elif index == 23:
            scope_kind, run_root = "run", other_run
        elif index == 22:
            scope_kind, run_root = "run", RUN_ROOT
        elif index == 21:
            scope_kind, run_root = "library", None
        elif index == 20:
            scope_kind, run_root = "global", None
        else:
            scope_kind, run_root = "run", RUN_ROOT
        history.append(
            {
                "id": f"history-{index:02d}",
                "name": "failed_calibration"
                if failed
                else f"completed_job_{index:02d}",
                "command": ["uv"],
                "cwd": "/repo",
                "status": status,
                "created_at": f"2026-07-{index + 1:02d}T12:00:00Z",
                "log_path": f"/tmp/history-{index:02d}.log",
                "started_at": f"2026-07-{index + 1:02d}T12:00:01Z",
                "ended_at": f"2026-07-{index + 1:02d}T12:00:02Z",
                "returncode": 1 if failed else 0,
                "message": "solver evidence failed" if failed else "complete",
                "tail": [],
                "resources": ["cpu"],
                "parameters": {} if run_root is None else {"run_root": run_root},
                "scope_kind": scope_kind,
                "run_root": run_root,
            }
        )

    requests: list[dict[str, list[str]]] = []

    def jobs_handler(route) -> None:
        parameters = parse_qs(urlparse(route.request.url).query)
        requests.append(parameters)
        matching = list(history)
        status_filter = parameters.get("status", ["all"])[0]
        if status_filter == "active":
            matching = [
                job
                for job in matching
                if job["status"] in {"queued", "running", "canceling"}
            ]
        elif status_filter == "failed":
            matching = [job for job in matching if job["status"] == "failed"]
        elif status_filter == "finished":
            matching = [
                job
                for job in matching
                if job["status"] not in {"queued", "running", "canceling"}
            ]
        scope_filter = parameters.get("scope_kind", [None])[0]
        if scope_filter:
            matching = [job for job in matching if job["scope_kind"] == scope_filter]
        run_filter = parameters.get("run_root", [None])[0]
        if run_filter:
            matching = [job for job in matching if job["run_root"] == run_filter]
        search = parameters.get("search", [""])[0].lower()
        if search:
            matching = [
                job
                for job in matching
                if search
                in " ".join(
                    [
                        str(job["id"]),
                        str(job["name"]),
                        str(job["run_root"] or ""),
                        " ".join(job["resources"]),
                    ]
                ).lower()
            ]
        matching.sort(
            key=lambda job: (
                job["status"] in {"queued", "running", "canceling"},
                job["created_at"],
            ),
            reverse=True,
        )
        counts: dict[str, int] = {}
        for job in matching:
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        terminal = [
            job
            for job in matching
            if job["status"] not in {"queued", "running", "canceling"}
        ]
        active = [
            job
            for job in matching
            if job["status"] in {"queued", "running", "canceling"}
        ]
        cursor = parameters.get("cursor", [None])[0]
        page_jobs = terminal[20:40] if cursor else [*active, *terminal[:20]]
        next_cursor = (
            "opaque-terminal-page-2" if cursor is None and len(terminal) > 20 else None
        )
        fulfill_json(
            route,
            {
                "jobs": page_jobs,
                "resources": {"camera:test": "active-1"} if active else {},
                "total": len(matching),
                "status_counts": counts,
                "next_cursor": next_cursor,
                "limit": 20,
            },
        )

    page.route("**/jobs?**", jobs_handler)

    page.goto(f"{console_server.url}/#/jobs", wait_until="networkidle")

    expect(page.get_by_text("Lab-wide job runner", exact=True)).to_be_visible()
    unscoped_entry = page.get_by_test_id("job-card-history-24")
    expect(
        unscoped_entry.get_by_text("Legacy unknown scope", exact=True)
    ).to_be_visible()
    other_entry = page.get_by_test_id("job-card-history-23")
    expect(other_entry.get_by_text("Other run", exact=True)).to_be_visible()
    expect(other_entry).to_contain_text(other_run)
    selected_entry = page.get_by_test_id("job-card-history-22")
    expect(selected_entry.get_by_text("Active run", exact=True)).to_be_visible()
    expect(selected_entry).to_contain_text(RUN_ROOT)
    expect(
        page.get_by_test_id("job-card-history-21").get_by_text(
            "Reusable library", exact=True
        )
    ).to_be_visible()
    expect(
        page.get_by_test_id("job-card-history-20").get_by_text("Lab-wide", exact=True)
    ).to_be_visible()
    expect(page.get_by_role("button", name="Log")).to_have_count(21)
    expect(
        page.get_by_test_id("job-card-active-1")
        .locator('[data-status-tone="warning"]')
        .first
    ).to_contain_text("running")
    expect(page.get_by_test_id("job-card-active-1")).to_contain_text("non-cancelable")
    expect(
        page.get_by_test_id("job-card-active-1").get_by_role(
            "button", name="Cancel", exact=True
        )
    ).to_have_count(0)
    expect(
        page.get_by_test_id("job-card-history-22")
        .locator('[data-status-tone="success"]')
        .first
    ).to_contain_text("succeeded")
    expect(
        unscoped_entry.locator('[data-status-tone="destructive"]').first
    ).to_contain_text("failed")
    page.get_by_role("button", name="Load older jobs").click()
    expect(page.get_by_role("button", name="Log")).to_have_count(26)
    expect(
        page.get_by_test_id("job-card-history-19")
        .locator('[data-status-tone="neutral"]')
        .first
    ).to_contain_text("canceled")

    page.get_by_role("combobox", name="Filter jobs by scope").click()
    page.get_by_role("option", name="Reusable library").click()
    expect(page.get_by_role("button", name="Log")).to_have_count(1)
    expect(page.get_by_text("completed_job_21", exact=True)).to_be_visible()

    page.get_by_role("combobox", name="Filter jobs by scope").click()
    page.get_by_role("option", name="All scopes").click()
    page.get_by_role("combobox", name="Filter jobs by status").click()
    page.get_by_role("option", name="Failed (1)").click()
    expect(page.get_by_text("failed_calibration", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Log")).to_have_count(1)
    page.get_by_label("Search jobs").fill("no such job")
    expect(page.get_by_role("heading", name="No matching jobs")).to_be_visible()
    page.get_by_role("button", name="Clear filters").click()
    expect(page.get_by_role("button", name="Log")).to_have_count(26)
    assert any(
        request.get("cursor") == ["opaque-terminal-page-2"] for request in requests
    )
    assert any(request.get("scope_kind") == ["library"] for request in requests)
    assert any(request.get("status") == ["failed"] for request in requests)


def test_calibration_target_unavailable_keeps_saved_library_navigation(
    console_server, page
) -> None:
    install_common_mocks(page, generator_available=False)
    page.route(
        "**/calibration-targets/bundles?**",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "calibration_target_library.v1",
                "run_root": RUN_ROOT,
                "bundles": [],
            },
        ),
    )
    page.goto(console_server.url, wait_until="networkidle")

    expect(page.get_by_role("link", name="Calibration Targets")).to_be_visible()
    page.goto(f"{console_server.url}/#/calibration-targets", wait_until="networkidle")
    expect(page.get_by_text("Target generation is unavailable")).to_be_visible()
    expect(page.get_by_text("Saved target library")).to_be_visible()
    expect(
        page.get_by_text("git submodule update --init third_party/PoseGridGen")
    ).to_be_visible()


def test_calibration_workflow_explains_intrinsics_and_saves_complete_bundle(
    console_server, page
) -> None:
    requests: list[dict] = []
    promotion_status: dict[str, str | None] = {"value": None}
    install_common_mocks(page)
    setup = {
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
            },
            {
                "sensor_key": "oak_d_pro:static-1",
                "sensor_name": "luxonis_static-1",
                "display_name": "Auxiliary OAK-D",
                "sensor_type": "oak_d_pro",
                "device_id": "static-1",
                "current_mounting_mode": "eye_in_hand",
            },
        ],
        "unavailable_cameras": [],
        "saved_targets": [
            {
                "target_id": "5f09f41c-dd91-44ef-a048-1f43fc990e17",
                "display_name": "Lab board",
                "valid": True,
                "selected": True,
            },
            {
                "target_id": "9ab5ff1c-60f6-46b1-823d-2a912d5d4e3f",
                "display_name": "Alternate board",
                "valid": True,
            },
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
            "default_extrinsic_methods": [
                "tsai",
                "park",
                "horaud",
                "andreff",
                "daniilidis",
                "shah",
                "li",
            ],
            "intrinsics_policy": "compare_factory_opencv",
            "intrinsics_policies": [
                {
                    "id": "compare_factory_opencv",
                    "label": "Compare captured factory intrinsics with a gated OpenCV calibration",
                },
                {
                    "id": "reuse_compatible_or_factory",
                    "label": "Reuse an exact compatible profile, otherwise captured factory intrinsics",
                },
            ],
            "synchronization": {
                "implementation_revision": "constant_latency_nearest_pose_motion_lomo_warn_fallback.v3",
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
                    "minimum_robot_pose_time_offset_ms": -300.0,
                    "maximum_robot_pose_time_offset_ms": 300.0,
                    "step_ms": 5.0,
                    "max_nearest_pose_delta_ms": 150.0,
                    "warning_nearest_pose_delta_ms": 20.0,
                    "warning_absolute_robot_pose_time_offset_ms": 150.0,
                    "time_offset_failure_policy": "warn_keep_zero",
                    "minimum_motion_count_per_cross_validation_fold": 4,
                    "maximum_leave_one_motion_out_search_adjusted_sign_p_value": 0.05,
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
                "image_coverage_tail_support_views": 5,
                "min_image_centroid_x_span_ratio": 0.45,
                "min_image_centroid_y_span_ratio": 0.35,
                "min_image_centroid_hull_area_ratio": 0.1,
                "max_per_view_reprojection_error_px": 3.0,
                "max_intrinsic_rms_reprojection_error_px": 1.5,
                "min_motion_poses": 4,
                "min_translation_span_mm": 20.0,
                "min_rotation_span_deg": 5.0,
                "min_rotation_axis_second_to_first_ratio": 0.15,
                "max_nearest_pose_delta_ms": 150.0,
                "warning_nearest_pose_delta_ms": 20.0,
            },
        },
        "latest_attempt": None,
    }
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))

    def create_handler(route) -> None:
        requests.append(
            {"path": "/calibration/attempts", "body": route.request.post_data_json}
        )
        fulfill_json(
            route,
            {"attempt_id": "a" * 32, "job_id": "calculation-1", "status": "queued"},
            status=202,
        )

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
        "companion_transform": {
            **transform,
            "from": "aruco_grid",
            "to": "template_base",
        },
        "held_out_residuals": {
            "mean_translation_mm": 0.8,
            "median_translation_mm": 0.7,
            "mean_rotation_deg": 0.3,
            "median_rotation_deg": 0.2,
        },
    }
    override = {
        **recommended,
        "candidate_id": "realsense_d435:wrist-1|SQPNP|tsai",
        "profile_id": "wrist_sqpnp_tsai",
        "pnp_method": "SQPNP",
        "extrinsic_method": "tsai",
        "recommended": False,
        "score": 0.2,
    }
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
        "candidate_id": "oak_d_pro:static-1|ITERATIVE|li",
        "pnp_method": "ITERATIVE",
        "extrinsic_method": "li",
        "algorithms": ["ITERATIVE", "li"],
        "status": "error",
        "validation_state": "failed",
        "score": None,
        "observation_count": 3,
        "inlier_count": 0,
        "outlier_count": 3,
        "outlier_ratio": 1,
        "error": "leave-one-pose-out validation requires at least four poses",
    }

    def attempt_payload() -> dict:
        return {
            "schema_version": "calibration_attempt.v1",
            "attempt_id": "a" * 32,
            "request": {
                "mode": "eye_in_hand",
                "sensor_keys": ["realsense_d435:wrist-1", "oak_d_pro:static-1"],
                "target_id": setup["saved_targets"][0]["target_id"],
                "solver_policy": "auto_compare",
                "intrinsics_policy": "compare_factory_opencv",
                "synchronization_policy": "auto_offset",
            },
            "progress": {
                "status": "complete",
                "message": "Calibration calculations are complete and awaiting review.",
                "phases": [
                    {
                        "id": "prepare_data",
                        "label": "Prepare data",
                        "status": "complete",
                    },
                    {
                        "id": "estimate_target_poses",
                        "label": "Estimate target poses",
                        "status": "complete",
                    },
                    {
                        "id": "estimate_time_offsets",
                        "label": "Estimate time alignment",
                        "status": "complete",
                    },
                    {
                        "id": "compare_robot_camera_solutions",
                        "label": "Compare robot-camera solutions",
                        "status": "complete",
                    },
                    {
                        "id": "validate_and_rank",
                        "label": "Validate and rank",
                        "status": "complete",
                    },
                ],
            },
            "results": {
                "status": "complete",
                "recommended_camera_count": 2,
                "failed_camera_count": 0,
                "results": [
                    {
                        **setup["cameras"][0],
                        "status": "passing",
                        "recommended_candidate_id": recommended["candidate_id"],
                        "recommendation": recommended,
                        "candidates": [recommended, override],
                    },
                    {
                        **setup["cameras"][1],
                        "status": "passing",
                        "recommended_candidate_id": static_recommended["candidate_id"],
                        "recommendation": static_recommended,
                        "candidates": [static_recommended, failed],
                    },
                ],
            },
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
                "implementation_revision": "constant_latency_nearest_pose_motion_lomo_warn_fallback.v3",
                "policy": "auto_offset",
                "status": "complete",
                "sign_convention": {
                    "operator_equation": "robot_pose_query_time = frame_time + offset",
                    "positive_operator_value": "pair the frame with a robot pose recorded later",
                    "conversion": "sync_delta_ms = -robot_pose_time_offset_ms",
                },
                "search": {
                    "minimum_robot_pose_time_offset_ms": -300.0,
                    "maximum_robot_pose_time_offset_ms": 300.0,
                    "step_ms": 5.0,
                    "max_nearest_pose_delta_ms": 150.0,
                    "warning_nearest_pose_delta_ms": 20.0,
                    "warning_absolute_robot_pose_time_offset_ms": 150.0,
                    "time_offset_failure_policy": "warn_keep_zero",
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
                        "selection_extrinsic_method": "shah",
                        "improvement_evidence_strategy": "leave_one_motion_out_consistency",
                        "split": {
                            "motion_count": 17,
                            "selected_observation_count": 102,
                            "fold_motion_counts": {"0": 6, "1": 6, "2": 5},
                        },
                        "cross_validation": {
                            "zero_offset": {
                                "residuals": {
                                    "mean_translation_mm": 3.91,
                                    "median_translation_mm": 3.8,
                                    "max_translation_mm": 6.0,
                                    "mean_rotation_deg": 0.42,
                                    "median_rotation_deg": 0.4,
                                    "max_rotation_deg": 0.8,
                                }
                            },
                            "candidate": {
                                "residuals": {
                                    "mean_translation_mm": 2.77,
                                    "median_translation_mm": 2.6,
                                    "max_translation_mm": 4.8,
                                    "mean_rotation_deg": 0.39,
                                    "median_rotation_deg": 0.37,
                                    "max_rotation_deg": 0.7,
                                }
                            },
                            "improvement": {
                                "absolute_translation_mm": 1.14,
                                "relative_translation": 0.29156,
                                "rotation_change_deg": -0.03,
                            },
                        },
                        "motion_consistency": {
                            "status": "ok",
                            "strategy": "leave_one_motion_out_candidate_consistency_bonferroni.v1",
                            "motion_count": 17,
                            "candidate_search_adjustment": "bonferroni",
                            "candidate_search_hypothesis_count": 120,
                            "methods": {
                                "shah": {
                                    "status": "ok",
                                    "motion_count": 17,
                                    "positive_motion_count": 17,
                                    "material_motion_count": 16,
                                    "positive_sign_p_value": 0.0000076294,
                                    "candidate_search_adjusted_positive_sign_p_value": 0.000457764,
                                    "median_improvement": {
                                        "absolute_translation_mm": 0.811,
                                        "relative_translation": 0.2864,
                                        "rotation_change_deg": -0.02,
                                    },
                                },
                                "li": {
                                    "status": "ok",
                                    "motion_count": 17,
                                    "positive_motion_count": 16,
                                    "material_motion_count": 16,
                                    "positive_sign_p_value": 0.000137329,
                                    "candidate_search_adjusted_positive_sign_p_value": 0.00823974,
                                    "median_improvement": {
                                        "absolute_translation_mm": 0.792,
                                        "relative_translation": 0.2941,
                                        "rotation_change_deg": -0.018,
                                    },
                                },
                            },
                            "thresholds": {
                                "minimum_median_absolute_translation_mm": 0.25,
                                "minimum_median_relative_translation": 0.1,
                                "maximum_search_adjusted_positive_sign_p_value": 0.05,
                            },
                        },
                        "checks": [
                            {
                                "name": "cross_validation_offset_stability",
                                "status": "ok",
                                "actual": 10.0,
                                "threshold": 22.0,
                            },
                        ],
                        "curve": [
                            {
                                "robot_pose_time_offset_ms": 0.0,
                                "residuals": {
                                    "mean_translation_mm": 3.91,
                                    "median_translation_mm": 3.8,
                                    "max_translation_mm": 6.0,
                                    "mean_rotation_deg": 0.42,
                                    "median_rotation_deg": 0.4,
                                    "max_rotation_deg": 0.8,
                                },
                            },
                            {
                                "robot_pose_time_offset_ms": 65.0,
                                "residuals": {
                                    "mean_translation_mm": 2.77,
                                    "median_translation_mm": 2.6,
                                    "max_translation_mm": 4.8,
                                    "mean_rotation_deg": 0.39,
                                    "median_rotation_deg": 0.37,
                                    "max_rotation_deg": 0.7,
                                },
                            },
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
                        "split": {
                            "motion_count": 15,
                            "selected_observation_count": 90,
                            "fold_motion_counts": {"0": 5, "1": 5, "2": 5},
                        },
                        "cross_validation": {
                            "zero_offset": {
                                "residuals": {
                                    "mean_translation_mm": 4.7,
                                    "median_translation_mm": 4.5,
                                    "max_translation_mm": 7.0,
                                    "mean_rotation_deg": 0.5,
                                    "median_rotation_deg": 0.45,
                                    "max_rotation_deg": 0.9,
                                }
                            },
                            "candidate": {
                                "residuals": {
                                    "mean_translation_mm": 3.0,
                                    "median_translation_mm": 2.8,
                                    "max_translation_mm": 5.0,
                                    "mean_rotation_deg": 0.4,
                                    "median_rotation_deg": 0.38,
                                    "max_rotation_deg": 0.8,
                                }
                            },
                            "improvement": {
                                "absolute_translation_mm": 1.7,
                                "relative_translation": 0.3617,
                                "rotation_change_deg": -0.1,
                            },
                        },
                        "checks": [
                            {
                                "name": "reference_method_sensitivity",
                                "status": "warning",
                                "actual": 28.0,
                                "warning_threshold": 22.0,
                                "failure_threshold": 44.0,
                            }
                        ],
                        "curve": [],
                    },
                ],
            },
            "promotion": (
                {
                    "status": "promoted",
                    "promoted_profile_ids": ["wrist_sqpnp_tsai", "static_ippe_park"],
                }
                if promotion_status["value"] == "promoted"
                else {
                    "status": promotion_status["value"],
                    "job_id": "promotion-1",
                }
                if promotion_status["value"] in {"queued", "running"}
                else None
            ),
        }

    page.route(
        "**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa?**",
        lambda route: fulfill_json(route, attempt_payload()),
    )

    def promote_handler(route) -> None:
        requests.append(
            {"path": "/calibration/promote", "body": route.request.post_data_json}
        )
        promotion_status["value"] = "queued"
        fulfill_json(
            route,
            {
                "attempt_id": "a" * 32,
                "job_id": "promotion-1",
                "status": "queued",
                "selections": route.request.post_data_json["candidate_ids"],
            },
            status=202,
        )

    page.route(
        "**/calibration/attempts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/promote",
        promote_handler,
    )
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=calculate",
        wait_until="networkidle",
    )

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
    page.get_by_text("Time-alignment search limits and warning policy").click()
    expect(page.get_by_test_id("calibration-synchronization-policy")).to_contain_text(
        "At least 12 eligible motion groups are requested"
    )
    expect(page.get_by_test_id("calibration-synchronization-policy")).to_contain_text(
        "above 20 ms are warnings; matches remain usable through 150 ms"
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
    expect(
        page.get_by_test_id("calibration-workflow").get_by_text("Lab board", exact=True)
    ).to_be_visible()
    expect(page.get_by_role("button", name="Analyze recording")).to_be_enabled()
    page.get_by_role("button", name="Analyze recording").click()
    expect(page.get_by_text("Calibration queued")).to_be_visible()
    assert requests[0]["body"]["mode"] == "eye_in_hand"
    assert requests[0]["body"]["sensor_keys"] == [
        "realsense_d435:wrist-1",
        "oak_d_pro:static-1",
    ]
    assert requests[0]["body"]["target_id"] == "5f09f41c-dd91-44ef-a048-1f43fc990e17"
    assert requests[0]["body"]["intrinsics_policy"] == "compare_factory_opencv"
    assert requests[0]["body"]["synchronization_policy"] == "auto_offset"

    expect(page.get_by_text("Prepare data")).to_be_visible()
    expect(page.locator('[data-phase-id="estimate_time_offsets"]')).to_contain_text(
        "Estimate time alignment"
    )
    attempt_job = page.get_by_test_id("calibration-attempt-job-status")
    expect(attempt_job.get_by_test_id("calibration-duration-guidance")).to_contain_text(
        "three-camera comparison usually takes 10–20 minutes"
    )
    expect(attempt_job).to_contain_text("background work continues after navigation")
    expect(attempt_job.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    expect(page.get_by_test_id("calibration-results")).to_be_visible()
    alignment = page.get_by_test_id("calibration-time-alignment")
    expect(alignment).to_contain_text(
        "not evidence that the hardware clocks are synchronized"
    )
    expect(page.get_by_test_id("calibration-time-alignment-warning")).to_contain_text(
        "Calibration continued with timing warnings"
    )
    page.mouse.move(0, 0)
    alignment.get_by_role(
        "button", name="About robot-pose time-offset evidence"
    ).hover()
    expect(page.get_by_role("tooltip")).to_contain_text(
        "positive offset uses a later robot pose"
    )
    expect(
        alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')
    ).to_contain_text("+65.0 ms")
    expect(
        alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')
    ).to_contain_text("3.910 → 2.770 mm")
    expect(
        alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')
    ).to_contain_text("29.2%")
    expect(
        page.get_by_test_id("timing-motion-summary-realsense_d435:wrist-1")
    ).to_contain_text("17/17 held-out motions improved")
    oak_alignment = alignment.locator('[data-time-offset-sensor="oak_d_pro:static-1"]')
    expect(oak_alignment).to_contain_text("Applied with 1 warning")
    expect(oak_alignment).to_contain_text("reference method sensitivity")
    alignment.get_by_text("Advanced offset evidence · Wrist RGB-D").click()
    expect(alignment).to_contain_text("cross validation offset stability")
    motion_consistency = page.get_by_test_id(
        "timing-motion-consistency-realsense_d435:wrist-1"
    )
    expect(motion_consistency).to_contain_text("Bonferroni-corrected")
    expect(motion_consistency).to_contain_text("120 nonzero offset candidates")
    expect(motion_consistency).to_contain_text("16/17")
    assert page.evaluate(
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )
    expect(page.get_by_test_id("calibration-acceptance-thresholds")).to_contain_text(
        "≥15 accepted views"
    )
    expect(page.get_by_test_id("calibration-acceptance-thresholds")).to_contain_text(
        "≥45% of image width"
    )
    expect(page.get_by_test_id("calibration-acceptance-thresholds")).to_contain_text(
        "3 × 3 centroid-cell count remains diagnostic"
    )
    wrist_intrinsics = page.get_by_test_id(
        "intrinsic-comparison-realsense_d435:wrist-1"
    )
    expect(wrist_intrinsics).to_contain_text("Using OpenCV estimate")
    expect(wrist_intrinsics).to_contain_text(
        "Factory values come from the camera SDK. The OpenCV estimate is fitted from this recording's grid views."
    )
    expect(wrist_intrinsics).to_contain_text("OpenCV training views / image coverage")
    expect(wrist_intrinsics).to_contain_text("18 views · 6 of 9 regions")
    static_intrinsics = page.get_by_test_id("intrinsic-comparison-oak_d_pro:static-1")
    expect(static_intrinsics).to_contain_text("Using factory SDK values")
    expect(static_intrinsics).to_contain_text("The factory SDK values are compatible")
    expect(static_intrinsics).to_contain_text("coverage 2/9 is below 6/9")
    expect(page.get_by_text("camera → robot_flange").last).to_be_visible()
    expect(
        page.get_by_text("All attempted solutions and failures").first
    ).to_be_visible()
    wrist_result = page.locator('[data-camera-key="realsense_d435:wrist-1"]')
    wrist_result.get_by_text("Alternative solution (advanced)", exact=True).click()
    wrist_result.get_by_label("Alternative solution", exact=True).click()
    page.get_by_role("option", name="SQPNP + tsai · score 0.2000").click()
    page.get_by_role("button", name="Save selected calibrations").click()
    expect(page.get_by_text("Calibration acceptance queued")).to_be_visible()
    promotion_job = page.get_by_test_id("calibration-promotion-job-status")
    expect(promotion_job).to_contain_text("continues after navigation")
    expect(promotion_job.get_by_role("link", name="Open Jobs")).to_have_attribute(
        "href", "#/jobs"
    )
    assert requests[-1]["body"]["candidate_ids"] == {
        "realsense_d435:wrist-1": override["candidate_id"],
        "oak_d_pro:static-1": static_recommended["candidate_id"],
    }
    promotion_status["value"] = "promoted"
    expect(page.get_by_role("button", name="Calibrations saved")).to_be_visible()
    expect(page.get_by_text("Saved 2 camera profile(s).")).to_be_visible()


def calibration_time_alignment_setup(
    *,
    latest_attempt_id: str | None,
    latest_status: str = "complete",
    implementation_revision: str | None = (
        "constant_latency_nearest_pose_motion_lomo_warn_fallback.v3"
    ),
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
                "implementation_revision": implementation_revision,
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
                    "minimum_robot_pose_time_offset_ms": -300.0,
                    "maximum_robot_pose_time_offset_ms": 300.0,
                    "step_ms": 5.0,
                    "max_nearest_pose_delta_ms": 150.0,
                    "warning_nearest_pose_delta_ms": 20.0,
                    "warning_absolute_robot_pose_time_offset_ms": 150.0,
                    "time_offset_failure_policy": "warn_keep_zero",
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
                "max_nearest_pose_delta_ms": 150.0,
                "warning_nearest_pose_delta_ms": 20.0,
            },
        },
        "latest_attempt": latest_attempt,
    }


def test_calibration_workflow_blocks_stale_backend_timing_revision(
    console_server,
    page,
) -> None:
    setup = calibration_time_alignment_setup(
        latest_attempt_id=None,
        implementation_revision="constant_latency_nearest_pose_motion_cv.v1",
    )
    install_common_mocks(page)
    page.route("**/calibration/setup?**", lambda route: fulfill_json(route, setup))

    page.goto(
        f"{console_server.url}/#/workflow/calibration?step=calculate",
        wait_until="networkidle",
    )

    warning = page.get_by_test_id("calibration-backend-restart-required")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Backend restart required")
    expect(warning).to_contain_text("constant_latency_nearest_pose_motion_cv.v1")
    expect(warning).to_contain_text(
        "constant_latency_nearest_pose_motion_lomo_warn_fallback.v3"
    )
    expect(page.get_by_role("button", name="Analyze recording")).to_be_disabled()
    expect(
        page.get_by_text(
            "Restart the PoseTestBot backend and reload this page to use the "
            "current Auto time-alignment rule."
        )
    ).to_be_visible()


def test_calibration_workflow_explains_immutable_legacy_timing_attempt(
    console_server,
    page,
) -> None:
    attempt_id = "e" * 32
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
            message="ValueError: Auto-sync evidence failed closed",
        ),
        "results": None,
        "intrinsic_comparison": None,
        "time_offset_search": {
            "implementation_revision": "constant_latency_nearest_pose_motion_cv.v1",
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
                    "status": "failed",
                    "decision_reason": "candidate_failed_safety_or_stability_checks",
                    "selected_robot_pose_time_offset_ms": 0.0,
                    "selected_sync_delta_ms": 0.0,
                    "candidate_robot_pose_time_offset_ms": 45.0,
                    "evidence_strength": "insufficient",
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

    expect(page.get_by_test_id("calibration-backend-restart-required")).to_have_count(0)
    warning = page.get_by_test_id("calibration-attempt-legacy-timing-revision")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("This attempt used an obsolete timing rule")
    expect(warning).to_contain_text("cannot be upgraded in place")
    expect(page.get_by_test_id("calibration-time-alignment-failed")).to_contain_text(
        "decided by the recorded legacy rule"
    )


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
            "implementation_revision": "constant_latency_nearest_pose_motion_lomo_cv.v2",
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
                    "decision_reason": ("candidate_failed_safety_or_stability_checks"),
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
    row = alignment.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')
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
        "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
    )


def test_degraded_auto_sync_warning_keeps_zero_and_shows_solver_results(
    console_server,
    page,
) -> None:
    attempt_id = "d" * 32
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
            "synchronization_policy": "auto_offset",
        },
        "progress": calibration_attempt_progress(
            status="complete",
            time_alignment_status="complete",
            message=(
                "Calibration calculations are complete with timing warnings "
                "and are awaiting review."
            ),
        ),
        "results": calibration_failed_results(setup["cameras"][0]),
        "intrinsic_comparison": None,
        "time_offset_search": {
            "implementation_revision": (
                "constant_latency_nearest_pose_motion_lomo_warn_fallback.v3"
            ),
            "policy": "auto_offset",
            "status": "complete",
            "warning_sensor_keys": ["realsense_d435:wrist-1"],
            "warning_sensor_count": 1,
            "sign_convention": {
                "operator_equation": "robot_pose_query_time = frame_time + offset",
                "positive_operator_value": (
                    "pair the frame with a robot pose recorded later"
                ),
                "conversion": "sync_delta_ms = -robot_pose_time_offset_ms",
            },
            "search": setup["solver"]["synchronization"]["search"],
            "sensors": [
                {
                    "sensor_key": "realsense_d435:wrist-1",
                    "sensor_name": "realsense_wrist-1",
                    "display_name": "Wrist RGB-D",
                    "status": "kept_zero",
                    "decision_reason": "time_offset_search_warning_fallback",
                    "selected_robot_pose_time_offset_ms": 0.0,
                    "selected_sync_delta_ms": 0.0,
                    "candidate_robot_pose_time_offset_ms": 0.0,
                    "evidence_strength": "degraded",
                    "warning_fallback_used": True,
                    "boundary_hit": False,
                    "checks": [
                        {
                            "name": "time_offset_search_execution",
                            "status": "warning",
                            "actual": (
                                "ValueError: auto sync requires at least 12 "
                                "motion groups"
                            ),
                        },
                        {
                            "name": "nearest_pose_delta_warning",
                            "status": "warning",
                            "actual": {
                                "maximum_abs_nearest_pose_delta_ms": 100.0,
                            },
                            "warning_threshold": 20.0,
                            "failure_threshold": 150.0,
                        },
                    ],
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

    warning = page.get_by_test_id("calibration-time-alignment-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Calibration continued with timing warnings")
    expect(page.get_by_test_id("calibration-time-alignment-failed")).to_have_count(0)
    expect(page.get_by_test_id("calibration-results")).to_be_visible()
    row = page.locator('[data-time-offset-sensor="realsense_d435:wrist-1"]')
    expect(row).to_contain_text("Recorded timing kept with warning")
    expect(row).to_contain_text("Applied +0.0 ms")
    expect(row).to_contain_text("time offset search execution")
    expect(row).to_contain_text("nearest pose delta warning")


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
            "implementation_revision": "constant_latency_nearest_pose_motion_lomo_cv.v2",
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
    expect(alignment).to_contain_text("not reusable for a new dataset")
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
    page.route(
        "**/calibration-targets/capabilities",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "posegridgen_capabilities.v1",
                "paper_sizes_mm": {
                    "A4": [210, 297],
                    "A3": [297, 420],
                    "A5": [148, 210],
                    "A6": [105, 148],
                },
                "dictionaries": {"DICT_5X5_50": 50},
                "defaults": configuration,
            },
        ),
    )

    def bundle_payload(run_root: str) -> dict:
        selected = run_root in selected_runs
        return {
            "schema_version": "calibration_target_library.v1",
            "run_root": run_root,
            "replacement_blockers": (
                ["calibration_observations.json"] if run_root in locked_runs else []
            ),
            "bundles": []
            if deleted["value"]
            else [
                {
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
                }
            ],
        }

    def library_handler(route) -> None:
        library_urls.append(route.request.url)
        run_root = route.request.url.split("run_root=", 1)[-1]
        run_root = run_root.replace("%2F", "/")
        fulfill_json(route, bundle_payload(run_root))

    page.route("**/calibration-targets/bundles?**", library_handler)
    png = cv2.imencode(".png", np.full((12, 16, 3), 220, dtype=np.uint8))[1].tobytes()

    def preview_handler(route) -> None:
        requests.append(
            {
                "path": "/calibration-targets/preview",
                "body": route.request.post_data_json,
            }
        )
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
        fulfill_json(
            route,
            {"request": body, "adjusted": False, "scale_factor": 1, "changes": []},
        )

    page.route("**/calibration-targets/fit", fit_handler)

    def generate_handler(route) -> None:
        requests.append(
            {
                "path": "/calibration-targets/generate",
                "body": route.request.post_data_json,
            }
        )
        fulfill_json(
            route,
            {"job_id": "generate-1", "job": {"id": "generate-1", "status": "queued"}},
            status=202,
        )

    page.route("**/calibration-targets/generate", generate_handler)
    page.route(
        "**/jobs/generate-1",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "generate-1",
                    "status": "succeeded",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )
    page.route(
        "**/jobs/select-1",
        lambda route: fulfill_json(
            route,
            {
                "job": {
                    "id": "select-1",
                    "status": "succeeded",
                    "message": None,
                    "tail": [],
                }
            },
        ),
    )

    def select_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/calibration-targets/select", "body": body})
        selected_runs.add(body["run_root"])
        locked_runs.add(body["run_root"])
        fulfill_json(
            route,
            {"job_id": "select-1", "job": {"id": "select-1", "status": "queued"}},
            status=202,
        )

    page.route(f"**/calibration-targets/bundles/{target_id}/select", select_handler)

    def delete_handler(route) -> None:
        body = route.request.post_data_json
        requests.append({"path": "/calibration-targets/delete", "body": body})
        deleted["value"] = True
        fulfill_json(
            route,
            {
                "status": "deleted",
                "target_id": target_id,
                "display_name": "Anisotropic calibration board",
            },
        )

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
    expect(
        page.get_by_role("img", name="Calibration target preview", exact=True)
    ).to_be_visible()
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

    paper = page.get_by_role("combobox", name="Paper")
    paper.click()
    expect(page.get_by_role("option", name="A5", exact=True)).to_be_visible()
    expect(page.get_by_role("option", name="A6", exact=True)).to_be_visible()
    page.get_by_role("option", name="A5", exact=True).click()
    expect(page.get_by_text("210 × 148 mm", exact=True)).to_be_visible()
    paper.click()
    page.get_by_role("option", name="A6", exact=True).click()
    expect(page.get_by_text("148 × 105 mm", exact=True)).to_be_visible()
    paper.click()
    page.get_by_role("option", name="A4", exact=True).click()
    expect(page.get_by_text("297 × 210 mm", exact=True)).to_be_visible()

    assert any(item["path"] == "/calibration-targets/preview" for item in requests)
    assert library_preview_urls

    page.get_by_role("button", name="Fit to page").click()
    expect(page.get_by_text("Board fitted to the selected page")).to_be_visible()
    page.get_by_label("Target display name").fill("Printed target 01")
    page.get_by_role("button", name="Generate bundle").click()
    expect(page.get_by_text("Calibration target generated")).to_be_visible()
    generated = next(
        item["body"]
        for item in requests
        if item["path"] == "/calibration-targets/generate"
    )
    assert generated["display_name"] == "Printed target 01"
    assert generated["configuration"]["print_compensation"] == {
        "x_percent": 101,
        "y_percent": 99,
    }
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
    selection = [
        item["body"]
        for item in requests
        if item["path"] == "/calibration-targets/select"
    ][-1]
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
    expect(
        page.get_by_text("Placement is fixed only for this completed run")
    ).to_be_visible()
    page.get_by_role("dialog").get_by_role(
        "button", name="Close", exact=True
    ).first.click()
    assert (
        len(
            [item for item in requests if item["path"] == "/calibration-targets/select"]
        )
        == select_request_count
    )
    expect(
        page.get_by_role("button", name="Delete Anisotropic calibration board")
    ).to_be_disabled()

    page.get_by_role("combobox", name="Active run folder").click()
    page.get_by_role("option", name="old-run · sync_aruco").click()
    expect(page.get_by_text("Active for this run", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Select for run")).to_be_visible()
    assert any("old-run" in url for url in library_urls)

    page.get_by_role("button", name="Delete Anisotropic calibration board").click()
    expect(
        page.get_by_role("heading", name="Delete Anisotropic calibration board?")
    ).to_be_visible()
    assert not any(item["path"] == "/calibration-targets/delete" for item in requests)
    page.get_by_role("button", name="Confirm delete").click()
    expect(page.get_by_text("Calibration target deleted")).to_be_visible()
    expect(
        page.get_by_role("heading", name="Anisotropic calibration board")
    ).to_have_count(0)
    deletion = next(
        item["body"]
        for item in requests
        if item["path"] == "/calibration-targets/delete"
    )
    assert deletion == {"run_root": old_run, "confirm": True}


def cell_scene_payload(
    *, objectless: bool = False, camera_frames_available: bool = False
) -> dict:
    identity = {
        "semantics": "entity_to_parent",
        "parent_frame": "template_base",
        "translation_mm": [0, 0, 0],
        "rotation_quaternion_wxyz": [1, 0, 0, 0],
    }
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
            {
                "id": "template_base",
                "type": "reference_frame",
                "label": "Template base",
                "status": "planned",
                "transform": {**identity, "parent_frame": None},
                "unresolved_reason": None,
                "geometry": {"kind": "axes", "size_mm": 100},
                "provenance": {"source": "config"},
            },
            {
                "id": "robot_flange",
                "type": "robot_flange",
                "label": "Robot flange",
                "status": "recorded",
                "transform": identity,
                "unresolved_reason": None,
                "geometry": {"kind": "flange_proxy"},
                "provenance": {"source": "match_robot_ee_poses.json"},
            },
            {
                "id": "camera:realsense_123",
                "type": "camera",
                "label": "Wrist D435",
                "status": "planned",
                "transform": {
                    **identity,
                    "parent_frame": "robot_flange",
                    "translation_mm": [10, 20, -500],
                },
                "unresolved_reason": None,
                "geometry": {
                    "kind": "camera_frustum",
                    "width": 1280,
                    "height": 720,
                    "fx": 900,
                    "fy": 900,
                    "cx": 640,
                    "cy": 360,
                },
                "provenance": {
                    "source": "calibration_profiles.json",
                    "profile_id": "wrist-profile",
                },
                "calibration": {
                    "profile_id": "wrist-profile",
                    "schema_version": "calibration.v2",
                    "status": "valid",
                    "mounting_mode": "eye_in_hand",
                    "rig_position": "wrist",
                    "extrinsics": {
                        "from": "camera",
                        "to": "robot_flange",
                        "matrix": [
                            [1, 0, 0, 10],
                            [0, 1, 0, 20],
                            [0, 0, 1, 30],
                            [0, 0, 0, 1],
                        ],
                        "rotation_quaternion_wxyz": [1, 0, 0, 0],
                        "translation_mm": [10, 20, 30],
                    },
                    "companion_transform": {
                        "from": "aruco_grid",
                        "to": "template_base",
                        "matrix": [
                            [1, 0, 0, 1],
                            [0, 1, 0, 2],
                            [0, 0, 1, 3],
                            [0, 0, 0, 1],
                        ],
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
                        "promotion_solver_provenance": {
                            "solver_policy": "auto_compare",
                            "pnp_method": "IPPE",
                            "extrinsic_method": "park",
                        },
                        "promoted_at": "2026-07-21T12:00:00+00:00",
                        "promoted_by": "operator",
                        "intrinsic_profile_id": "123_1280x720_normal_factory",
                    },
                },
            },
            {
                "id": "camera:missing",
                "type": "camera",
                "label": "Uncalibrated camera",
                "status": "unresolved",
                "transform": None,
                "unresolved_reason": "No valid calibration profile",
                "geometry": {"kind": "camera_frustum"},
                "provenance": {"source": "calibration_profiles"},
            },
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
                    "target_bounds": {
                        "x_mm": 0,
                        "y_mm": 0,
                        "width_mm": 90,
                        "height_mm": 40,
                    },
                    "markers": [
                        {
                            "id": 0,
                            "corners_mm": [
                                [0, 0, 0],
                                [40, 0, 0],
                                [40, 40, 0],
                                [0, 40, 0],
                            ],
                        }
                    ],
                    "pdf_url": "/ui/cell-calibration-target-pdf?run_root=test",
                },
                "provenance": {
                    "source": "processed/calibration/attempt/target_bundle/calibration_target.json",
                    "placement_known": False,
                },
            },
            *(
                []
                if objectless
                else [
                    {
                        "id": "pose_template_footprint",
                        "type": "template",
                        "label": "Exact object footprint",
                        "status": "planned",
                        "transform": identity,
                        "unresolved_reason": None,
                        "geometry": {
                            "kind": "pose_template_footprint",
                            "page": {"width_mm": 420, "height_mm": 297},
                            "page_configuration": {
                                "origin_from_lower_left_mm": [15, 15]
                            },
                            "contours": [
                                {
                                    "instance_uuid": "object-1",
                                    "contours": [
                                        [
                                            {"x_mm": 20, "y_mm": 20},
                                            {"x_mm": 50, "y_mm": 20},
                                            {"x_mm": 35, "y_mm": 50},
                                        ]
                                    ],
                                }
                            ],
                        },
                        "provenance": {"source": "pose_template_preview.json"},
                    }
                ]
            ),
        ],
        "warnings": [
            {
                "code": "missing_calibration_profiles",
                "message": "No calibration profile collection is available",
            }
        ],
        "timelines": [
            {
                "id": "sensor:realsense_123",
                "label": "realsense_123",
                "kind": "synchronized",
                "frame_count": 2,
                "default": True,
                "exact": True,
                "interpolation": "none",
                "page_limit": 2000,
                "source": "match_robot_ee_poses.json",
                "camera": {
                    "sensor_folder": "realsense_123",
                    "sensor_type": "realsense_d435",
                    "device_id": "123",
                    "display_name": "Wrist D435",
                    "mounting_mode": "eye_in_hand",
                    "inverted": True,
                    "image_presentation": {
                        "configured_inverted": True,
                        "stored_rotation_degrees": None,
                        "display_rotation_degrees": 180,
                        "correction": "viewer",
                    },
                },
                "camera_frames": {
                    "available": camera_frames_available,
                    "rgb": {
                        "available": camera_frames_available,
                        "kind": "rgb",
                        "media_type": "image/png",
                        "source": (
                            "/tmp/run/processed/synchronized/realsense_123/rgb"
                            if camera_frames_available
                            else None
                        ),
                    },
                    "depth": {
                        "available": camera_frames_available,
                        "kind": "depth",
                        "media_type": "image/png",
                        "source": (
                            "/tmp/run/processed/synchronized/realsense_123/depth"
                            if camera_frames_available
                            else None
                        ),
                        "depth_scale_to_mm": (1.0 if camera_frames_available else None),
                        "visualization": "turbo_near_warm_fixed_range",
                        "preview_min_depth_mm": 200.0,
                        "preview_max_depth_mm": 3000.0,
                        "invalid_depth_value": 0,
                    },
                },
            }
        ],
        "default_timeline_id": "sensor:realsense_123",
        "trajectory_preview": [
            {
                "index": 0,
                "frame_index": 0,
                "frame_id": "000000.png",
                "timestamp_ns": 1,
                "motion": "arc",
                "transform": identity,
            },
            {
                "index": 1,
                "frame_index": 1,
                "frame_id": "000001.png",
                "timestamp_ns": 2,
                "motion": "arc",
                "transform": {**identity, "translation_mm": [10, 20, 30]},
            },
        ],
        "object_selection": {
            "objectless": objectless,
            "dataset_mode": "objectless" if objectless else "pose_template",
            "instance_count": 0 if objectless else 1,
            "pose_template": None if objectless else {"template_uuid": "test-template"},
            "bop_export": {"status": "not_exported"},
        },
    }


def project_cell_world_point(
    canvas_box: dict[str, float],
    point: tuple[float, float, float],
) -> tuple[float, float]:
    camera_position = np.asarray([650.0, -700.0, 520.0])
    camera_target = np.asarray([0.0, 0.0, 80.0])
    camera_up = np.asarray([0.0, 0.0, 1.0])
    forward = camera_target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(point) - camera_position
    depth = float(relative @ forward)
    tangent = np.tan(np.deg2rad(42.0) / 2.0)
    aspect = canvas_box["width"] / canvas_box["height"]
    ndc_x = float(relative @ right) / (depth * tangent * aspect)
    ndc_y = float(relative @ up) / (depth * tangent)
    return (
        canvas_box["x"] + (ndc_x + 1.0) * canvas_box["width"] / 2.0,
        canvas_box["y"] + (1.0 - ndc_y) * canvas_box["height"] / 2.0,
    )


def test_cell_canvas_local_frames_flange_axis_and_camera_hit_target(
    console_server, page
) -> None:
    install_common_mocks(page)
    scene = cell_scene_payload(objectless=True)
    root = scene["entities"][0]
    root["geometry"] = {"kind": "none"}
    flange = scene["entities"][1]
    flange["transform"] = {
        **flange["transform"],
        "translation_mm": [-170, 0, 120],
    }
    camera = scene["entities"][2]
    camera["id"] = "camera:inspection"
    camera["label"] = "Inspection camera"
    camera["transform"] = {
        **camera["transform"],
        "parent_frame": "template_base",
        "translation_mm": [170, 0, 100],
    }
    camera.pop("calibration")
    scene["coordinate_system"] = {
        **scene["coordinate_system"],
        "up_axis": "+Z",
        "presentation": {
            "mode": "reference_z_up",
            "presentation_only": True,
            "source_frame": "template_base",
            "anchor_frame": "template_base",
            "display_up_axis": "+Z",
            "matrix": np.eye(4).tolist(),
            "transform": {
                "semantics": "entity_to_parent",
                "parent_frame": "display",
                "translation_mm": [0, 0, 0],
                "rotation_quaternion_wxyz": [1, 0, 0, 0],
            },
            "target_frame": None,
        },
    }
    scene["entities"] = [root, flange, camera]
    scene["warnings"] = []
    scene["timelines"] = []
    scene["default_timeline_id"] = None
    scene["trajectory_preview"] = []
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, scene))

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    canvas = page.get_by_test_id("cell-webgl-canvas")
    expect(canvas).to_be_visible()
    expect(canvas).to_have_attribute("data-reference-grid-clearance-mm", "12")
    expect(page.get_by_test_id("cell-axis-legend")).to_have_attribute(
        "aria-label", "Coordinate axes: X red, Y green, Z blue"
    )
    canvas_box = canvas.bounding_box()
    assert canvas_box is not None

    # This point is inside the camera housing behind the optical origin, where
    # the old line-only +Z frustum had no raycastable geometry.
    camera_hit = project_cell_world_point(canvas_box, (190.0, 0.0, 87.0))
    page.mouse.click(*camera_hit)
    expect(
        page.get_by_text("camera:inspection → template_base", exact=True)
    ).to_be_visible()

    screenshot = cv2.imdecode(
        np.frombuffer(canvas.screenshot(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert screenshot is not None
    blue, green, red = cv2.split(screenshot)
    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    red_axis = (red_i > 140) & (red_i > green_i * 1.45) & (red_i > blue_i * 1.45)
    green_axis = (green_i > 120) & (green_i > red_i * 1.45) & (green_i > blue_i * 1.45)
    blue_axis = (blue_i > 140) & (blue_i > red_i * 1.45) & (blue_i > green_i * 1.45)
    assert int(red_axis.sum()) > 10
    assert int(green_axis.sum()) > 10
    assert int(blue_axis.sum()) > 10

    image_box = {
        "x": 0.0,
        "y": 0.0,
        "width": float(screenshot.shape[1]),
        "height": float(screenshot.shape[0]),
    }
    flange_origin = np.asarray(
        project_cell_world_point(image_box, (-170.0, 0.0, 120.0))
    )
    flange_positive_z = np.asarray(
        project_cell_world_point(image_box, (-170.0, 0.0, 210.0))
    )
    screen_positive_z = flange_positive_z - flange_origin
    screen_positive_z /= np.linalg.norm(screen_positive_z)
    rows, columns = np.indices(green.shape)
    flange_roi = (columns - flange_origin[0]) ** 2 + (
        rows - flange_origin[1]
    ) ** 2 < 80**2
    flange_body = (
        flange_roi
        & (green_i > 65)
        & (green_i > red_i * 1.35)
        & (green_i > blue_i * 1.25)
    )
    flange_pixels = np.column_stack(np.nonzero(flange_body))[:, ::-1]
    assert len(flange_pixels) > 100
    flange_z_projection = (flange_pixels - flange_origin) @ screen_positive_z
    assert float(np.median(flange_z_projection)) < -3.0


def test_cell_canvas_print_surfaces_clear_reference_grid(console_server, page) -> None:
    install_common_mocks(page)
    target_scene = cell_scene_payload(objectless=True)
    root = target_scene["entities"][0]
    root["geometry"] = {"kind": "none"}
    target = next(
        entity
        for entity in target_scene["entities"]
        if entity["id"] == "calibration_target"
    )
    target["label"] = "Layered calibration target"
    target["status"] = "planned"
    target["geometry"] = {
        **target["geometry"],
        "target_bounds": {
            "x_mm": -160,
            "y_mm": -110,
            "width_mm": 320,
            "height_mm": 220,
        },
        "markers": [
            {
                "id": marker_id,
                "corners_mm": [
                    [x, y, 0],
                    [x + 60, y, 0],
                    [x + 60, y + 60, 0],
                    [x, y + 60, 0],
                ],
            }
            for marker_id, (x, y) in enumerate(
                [(-125, -75), (65, -75), (-125, 15), (65, 15)]
            )
        ],
    }
    target_scene["entities"] = [root, target]
    target_scene["warnings"] = []
    target_scene["timelines"] = []
    target_scene["default_timeline_id"] = None
    target_scene["trajectory_preview"] = []
    active_scene = {"value": target_scene}
    page.route(
        "**/ui/cell-scene?**",
        lambda route: fulfill_json(route, active_scene["value"]),
    )
    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    def canvas_image() -> np.ndarray:
        page.get_by_role("button", name="Top").click()
        canvas = page.get_by_test_id("cell-webgl-canvas")
        expect(canvas).to_be_visible()
        image = cv2.imdecode(
            np.frombuffer(canvas.screenshot(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert image is not None
        return image

    def white_surface_component(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        channel_min = image.min(axis=2)
        channel_max = image.max(axis=2)
        white = (channel_min > 145) & ((channel_max - channel_min) < 65)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            white.astype(np.uint8),
            connectivity=8,
        )
        assert component_count > 1
        component = max(
            range(1, component_count),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )
        left, top, width, height, area = stats[component]
        assert area > 10_000
        assert area / float(width * height) > 0.45
        return labels == component, stats[component]

    target_image = canvas_image()
    target_white, target_stats = white_surface_component(target_image)
    left, top, width, height, _area = target_stats
    target_interior = np.zeros(target_white.shape, dtype=bool)
    target_interior[top + 4 : top + height - 4, left + 4 : left + width - 4] = True
    dark = target_image.max(axis=2) < 55
    dark_components, _labels, dark_stats, _centroids = cv2.connectedComponentsWithStats(
        (dark & target_interior).astype(np.uint8),
        connectivity=8,
    )
    printed_markers = [
        index
        for index in range(1, dark_components)
        if dark_stats[index, cv2.CC_STAT_AREA] > 250
        and dark_stats[index, cv2.CC_STAT_WIDTH] > 10
        and dark_stats[index, cv2.CC_STAT_HEIGHT] > 10
    ]
    assert len(printed_markers) >= 4

    template_scene = cell_scene_payload(objectless=False)
    template_root = template_scene["entities"][0]
    template_root["geometry"] = {"kind": "none"}
    footprint = next(
        entity
        for entity in template_scene["entities"]
        if entity["id"] == "pose_template_footprint"
    )
    footprint["geometry"] = {
        **footprint["geometry"],
        "page": {"width_mm": 320, "height_mm": 220},
        "page_configuration": {"origin_from_lower_left_mm": [160, 110]},
        "contours": [
            {
                "instance_uuid": "object-1",
                "contours": [
                    [
                        {"x_mm": -85, "y_mm": -50},
                        {"x_mm": 85, "y_mm": -50},
                        {"x_mm": 85, "y_mm": 50},
                        {"x_mm": -85, "y_mm": 50},
                    ]
                ],
            }
        ],
    }
    template_scene["coordinate_system"] = {
        **template_scene["coordinate_system"],
        "up_axis": "+Z",
        "presentation": {
            "mode": "reference_z_up",
            "presentation_only": True,
            "source_frame": "template_base",
            "anchor_frame": "template_base",
            "display_up_axis": "+Z",
            "matrix": np.eye(4).tolist(),
            "transform": {
                "semantics": "entity_to_parent",
                "parent_frame": "display",
                "translation_mm": [0, 0, 0],
                "rotation_quaternion_wxyz": [1, 0, 0, 0],
            },
            "target_frame": None,
        },
    }
    template_scene["entities"] = [template_root, footprint]
    template_scene["warnings"] = []
    template_scene["timelines"] = []
    template_scene["default_timeline_id"] = None
    template_scene["trajectory_preview"] = []
    active_scene["value"] = template_scene
    page.reload(wait_until="networkidle")

    template_image = canvas_image()
    white_surface_component(template_image)
    blue, green, red = cv2.split(template_image.astype(np.int16))
    olive_contour = (
        (green > 145)
        & (red > 115)
        & (blue < 175)
        & (green > blue * 1.2)
        & (red > blue * 1.1)
    )
    assert int(olive_contour.sum()) > 500


def test_cell_canvas_layers_inspection_and_exact_seeking(console_server, page) -> None:
    install_common_mocks(page)
    workflow_sessions = json.dumps(
        {
            RUN_ROOT: {
                "journey": "dataset",
                "stepId": "capture",
                "status": "current",
            }
        }
    )
    page.add_init_script(
        f"localStorage.setItem('posetestbot.workflowSessions.v1', {json.dumps(workflow_sessions)})"
    )
    page.set_viewport_size({"width": 1920, "height": 1080})
    scene = cell_scene_payload(camera_frames_available=True)
    static_timeline = {
        **scene["timelines"][0],
        "id": "sensor:realsense_456",
        "label": "realsense_456",
        "default": False,
        "source": "/tmp/run/processed/synchronized/realsense_456/match_robot_ee_poses.json",
        "camera": {
            "sensor_folder": "realsense_456",
            "sensor_type": "realsense_d435",
            "device_id": "456",
            "display_name": "Static D435",
            "mounting_mode": "static",
            "inverted": False,
            "image_presentation": {
                "configured_inverted": False,
                "stored_rotation_degrees": 0,
                "display_rotation_degrees": 0,
                "correction": "not_required",
            },
        },
        "camera_frames": {
            "available": True,
            "rgb": {
                "available": True,
                "kind": "rgb",
                "media_type": "image/png",
                "source": "/tmp/run/processed/synchronized/realsense_456/rgb",
            },
            "depth": {
                "available": True,
                "kind": "depth",
                "media_type": "image/png",
                "source": "/tmp/run/processed/synchronized/realsense_456/depth",
                "depth_scale_to_mm": 1.0,
                "visualization": "turbo_near_warm_fixed_range",
                "preview_min_depth_mm": 200.0,
                "preview_max_depth_mm": 3000.0,
                "invalid_depth_value": 0,
            },
        },
    }
    scene["timelines"].append(static_timeline)
    static_poses = [
        {
            **pose,
            "frame_id": f"10000{index}.png",
            "transform": {
                **pose["transform"],
                "translation_mm": [100 + index, 20, 30],
            },
        }
        for index, pose in enumerate(scene["trajectory_preview"])
    ]
    calibration = scene["entities"][2]["calibration"]
    calibration["extrinsics"]["matrix"][0][3] = "10"
    calibration["extrinsics"]["rotation_quaternion_wxyz"] = ["1", "0", "0", "0"]
    calibration["extrinsics"]["translation_mm"] = ["10", "20", "30"]
    calibration["quality"]["mean_reprojection_error_px"] = "0.321"
    page.route("**/ui/cell-scene?**", lambda route: fulfill_json(route, scene))

    def timeline_handler(route) -> None:
        timeline_id = parse_qs(urlparse(route.request.url).query)["timeline_id"][0]
        selected_timeline = next(
            item for item in scene["timelines"] if item["id"] == timeline_id
        )
        fulfill_json(
            route,
            {
                "schema_version": "cell_timeline.v1",
                "timeline": selected_timeline,
                "offset": 0,
                "limit": 2000,
                "total": 2,
                "next_offset": None,
                "previous_offset": None,
                "poses": (
                    static_poses
                    if timeline_id == "sensor:realsense_456"
                    else scene["trajectory_preview"]
                ),
            },
        )

    page.route("**/ui/cell-scene/timeline?**", timeline_handler)
    encoded, camera_png = cv2.imencode(
        ".png", np.full((32, 64, 3), (30, 120, 220), dtype=np.uint8)
    )
    assert encoded
    page.route(
        "**/ui/cell-scene/camera-frame?**",
        lambda route: route.fulfill(
            status=200,
            content_type="image/png",
            body=camera_png.tobytes(),
        ),
    )

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_role("heading", name="Cell View")).to_be_visible()
    cell_handoff = page.get_by_role(
        "complementary", name="Where this page fits in the operator workflow"
    )
    expect(cell_handoff.get_by_role("link", name="Open workflow")).to_have_attribute(
        "href", "#/workflow/dataset?step=capture"
    )
    expect(page.get_by_test_id("cell-webgl-canvas")).to_be_visible()
    expect(page.get_by_test_id("cell-webgl-canvas")).to_have_attribute(
        "data-presentation-mode", "calibration_target_front"
    )
    expect(page.get_by_test_id("cell-webgl-canvas")).to_have_attribute(
        "data-presentation-quaternion", "0,1,0,0"
    )
    expect(page.get_by_test_id("cell-webgl-canvas")).to_have_attribute(
        "data-reference-grid-clearance-mm", "12"
    )
    expect(page.get_by_test_id("cell-axis-legend")).to_have_attribute(
        "aria-label", "Coordinate axes: X red, Y green, Z blue"
    )
    expect(page.get_by_test_id("cell-coordinate-convention")).to_contain_text(
        "origin top-left · +X right · +Y down · +Z into grid"
    )
    expect(page.get_by_text("Partial cell scene")).to_be_visible()
    expect(page.get_by_text("1 camera is hidden", exact=False)).to_be_visible()
    expect(page.get_by_text("Exact object footprint", exact=True)).to_be_visible()
    page.get_by_text("calib00 (reference placement)", exact=True).click()
    expect(
        page.get_by_text("Shown at the reference origin", exact=False)
    ).to_be_visible()
    expect(
        page.get_by_role("link", name="Open exact calibration-target PDF")
    ).to_be_visible()
    page.get_by_text("Wrist D435", exact=True).click()
    evidence = page.get_by_test_id("cell-calibration-evidence")
    expect(evidence.get_by_text("Calibration extrinsic", exact=True)).to_be_visible()
    expect(page.get_by_test_id("cell-calibration-transform-frames")).to_have_text(
        "camera → robot_flange"
    )
    expect(
        evidence.get_by_text(
            "1.0000000, 0.0000000, 0.0000000, 0.0000000",
            exact=True,
        ).first
    ).to_be_visible()
    expect(
        evidence.get_by_text("10.0000, 20.0000, 30.0000", exact=True)
    ).to_be_visible()
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
    expect(page.get_by_test_id("cell-calibration-companion-frames")).to_have_text(
        "aruco_grid → template_base"
    )
    expect(page.get_by_test_id("cell-calibration-companion-matrix")).to_contain_text(
        "3.000000"
    )
    expect(evidence.get_by_text("joint:IPPE:park", exact=True)).to_be_visible()
    page.get_by_text("Raw provenance", exact=True).click()
    raw_provenance = page.get_by_test_id("cell-raw-provenance")
    expect(raw_provenance).to_contain_text(
        '"calibration_dataset_id": "attempt-dataset"'
    )
    expect(raw_provenance).to_contain_text('"outlier_ratio": 0.1667')
    expect(raw_provenance).to_contain_text('"sync_delta_ms": 1.2')
    page.get_by_text("Robot flange", exact=True).click()
    expect(page.get_by_text("10.00, 20.00, 30.00")).not_to_be_visible()
    camera_section = page.get_by_test_id("cell-camera-frames")
    expect(camera_section).to_contain_text("2 cameras retain image data")
    page.get_by_role("button", name="Show frames").click()
    expect(page.get_by_role("checkbox", name="Show Wrist D435")).to_be_checked()
    expect(page.get_by_role("checkbox", name="Show Static D435")).not_to_be_checked()
    page.get_by_role("checkbox", name="Show Static D435").check()

    camera_columns = page.get_by_test_id("cell-camera-column")
    expect(camera_columns).to_have_count(2)
    wrist_column = camera_columns.filter(has_text="Wrist D435")
    static_column = camera_columns.filter(has_text="Static D435")
    wrist_image = wrist_column.get_by_test_id("cell-camera-frame-image")
    static_image = static_column.get_by_test_id("cell-camera-frame-image")
    expect(wrist_image).to_have_attribute("alt", "Wrist D435 RGB frame 000000.png")
    expect(static_image).to_have_attribute("alt", "Static D435 RGB frame 100000.png")
    assert "frame_id=000000.png" in (wrist_image.get_attribute("src") or "")
    assert "frame_id=100000.png" in (static_image.get_attribute("src") or "")
    expect(wrist_image).to_have_attribute("data-display-rotation-degrees", "180")
    expect(wrist_image).to_have_attribute("style", "transform: rotate(180deg);")
    expect(wrist_column.get_by_test_id("cell-camera-orientation")).to_contain_text(
        "rotated 180° for display in Cell"
    )
    expect(static_image).to_have_attribute("data-display-rotation-degrees", "0")
    expect(static_image).not_to_have_attribute("style", re.compile("rotate"))
    canvas_box = page.get_by_test_id("cell-webgl-canvas").bounding_box()
    camera_section_box = camera_section.bounding_box()
    wrist_box = wrist_column.bounding_box()
    static_box = static_column.bounding_box()
    assert (
        canvas_box is not None
        and camera_section_box is not None
        and wrist_box is not None
        and static_box is not None
    )
    assert camera_section_box["y"] >= canvas_box["y"] + canvas_box["height"]
    assert abs(wrist_box["y"] - static_box["y"]) < 2
    assert wrist_box["x"] + wrist_box["width"] <= static_box["x"]
    wrist_image_box = wrist_image.bounding_box()
    static_image_box = static_image.bounding_box()
    assert wrist_image_box is not None and static_image_box is not None
    assert abs(wrist_image_box["y"] - static_image_box["y"]) < 2
    expect(
        page.get_by_text(
            "Side-by-side display does not claim simultaneous exposure",
            exact=False,
        )
    ).to_be_visible()
    page.get_by_role("slider", name="Frame scrubber").fill("1")
    expect(page.get_by_text("Exact 3D pose frame 000001.png · arc")).to_be_visible()
    expect(wrist_image).to_have_attribute("alt", "Wrist D435 RGB frame 000001.png")
    expect(static_image).to_have_attribute("alt", "Static D435 RGB frame 100001.png")
    assert "frame_id=000001.png" in (wrist_image.get_attribute("src") or "")
    assert "timeline_id=sensor%3Arealsense_456" in (
        static_image.get_attribute("src") or ""
    )
    expect(static_column).to_contain_text("realsense_456 · Realsense D435 · Static")

    page.get_by_role("button", name="Show Depth").click()
    depth_images = page.locator(
        '[data-testid="cell-camera-frame-image"][data-modality="depth"]'
    )
    expect(depth_images).to_have_count(2)
    expect(page.get_by_test_id("cell-depth-legend")).to_have_count(2)
    expect(depth_images.first).to_have_attribute(
        "alt", "Wrist D435 Depth frame 000001.png"
    )
    assert "modality=depth" in (depth_images.first.get_attribute("src") or "")
    expect(depth_images.first).to_have_attribute("data-display-rotation-degrees", "180")

    page.get_by_role("button", name="Show RGB + depth").click()
    expect(page.get_by_test_id("cell-camera-frame-image")).to_have_count(4)
    expect(wrist_column.get_by_test_id("cell-camera-frame-panel")).to_have_count(2)
    expect(static_column.get_by_test_id("cell-camera-frame-panel")).to_have_count(2)
    expect(page.get_by_role("slider", name="Frame scrubber")).to_have_value("1")

    page.get_by_role("button", name="Hide frames").click()
    expect(camera_columns).not_to_be_visible()
    page.get_by_text("Recorded trajectory", exact=True).click()
    expect(page.get_by_role("checkbox", name="Recorded trajectory")).not_to_be_checked()


def test_cell_webgl_fallback_and_objectless_state(console_server, page) -> None:
    install_common_mocks(page)
    page.add_init_script("HTMLCanvasElement.prototype.getContext = () => null")
    page.route(
        "**/ui/cell-scene?**",
        lambda route: fulfill_json(route, cell_scene_payload(objectless=True)),
    )
    page.route(
        "**/ui/cell-scene/timeline?**",
        lambda route: fulfill_json(
            route,
            {
                "schema_version": "cell_timeline.v1",
                "timeline": cell_scene_payload()["timelines"][0],
                "offset": 0,
                "limit": 2000,
                "total": 0,
                "next_offset": None,
                "previous_offset": None,
                "poses": [],
            },
        ),
    )

    page.goto(f"{console_server.url}/#/cell", wait_until="networkidle")

    expect(page.get_by_test_id("cell-webgl-fallback")).to_be_visible()
    expect(page.get_by_text("WebGL is unavailable")).to_be_visible()
    expect(page.get_by_text("Objectless RGB-D run")).to_be_visible()
    expect(page.get_by_text("Robot flange", exact=True)).to_be_visible()
    expect(page.get_by_test_id("cell-camera-frames")).to_have_count(0)
