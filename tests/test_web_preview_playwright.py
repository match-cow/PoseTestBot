from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from werkzeug.serving import make_server

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from posetestbot.web import legacy as web_legacy
from posetestbot.web.app import create_app
from posetestbot.web.routes import monitoring as web_monitoring
from posetestbot.web.routes import sensors as web_sensors


SENSOR_A = "realsense_d435:825412070181"
SENSOR_B = "realsense_d435:923322072633"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    image[:, :, 2] = 40
    assert cv2.imwrite(path.as_posix(), image)


def wait_for(condition, *, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.03)
    assert condition()


def fake_sensor_status(expected_counts=None) -> dict:
    return {
        "schema_version": "sensor_status.v1",
        "families": [
            {
                "sensor_type": "realsense_d435",
                "display_name": "Intel RealSense D435",
                "devices": [
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "825412070181",
                        "display_name": "RealSense 1",
                        "effective_display_name": "Wrist RealSense",
                        "connected": True,
                        "inverted": False,
                        "metadata": {
                            "video_nodes": [{"path": "/dev/video4", "accessible": True}],
                            "video_accessible": True,
                        },
                    },
                    {
                        "sensor_type": "realsense_d435",
                        "device_id": "923322072633",
                        "display_name": "RealSense 2",
                        "effective_display_name": "Overhead RealSense",
                        "connected": True,
                        "inverted": False,
                        "metadata": {
                            "video_nodes": [{"path": "/dev/video8", "accessible": True}],
                            "video_accessible": True,
                        },
                    },
                ],
            }
        ],
        "total_connected": 2,
        "all_expected_connected": True,
        "expected_counts_requested": False,
    }


class FakePreviewJob:
    def __init__(self, job_id: str, *, name: str, parameters: dict, resources: list[str]):
        self.id = job_id
        self.name = name
        self.status = "queued"
        self.message = None
        self.parameters = parameters
        self.resources = resources
        self.tail = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "parameters": self.parameters,
            "resources": self.resources,
            "tail": self.tail,
        }


class FakePreviewRunner:
    def __init__(self):
        self.jobs: dict[str, FakePreviewJob] = {}
        self.submitted: list[dict] = []
        self.canceled: list[str] = []

    def submit(self, **kwargs):
        self.submitted.append(kwargs)
        job_id = f"preview-{len(self.submitted)}"
        job = FakePreviewJob(
            job_id,
            name=kwargs["name"],
            parameters=dict(kwargs["parameters"]),
            resources=list(kwargs["resources"]),
        )
        self.jobs[job_id] = job
        return job

    def list(self):
        return list(self.jobs.values())

    def get(self, job_id: str):
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown job: {job_id}") from exc

    def cancel(self, job_id: str):
        job = self.get(job_id)
        job.status = "canceled"
        job.message = "Cancellation requested."
        self.canceled.append(job_id)
        return job


class EmptyRunner:
    def list(self):
        return []


class PreviewRootFactory:
    def __init__(self, root: Path):
        self.root = root
        self.count = 0

    def __call__(self) -> Path:
        self.count += 1
        return self.root / f"preview-{self.count}"


class LiveServer:
    def __init__(self, app):
        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)


@pytest.fixture
def preview_server(monkeypatch, tmp_path: Path):
    runner = FakePreviewRunner()
    monitor_runner = FakePreviewRunner()
    runner.monitor_runner = monitor_runner
    monkeypatch.setattr(web_sensors, "job_runner", runner)
    monkeypatch.setattr(web_sensors, "collect_sensor_status", fake_sensor_status)
    monkeypatch.setattr(web_sensors, "preview_stream_root", PreviewRootFactory(tmp_path))
    monkeypatch.setattr(web_sensors, "DEFAULT_SENSOR_ALIASES_PATH", tmp_path / "aliases.json")
    monkeypatch.setattr(web_monitoring, "job_runner", monitor_runner)
    monkeypatch.setattr(
        web_monitoring,
        "preview_stream_root",
        PreviewRootFactory(tmp_path / "monitor"),
    )
    monkeypatch.setattr(web_legacy, "job_runner", EmptyRunner())
    app = create_app()
    app.config.update(TESTING=True)
    server = LiveServer(app)
    server.start()
    try:
        yield server, runner
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
                "`UV_CACHE_DIR=/tmp/uv-cache uv run playwright install chromium` "
                f"and rerun this test. Original error: {exc}"
            )
        page = browser.new_page()
        try:
            yield page
        finally:
            browser.close()


def sensor_card(page, sensor_key: str):
    return page.locator(f'[data-testid="sensor-card"][data-sensor-key="{sensor_key}"]')


def test_sidebar_webcam_monitor_displays_latest_frame(preview_server, page) -> None:
    server, runner = preview_server
    page.goto(server.url, wait_until="domcontentloaded")

    monitor_runner = runner.monitor_runner
    wait_for(lambda: len(monitor_runner.jobs) == 1)
    job = next(iter(monitor_runner.jobs.values()))
    preview_root = Path(job.parameters["preview_root"])
    write_jpeg(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "frame_count": 3,
            "latest_image": "latest.jpg",
            "selected_node": {"path": "/dev/video18"},
            "error": None,
        },
    )
    job.status = "running"

    expect(page.locator("#webcamMonitorImage")).to_be_visible(timeout=4000)
    expect(page.locator("#webcamMonitorStatus")).to_have_text("running")
    expect(page.locator("#retryWebcamBtn")).to_be_hidden()


def test_card_local_preview_stream_lifecycle(preview_server, page) -> None:
    server, runner = preview_server
    page.goto(server.url, wait_until="domcontentloaded")

    expect(page.locator('[data-testid="sensor-card"]')).to_have_count(2)
    first = sensor_card(page, SENSOR_A)
    second = sensor_card(page, SENSOR_B)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    toggle.check()

    expect(toggle).to_be_checked()
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_visible()
    expect(first.locator(".sensor-preview-empty")).to_contain_text("Waiting")
    wait_for(lambda: len(runner.submitted) == 1)
    job = runner.jobs["preview-1"]
    assert runner.submitted[0]["parameters"]["sensor_key"] == SENSOR_A

    preview_root = Path(job.parameters["preview_root"])
    write_jpeg(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "sensor_key": SENSOR_A,
            "effective_display_name": "Wrist RealSense",
            "frame_count": 3,
            "latest_image": "latest.jpg",
            "selected_node": {"path": "/dev/video4"},
            "inverted": False,
            "error": None,
        },
    )
    job.status = "running"

    expect(first.locator('[data-testid="sensor-preview-image"]')).to_be_visible(timeout=4000)
    expect(first.locator('[data-testid="sensor-preview-meta"]')).to_contain_text("/dev/video4")
    expect(second.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)
    expect(page.locator("#previewPanel img")).to_have_count(0)

    toggle.uncheck()

    wait_for(lambda: runner.canceled == ["preview-1"])
    expect(toggle).not_to_be_checked()
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_hidden()
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)


def test_terminal_failed_preview_unchecks_switch_and_keeps_inline_error(
    preview_server,
    page,
) -> None:
    server, runner = preview_server
    page.goto(server.url, wait_until="domcontentloaded")
    first = sensor_card(page, SENSOR_A)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    toggle.check()
    wait_for(lambda: len(runner.submitted) == 1)
    job = runner.jobs["preview-1"]
    preview_root = Path(job.parameters["preview_root"])
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "failed",
            "sensor_key": SENSOR_A,
            "frame_count": 0,
            "latest_image": None,
            "selected_node": {"path": "/dev/video4"},
            "inverted": False,
            "error": "RuntimeError: camera missing",
        },
    )
    job.status = "failed"
    job.message = "Command exited with status 2."

    expect(toggle).not_to_be_checked(timeout=4000)
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_visible()
    expect(first.locator('[data-testid="sensor-preview-error"]')).to_contain_text(
        "camera missing"
    )


def test_inverted_change_restarts_preview_in_waiting_state(preview_server, page) -> None:
    server, runner = preview_server
    page.goto(server.url, wait_until="domcontentloaded")
    first = sensor_card(page, SENSOR_A)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    toggle.check()
    wait_for(lambda: len(runner.submitted) == 1)
    first_job = runner.jobs["preview-1"]
    preview_root = Path(first_job.parameters["preview_root"])
    write_jpeg(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "sensor_key": SENSOR_A,
            "frame_count": 1,
            "latest_image": "latest.jpg",
            "selected_node": {"path": "/dev/video4"},
            "inverted": False,
            "error": None,
        },
    )
    first_job.status = "running"
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_be_visible(timeout=4000)

    first.locator(".inverted-input").check()

    wait_for(lambda: len(runner.submitted) == 2)
    assert runner.canceled == ["preview-1"]
    command = runner.submitted[1]["command"]
    sensor_spec = json.loads(command[command.index("--sensor-json") + 1])
    assert sensor_spec["inverted"] is True
    expect(toggle).to_be_checked()
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_visible()
    expect(first.locator(".sensor-preview-empty")).to_contain_text("Waiting")
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)
