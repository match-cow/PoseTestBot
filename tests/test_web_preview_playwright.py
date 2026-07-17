from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from werkzeug.serving import make_server

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from posetestbot.web import legacy as web_legacy
from posetestbot.web.app import create_app
from posetestbot.web.routes import monitoring as web_monitoring
from posetestbot.web.routes import sensors as web_sensors
from posetestbot.monitoring.webrtc import (
    MonitorStatusWriter,
    MonitorWebRTCServer,
    bgr_frame_to_av,
    load_monitor_status,
)


SENSOR_A = "realsense_d435:825412070181"
SENSOR_B = "realsense_d435:923322072633"
SENSOR_C = "realsense_d435:944122070001"
OAK_SENSOR = "oak_d_pro:18443010D1A2B30D00"
ZED_SENSOR = "zed_2i:zed-001"


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


def fake_full_lab_sensor_status(expected_counts=None) -> dict:
    status = fake_sensor_status(expected_counts)
    status["families"][0]["devices"].append(
        {
            "sensor_type": "realsense_d435",
            "device_id": SENSOR_C.split(":", 1)[1],
            "display_name": "RealSense 3",
            "effective_display_name": "Side RealSense",
            "connected": True,
            "inverted": False,
            "metadata": {
                "video_nodes": [{"path": "/dev/video12", "accessible": True}],
                "video_accessible": True,
            },
        }
    )
    status["families"].extend(
        [
            {
                "sensor_type": "oak_d_pro",
                "display_name": "Luxonis OAK-D Pro",
                "devices": [
                    {
                        "sensor_type": "oak_d_pro",
                        "device_id": OAK_SENSOR.split(":", 1)[1],
                        "display_name": "OAK-D Pro",
                        "effective_display_name": "OAK-D Pro",
                        "connected": True,
                        "inverted": False,
                        "metadata": {},
                    }
                ],
            },
            {
                "sensor_type": "zed_2i",
                "display_name": "Stereolabs ZED 2i",
                "devices": [
                    {
                        "sensor_type": "zed_2i",
                        "device_id": ZED_SENSOR.split(":", 1)[1],
                        "display_name": "ZED 2i",
                        "effective_display_name": "ZED 2i",
                        "connected": True,
                        "inverted": False,
                        "metadata": {},
                    }
                ],
            },
        ]
    )
    status["total_connected"] = 5
    return status


def fake_full_lab_sensor_status_with_claimed_oak(expected_counts=None) -> dict:
    """Model DepthAI discovery while the preview process owns the OAK device."""

    status = fake_full_lab_sensor_status(expected_counts)
    oak_family = next(
        family
        for family in status["families"]
        if family["sensor_type"] == "oak_d_pro"
    )
    oak_family["devices"] = []
    oak_family["connected_count"] = 0
    status["total_connected"] = 4
    return status


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


class SyntheticVideoTrack(VideoStreamTrack):
    def __init__(self, on_frame) -> None:
        super().__init__()
        self.frame_count = 0
        self.on_frame = on_frame
        self.image = np.random.default_rng(42).integers(
            0,
            256,
            size=(480, 640, 3),
            dtype=np.uint8,
        )
        self.recv_idle = asyncio.Event()
        self.recv_idle.set()

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        self.recv_idle.clear()
        try:
            await asyncio.sleep(1 / 30)
            if self.readyState != "live":
                raise MediaStreamError
            # A textured frame produces the same multi-packet VP8 keyframe as
            # the real room camera. Flat-color fixtures hide path-MTU bugs.
            image = np.roll(self.image, self.frame_count % 32, axis=1)
            frame = bgr_frame_to_av(image, frame_index=self.frame_count, fps=30)
            self.frame_count += 1
            self.on_frame(self.frame_count)
            return frame
        finally:
            self.recv_idle.set()

    async def wait_stopped(self) -> None:
        await self.recv_idle.wait()


class StalledVideoTrack(VideoStreamTrack):
    """Connect transport without ever yielding a decodable video frame."""

    frame_count = 0

    def __init__(self) -> None:
        super().__init__()
        self.recv_idle = asyncio.Event()
        self.recv_idle.set()

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        self.recv_idle.clear()
        try:
            while self.readyState == "live":
                await asyncio.sleep(0.05)
            raise MediaStreamError
        finally:
            self.recv_idle.set()

    async def wait_stopped(self) -> None:
        await self.recv_idle.wait()


class SyntheticMonitorServer:
    def __init__(self, root: Path, *, emit_frames: bool = True) -> None:
        self.root = root
        self.emit_frames = emit_frames
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.error: BaseException | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)

        async def serve() -> None:
            status = MonitorStatusWriter(self.root)

            def on_frame(frame_count: int) -> None:
                if frame_count == 1 or frame_count % 5 == 0:
                    status.update(frame_count=frame_count)

            def on_peers(peer_count: int, connected_count: int) -> None:
                status.update(
                    status="connected" if connected_count else "ready",
                    peer_count=peer_count,
                )

            track = (
                SyntheticVideoTrack(on_frame)
                if self.emit_frames
                else StalledVideoTrack()
            )
            server = MonitorWebRTCServer(track, on_peers_changed=on_peers)
            self.stop_event = asyncio.Event()
            try:
                port = await server.start()
                status.update(
                    status="ready",
                    signaling_ready=True,
                    signaling_port=port,
                    selected_node={"path": "synthetic://color-bars"},
                )
                self.started.set()
                await self.stop_event.wait()
            finally:
                await server.stop()
                track.stop()
                status.update(
                    status="stopped",
                    signaling_ready=False,
                    signaling_port=None,
                    peer_count=0,
                    frame_count=track.frame_count,
                )

        try:
            loop.run_until_complete(serve())
        except BaseException as exc:
            self.error = exc
            self.started.set()
        finally:
            loop.close()

    def start(self) -> None:
        self.thread.start()
        assert self.started.wait(timeout=5)
        if self.error is not None:
            raise self.error

    def stop(self) -> None:
        if self.loop is not None and self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        if self.error is not None:
            raise self.error


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
        "monitor_stream_root",
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


def test_sidebar_webcam_monitor_plays_synthetic_webrtc_without_jpegs(
    preview_server,
    page,
    tmp_path: Path,
) -> None:
    server, runner = preview_server
    monitor_runner = runner.monitor_runner
    monitor_root = tmp_path / "synthetic-monitor"
    synthetic = SyntheticMonitorServer(monitor_root)
    synthetic.start()
    job = FakePreviewJob(
        "synthetic-monitor",
        name="monitor-webrtc:synthetic",
        parameters={
            "monitor_webcam": True,
            "monitor_webrtc": True,
            "monitor_root": monitor_root.as_posix(),
        },
        resources=["monitoring_camera:0c45:2283"],
    )
    job.status = "running"
    monitor_runner.jobs[job.id] = job
    jpeg_requests: list[str] = []
    page.on(
        "request",
        lambda request: jpeg_requests.append(request.url)
        if "/monitoring/webcam/" in request.url and ".jpg" in request.url
        else None,
    )

    try:
        page.goto(server.url, wait_until="domcontentloaded")
        expect(page.get_by_role("heading", name="Test cell monitor")).to_be_visible()
        expect(
            page.get_by_text("UGREEN safety overview · WebRTC video", exact=True)
        ).to_have_count(0)
        video = page.locator('[data-testid="room-monitor-video"]')
        expect(video).to_have_attribute("data-connection-state", "connected", timeout=15_000)
        expect(video).to_be_visible()
        wait_for(
            lambda: bool(
                (status := load_monitor_status(monitor_root))
                and status["frame_count"] >= 5
                and status["peer_count"] == 1
            ),
            timeout_s=10,
        )
        assert video.evaluate("element => element.readyState >= 2 && element.videoWidth === 640")
        assert jpeg_requests == []

        page.goto("about:blank")
        wait_for(
            lambda: bool(
                (status := load_monitor_status(monitor_root))
                and status["peer_count"] == 0
            ),
            timeout_s=5,
        )
    finally:
        synthetic.stop()


def test_sidebar_webcam_does_not_call_transport_connected_usable_video(
    preview_server,
    page,
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, runner = preview_server
    monitor_runner = runner.monitor_runner
    monitor_root = tmp_path / "stalled-monitor"
    synthetic = SyntheticMonitorServer(monitor_root, emit_frames=False)
    synthetic.start()
    job = FakePreviewJob(
        "stalled-monitor",
        name="monitor-webrtc:stalled",
        parameters={
            "monitor_webcam": True,
            "monitor_webrtc": True,
            "monitor_root": monitor_root.as_posix(),
        },
        resources=["monitoring_camera:0c45:2283"],
    )
    job.status = "running"
    monitor_runner.jobs[job.id] = job
    monkeypatch.setattr(
        web_monitoring,
        "_monitor_health",
        lambda _job, _status: (True, None),
    )

    try:
        page.goto(server.url, wait_until="domcontentloaded")
        video = page.locator('[data-testid="room-monitor-video"]')
        message = page.locator('[data-testid="room-monitor-message"]')

        expect(video).to_have_attribute(
            "data-connection-state",
            "receiving",
            timeout=10_000,
        )
        expect(message).to_contain_text("waiting for the first camera frame")
        expect(message).to_contain_text(
            "did not render a camera frame",
            timeout=7_000,
        )
        assert video.get_attribute("data-connection-state") != "connected"
        assert load_monitor_status(monitor_root)["frame_count"] == 0
    finally:
        page.goto("about:blank")
        synthetic.stop()


def test_sidebar_webcam_monitor_restarts_one_stale_failed_job(
    preview_server,
    page,
    tmp_path: Path,
) -> None:
    server, runner = preview_server
    monitor_runner = runner.monitor_runner
    stale_job = FakePreviewJob(
        "stale-monitor",
        name="monitor-webrtc:ugreen",
        parameters={
            "monitor_webcam": True,
            "monitor_webrtc": True,
            "monitor_root": (tmp_path / "stale-monitor").as_posix(),
        },
        resources=["monitoring_camera:0c45:2283"],
    )
    stale_job.status = "failed"
    stale_job.message = "Could not open RGB preview node /dev/video18."
    monitor_runner.jobs[stale_job.id] = stale_job

    page.goto(server.url, wait_until="domcontentloaded")

    wait_for(lambda: len(monitor_runner.submitted) == 1)
    replacement = monitor_runner.jobs["preview-1"]
    replacement.status = "failed"
    time.sleep(1.2)
    assert len(monitor_runner.submitted) == 1


def test_sidebar_webcam_webrtc_retry_is_bounded_and_manual_retry_reuses_worker(
    preview_server,
    page,
    monkeypatch,
    tmp_path: Path,
) -> None:
    server, runner = preview_server
    monitor_runner = runner.monitor_runner
    monitor_root = tmp_path / "failed-signaling-monitor"
    status = MonitorStatusWriter(monitor_root)
    status.update(
        status="ready",
        signaling_ready=True,
        signaling_port=39876,
        selected_node={"path": "synthetic://unavailable"},
    )
    job = FakePreviewJob(
        "active-monitor",
        name="monitor-webrtc:synthetic",
        parameters={
            "monitor_webcam": True,
            "monitor_webrtc": True,
            "monitor_root": monitor_root.as_posix(),
        },
        resources=["monitoring_camera:0c45:2283"],
    )
    job.status = "running"
    monitor_runner.jobs[job.id] = job
    offer_count = 0

    def fail_offer(_port, _payload):
        nonlocal offer_count
        offer_count += 1
        raise RuntimeError("synthetic signaling failure")

    monkeypatch.setattr(web_monitoring, "_proxy_webrtc_offer", fail_offer)
    monkeypatch.setattr(
        web_monitoring,
        "_monitor_health",
        lambda _job, _status: (True, None),
    )

    page.goto(server.url, wait_until="domcontentloaded")
    wait_for(lambda: offer_count == 4, timeout_s=20)
    page.wait_for_timeout(1500)
    assert offer_count == 4
    expect(page.locator('[data-testid="room-monitor-video"]')).to_have_attribute(
        "data-connection-state",
        "failed",
    )

    page.get_by_role("button", name="Retry").click()
    wait_for(lambda: offer_count == 5, timeout_s=5)
    page.wait_for_timeout(1500)
    assert offer_count == 6
    assert monitor_runner.submitted == []


def test_card_local_preview_stream_lifecycle(preview_server, page) -> None:
    server, runner = preview_server
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")

    expect(page.locator('[data-testid="sensor-card"]')).to_have_count(2)
    first = sensor_card(page, SENSOR_A)
    second = sensor_card(page, SENSOR_B)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    expect(toggle).to_have_attribute("aria-pressed", "false")
    expect(toggle).to_contain_text("Preview off")

    toggle.click()

    expect(toggle).to_have_attribute("aria-pressed", "true")
    expect(toggle).to_contain_text("Preview on")
    expect(first.get_by_role("button", name="Snapshot")).to_be_disabled()
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
    toggle.click()

    wait_for(lambda: runner.canceled == ["preview-1"])
    expect(toggle).to_have_attribute("aria-pressed", "false")
    expect(toggle).to_contain_text("Preview off")
    expect(first.get_by_role("button", name="Snapshot")).to_be_enabled()
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_hidden()
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)


def test_running_preview_wins_over_stale_job_for_same_sensor(
    preview_server,
    page,
    tmp_path: Path,
) -> None:
    server, runner = preview_server
    active_root = tmp_path / "active-preview"
    write_jpeg(active_root / "latest.jpg")
    write_json(
        active_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "sensor_key": SENSOR_A,
            "frame_count": 7,
            "latest_image": "latest.jpg",
            "selected_node": {"path": "/dev/video4"},
            "inverted": False,
            "error": None,
        },
    )
    active = FakePreviewJob(
        "active-preview",
        name=f"sensor-preview:{SENSOR_A}",
        parameters={
            "preview_root": active_root.as_posix(),
            "sensor_key": SENSOR_A,
            "sensor_type": "realsense_d435",
            "device_id": "825412070181",
            "sensor_preview": True,
        },
        resources=[f"camera:{SENSOR_A}"],
    )
    active.status = "running"
    runner.jobs[active.id] = active

    stale = FakePreviewJob(
        "stale-preview",
        name=f"sensor-preview:{SENSOR_A}",
        parameters={
            "preview_root": (tmp_path / "stale-preview").as_posix(),
            "sensor_key": SENSOR_A,
            "sensor_type": "realsense_d435",
            "device_id": "825412070181",
            "sensor_preview": True,
        },
        resources=[f"camera:{SENSOR_A}"],
    )
    stale.status = "failed"
    stale.message = "Historical preview failure."
    runner.jobs[stale.id] = stale

    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")

    first = sensor_card(page, SENSOR_A)
    expect(first.locator('[data-testid="sensor-preview-toggle"]')).to_have_attribute(
        "aria-pressed",
        "true",
    )
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_be_visible(
        timeout=4_000
    )
    expect(first.locator('[data-testid="sensor-preview-meta"]')).to_contain_text(
        "/dev/video4"
    )


def test_terminal_failed_preview_unchecks_switch_and_keeps_inline_error(
    preview_server,
    page,
) -> None:
    server, runner = preview_server
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")
    first = sensor_card(page, SENSOR_A)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    toggle.click()
    wait_for(lambda: len(runner.submitted) == 1)
    job = runner.jobs["preview-1"]
    preview_root = Path(job.parameters["preview_root"])
    write_jpeg(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "failed",
            "sensor_key": SENSOR_A,
            "frame_count": 4,
            "latest_image": "latest.jpg",
            "selected_node": {"path": "/dev/video4"},
            "inverted": False,
            "error": "RuntimeError: camera missing",
        },
    )
    job.status = "failed"
    job.message = "Command exited with status 2."

    expect(toggle).to_have_attribute("aria-pressed", "false", timeout=4000)
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_visible()
    expect(first.locator('[data-testid="sensor-preview-error"]')).to_contain_text(
        "camera missing"
    )
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)


def test_three_live_realsense_previews_keep_lower_lab_sensors_reachable(
    preview_server,
    page,
    monkeypatch,
) -> None:
    server, runner = preview_server
    monkeypatch.setattr(web_sensors, "collect_sensor_status", fake_full_lab_sensor_status)
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")

    expect(page.locator('[data-testid="sensor-card"]')).to_have_count(5)
    for sensor_key in (SENSOR_A, SENSOR_B, SENSOR_C):
        card = sensor_card(page, sensor_key)
        card.locator('[data-testid="sensor-preview-toggle"]').click()
        expect(card.locator('[data-testid="sensor-preview-toggle"]')).to_have_attribute(
            "aria-pressed", "true"
        )

    wait_for(lambda: len(runner.submitted) == 3)
    for job in runner.jobs.values():
        preview_root = Path(job.parameters["preview_root"])
        write_jpeg(preview_root / "latest.jpg")
        write_json(
            preview_root / "preview_status.json",
            {
                "schema_version": "sensor_rgb_preview.v1",
                "status": "running",
                "sensor_key": job.parameters["sensor_key"],
                "frame_count": 1,
                "latest_image": "latest.jpg",
                "selected_node": {"path": "/dev/video-test"},
                "inverted": False,
                "error": None,
            },
        )
        job.status = "running"

    expect(page.locator('[data-testid="sensor-preview-image"]')).to_have_count(3, timeout=4_000)
    oak_card = sensor_card(page, OAK_SENSOR)
    oak_card.scroll_into_view_if_needed()
    expect(oak_card).to_be_visible()
    use_in_run = oak_card.get_by_text("Use in run").locator('[role="checkbox"]')
    use_in_run.click()
    expect(use_in_run).to_be_checked()
    zed_card = sensor_card(page, ZED_SENSOR)
    zed_card.scroll_into_view_if_needed()
    expect(zed_card).to_be_visible()


def test_oak_preview_toggle_keeps_full_devices_page_reachable(
    preview_server,
    page,
    monkeypatch,
) -> None:
    server, runner = preview_server
    monkeypatch.setattr(web_sensors, "collect_sensor_status", fake_full_lab_sensor_status)
    page.set_viewport_size({"width": 1280, "height": 720})
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")

    cards = page.locator('[data-testid="sensor-card"]')
    grid = page.locator('[data-testid="sensor-grid"]')
    oak_card = sensor_card(page, OAK_SENSOR)
    zed_card = sensor_card(page, ZED_SENSOR)
    toggle = oak_card.locator('[data-testid="sensor-preview-toggle"]')

    expect(cards).to_have_count(5)
    oak_card.scroll_into_view_if_needed()
    toggle.click()
    wait_for(lambda: len(runner.submitted) == 1)
    job = runner.jobs["preview-1"]
    assert job.parameters["sensor_key"] == OAK_SENSOR

    preview_root = Path(job.parameters["preview_root"])
    write_jpeg(preview_root / "latest.jpg")
    write_json(
        preview_root / "preview_status.json",
        {
            "schema_version": "sensor_rgb_preview.v1",
            "status": "running",
            "sensor_key": OAK_SENSOR,
            "frame_count": 2,
            "latest_image": "latest.jpg",
            "selected_node": {
                "kind": "depthai",
                "device_id": OAK_SENSOR.split(":", 1)[1],
                "queue_blocking": False,
                "queue_max_size": 1,
            },
            "inverted": False,
            "error": None,
        },
    )
    job.status = "running"

    expect(oak_card.locator('[data-testid="sensor-preview-image"]')).to_be_visible(
        timeout=4_000
    )
    expect(oak_card.locator('[data-testid="sensor-preview-meta"]')).to_contain_text(
        OAK_SENSOR.split(":", 1)[1]
    )
    expect(cards).to_have_count(5)
    expect(grid).to_be_visible()
    zed_card.scroll_into_view_if_needed()
    expect(zed_card).to_be_visible()

    # DepthAI no longer includes an OAK device in discovery while another
    # process owns it. Reload through that transition and require the active
    # preview specification to keep the card and frame in the Devices UI.
    monkeypatch.setattr(
        web_sensors,
        "collect_sensor_status",
        fake_full_lab_sensor_status_with_claimed_oak,
    )
    page.reload(wait_until="domcontentloaded")
    oak_card = sensor_card(page, OAK_SENSOR)
    zed_card = sensor_card(page, ZED_SENSOR)
    toggle = oak_card.locator('[data-testid="sensor-preview-toggle"]')
    expect(page.locator('[data-testid="sensor-card"]')).to_have_count(5)
    expect(oak_card.locator('[data-testid="sensor-preview-image"]')).to_be_visible(
        timeout=4_000
    )
    expect(toggle).to_have_attribute("aria-pressed", "true")

    oak_card.scroll_into_view_if_needed()
    toggle.click()
    wait_for(lambda: runner.canceled == ["preview-1"])

    expect(toggle).to_have_attribute("aria-pressed", "false")
    expect(oak_card.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)
    expect(cards).to_have_count(5)
    expect(grid).to_be_visible()
    zed_card.scroll_into_view_if_needed()
    expect(zed_card).to_be_visible()


def test_inverted_change_restarts_preview_in_waiting_state(preview_server, page) -> None:
    server, runner = preview_server
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")
    first = sensor_card(page, SENSOR_A)
    toggle = first.locator('[data-testid="sensor-preview-toggle"]')

    toggle.click()
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

    first.locator('[data-testid="sensor-orientation"]').click()
    page.get_by_role("option", name="Inverted").click()

    wait_for(lambda: len(runner.submitted) == 2)
    assert runner.canceled == ["preview-1"]
    command = runner.submitted[1]["command"]
    sensor_spec = json.loads(command[command.index("--sensor-json") + 1])
    assert sensor_spec["inverted"] is True
    expect(toggle).to_have_attribute("aria-pressed", "true")
    expect(first.locator('[data-testid="sensor-preview-slot"]')).to_be_visible()
    expect(first.locator(".sensor-preview-empty")).to_contain_text("Waiting")
    expect(first.locator('[data-testid="sensor-preview-image"]')).to_have_count(0)


def test_card_snapshot_lifecycle_displays_completed_thumbnail(
    preview_server,
    page,
) -> None:
    server, runner = preview_server
    page.goto(f"{server.url}/#/devices", wait_until="domcontentloaded")
    first = sensor_card(page, SENSOR_A)

    first.get_by_role("button", name="Snapshot").click()

    wait_for(lambda: len(runner.submitted) == 1)
    job = runner.jobs["preview-1"]
    snapshot_root = Path(job.parameters["snapshot_root"])
    write_jpeg(snapshot_root / "wrist" / "rgb_thumbnail.png")
    write_json(
        snapshot_root / "sensor_snapshot_manifest.json",
        {
            "schema_version": "sensor_snapshot_manifest.v1",
            "sensors": [
                {
                    "sensor_key": SENSOR_A,
                    "status": "succeeded",
                    "rgb_thumbnail": "wrist/rgb_thumbnail.png",
                    "error": None,
                }
            ],
        },
    )
    job.status = "succeeded"

    expect(first.locator('[data-testid="sensor-snapshot"] img')).to_be_visible(
        timeout=4000
    )
