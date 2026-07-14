"""Queued UGREEN room-monitor capture and WebRTC signaling."""

from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from posetestbot.io.atomic import atomic_write_json
from posetestbot.io.manifest import utc_now_iso
from posetestbot.sensors.v4l2_preview import (
    open_v4l2_capture,
    select_usb_rgb_node,
)


MONITOR_STATUS_NAME = "monitor_webrtc_status.json"
MONITOR_STATUS_SCHEMA = "monitor_webrtc.v1"
DEFAULT_MONITOR_ROOT = Path("working_data") / "monitor_webrtc"
UGREEN_USB_VENDOR_ID = "0c45"
UGREEN_USB_PRODUCT_ID = "2283"
VIDEO_CLOCK_RATE = 90_000
MAX_SDP_BYTES = 256 * 1024


def monitor_stream_root(
    root: str | Path = DEFAULT_MONITOR_ROOT,
    *,
    monitor_id: str | None = None,
) -> Path:
    return Path(root) / (monitor_id or uuid.uuid4().hex[:12])


def build_monitor_webrtc_command(
    *,
    monitor_root: str | Path,
    vendor_id: str = UGREEN_USB_VENDOR_ID,
    product_id: str = UGREEN_USB_PRODUCT_ID,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/run_monitor_webrtc.py",
        Path(monitor_root).as_posix(),
        "--vendor-id",
        vendor_id,
        "--product-id",
        product_id,
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
    ]


def load_monitor_status(monitor_root: str | Path) -> dict[str, Any] | None:
    path = Path(monitor_root) / MONITOR_STATUS_NAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def public_monitor_status(status: Mapping[str, Any]) -> dict[str, Any]:
    """Return browser-safe status without the private loopback port."""

    return {
        key: value
        for key, value in status.items()
        if key not in {"signaling_port", "monitor_root"}
    }


class MonitorStatusWriter:
    def __init__(self, monitor_root: str | Path) -> None:
        self.root = Path(monitor_root)
        self.path = self.root / MONITOR_STATUS_NAME
        self._value: dict[str, Any] = {
            "schema_version": MONITOR_STATUS_SCHEMA,
            "generated_at": utc_now_iso(),
            "transport": "webrtc",
            "status": "starting",
            "signaling_ready": False,
            "signaling_port": None,
            "peer_count": 0,
            "frame_count": 0,
            "selected_node": None,
            "error": None,
        }
        self.write()

    @property
    def value(self) -> dict[str, Any]:
        return dict(self._value)

    def update(self, **changes: Any) -> None:
        self._value.update(changes)
        self.write()

    def write(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._value["generated_at"] = utc_now_iso()
        atomic_write_json(self.path, self._value)


def normalize_bgr_frame(frame: Any) -> Any:
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def video_timestamp(frame_index: int, *, fps: int = 30) -> tuple[int, Fraction]:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return round(frame_index * VIDEO_CLOCK_RATE / fps), Fraction(1, VIDEO_CLOCK_RATE)


def bgr_frame_to_av(frame: Any, *, frame_index: int, fps: int = 30) -> VideoFrame:
    normalized = normalize_bgr_frame(frame)
    if normalized is None:
        raise ValueError("Cannot convert an empty camera frame")
    pts, time_base = video_timestamp(frame_index, fps=fps)
    video_frame = VideoFrame.from_ndarray(normalized, format="bgr24")
    video_frame.pts = pts
    video_frame.time_base = time_base
    return video_frame


class OpenCVVideoTrack(VideoStreamTrack):
    """Pull unbuffered camera frames and expose them as timestamped PyAV frames."""

    def __init__(
        self,
        capture: Any,
        *,
        fps: int,
        selected_path: str,
        on_frame: Callable[[int], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.capture = capture
        self.fps = fps
        self.selected_path = selected_path
        self.frame_count = 0
        self._failure_count = 0
        self._on_frame = on_frame
        self._on_error = on_error
        self._recv_idle = asyncio.Event()
        self._recv_idle.set()

    def _read_frame(self) -> Any:
        failure_limit = max(10, self.fps * 5)
        while True:
            if self.readyState != "live":
                raise MediaStreamError
            ok, frame = self.capture.read()
            if ok and frame is not None:
                self._failure_count = 0
                return frame
            self._failure_count += 1
            if self._failure_count > failure_limit:
                raise RuntimeError(f"No RGB frames received from {self.selected_path}.")
            time.sleep(min(0.2, 1.0 / self.fps))

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        self._recv_idle.clear()
        try:
            try:
                frame = await asyncio.to_thread(self._read_frame)
                if self.readyState != "live":
                    raise MediaStreamError
                video_frame = bgr_frame_to_av(
                    frame,
                    frame_index=self.frame_count,
                    fps=self.fps,
                )
                self.frame_count += 1
                if self._on_frame is not None:
                    self._on_frame(self.frame_count)
                return video_frame
            except asyncio.CancelledError:
                raise
            except MediaStreamError:
                raise
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(f"{type(exc).__name__}: {exc}")
                self.stop()
                raise MediaStreamError from exc
        finally:
            self._recv_idle.set()

    def stop(self) -> None:
        if self.readyState == "live":
            super().stop()
            self.capture.release()

    async def wait_stopped(self) -> None:
        await self._recv_idle.wait()


def prefer_vp8(peer_connection: RTCPeerConnection) -> None:
    codecs = list(RTCRtpSender.getCapabilities("video").codecs)
    codecs.sort(key=lambda codec: codec.mimeType.lower() != "video/vp8")
    for transceiver in peer_connection.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(codecs)


class MonitorWebRTCServer:
    """Loopback-only SDP server sharing one live track across browser peers."""

    def __init__(
        self,
        track: VideoStreamTrack,
        *,
        on_peers_changed: Callable[[int, int], None] | None = None,
    ) -> None:
        self.track = track
        self.relay = MediaRelay()
        self._peers: set[RTCPeerConnection] = set()
        self._on_peers_changed = on_peers_changed
        self._runner: web.AppRunner | None = None
        self._socket: socket.socket | None = None

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def connected_peer_count(self) -> int:
        return sum(peer.connectionState == "connected" for peer in self._peers)

    def _notify_peers_changed(self) -> None:
        if self._on_peers_changed is not None:
            self._on_peers_changed(self.peer_count, self.connected_peer_count)

    async def start(self) -> int:
        app = web.Application(client_max_size=MAX_SDP_BYTES + 4096)
        app.router.add_post("/offer", self._offer_request)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        self._socket = sock
        site = web.SockSite(self._runner, sock)
        await site.start()
        return int(sock.getsockname()[1])

    async def _offer_request(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, web.HTTPException):
            return web.json_response({"error": "Malformed JSON offer"}, status=400)
        if not isinstance(payload, Mapping):
            return web.json_response({"error": "Offer must be a JSON object"}, status=400)
        offer_type = payload.get("type")
        sdp = payload.get("sdp")
        if offer_type != "offer" or not isinstance(sdp, str) or not sdp.strip():
            return web.json_response({"error": "Expected a non-empty SDP offer"}, status=400)
        if len(sdp.encode("utf-8")) > MAX_SDP_BYTES:
            return web.json_response({"error": "SDP offer is too large"}, status=400)
        try:
            answer = await self.accept_offer(sdp)
        except Exception as exc:
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=500,
            )
        return web.json_response(answer)

    async def accept_offer(self, sdp: str) -> dict[str, str]:
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._peers.add(peer)
        self._notify_peers_changed()

        @peer.on("connectionstatechange")
        async def connectionstatechange() -> None:
            self._notify_peers_changed()
            if peer.connectionState in {"failed", "closed"}:
                await self._discard_peer(peer)

        try:
            peer.addTrack(self.relay.subscribe(self.track, buffered=False))
            prefer_vp8(peer)
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            if peer.localDescription is None:
                raise RuntimeError("WebRTC answer did not produce a local description")
            return {
                "type": peer.localDescription.type,
                "sdp": peer.localDescription.sdp,
            }
        except Exception:
            await self._discard_peer(peer)
            raise

    async def _discard_peer(self, peer: RTCPeerConnection) -> None:
        self._peers.discard(peer)
        if peer.connectionState != "closed":
            await peer.close()
        self._notify_peers_changed()

    async def stop(self) -> None:
        self.track.stop()
        wait_stopped = getattr(self.track, "wait_stopped", None)
        if callable(wait_stopped):
            try:
                await asyncio.wait_for(wait_stopped(), timeout=2)
            except TimeoutError:
                pass
        await asyncio.sleep(0)
        peers = list(self._peers)
        self._peers.clear()
        if peers:
            await asyncio.gather(*(peer.close() for peer in peers), return_exceptions=True)
        self._notify_peers_changed()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None


async def run_monitor_webrtc(
    monitor_root: str | Path,
    *,
    stop_event: asyncio.Event,
    vendor_id: str = UGREEN_USB_VENDOR_ID,
    product_id: str = UGREEN_USB_PRODUCT_ID,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> int:
    status = MonitorStatusWriter(monitor_root)
    capture = None
    track: OpenCVVideoTrack | None = None
    server: MonitorWebRTCServer | None = None
    fatal_event = asyncio.Event()
    fatal_error: list[str] = []

    def on_frame(frame_count: int) -> None:
        if frame_count == 1 or frame_count % fps == 0:
            status.update(frame_count=frame_count, error=None)

    def on_track_error(error: str) -> None:
        fatal_error[:] = [error]
        status.update(status="failed", signaling_ready=False, error=error)
        fatal_event.set()

    def on_peers_changed(peer_count: int, connected_count: int) -> None:
        if status.value["status"] == "failed":
            status.update(peer_count=peer_count)
            return
        current_status = "connected" if connected_count else "ready"
        status.update(status=current_status, peer_count=peer_count)

    try:
        selection = select_usb_rgb_node(vendor_id, product_id)
        if "MJPG" not in selection.formats:
            raise RuntimeError(
                f"UGREEN node {selection.path} does not advertise MJPEG capture."
            )
        status.update(status="opening", selected_node=selection.as_dict())
        capture = open_v4l2_capture(
            selection.path,
            width=width,
            height=height,
            fps=fps,
            pixel_format="MJPG",
        )
        track = OpenCVVideoTrack(
            capture,
            fps=fps,
            selected_path=selection.path,
            on_frame=on_frame,
            on_error=on_track_error,
        )
        server = MonitorWebRTCServer(track, on_peers_changed=on_peers_changed)
        port = await server.start()
        status.update(
            status="ready",
            signaling_ready=True,
            signaling_port=port,
            peer_count=0,
            error=None,
        )

        stop_task = asyncio.create_task(stop_event.wait())
        fatal_task = asyncio.create_task(fatal_event.wait())
        done, pending = await asyncio.wait(
            {stop_task, fatal_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if fatal_task in done and fatal_error:
            return 2
        return 0
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status.update(
            status="failed",
            signaling_ready=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 2
    finally:
        if track is not None:
            track.stop()
        if server is not None:
            await server.stop()
        if capture is not None:
            capture.release()
        if status.value["status"] != "failed":
            status.update(
                status="stopped",
                signaling_ready=False,
                signaling_port=None,
                peer_count=0,
                frame_count=track.frame_count if track is not None else 0,
            )
