"""Queued UGREEN room-monitor capture and WebRTC signaling."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
from aioice import stun
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.codecs import vpx as aiortc_vpx
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
MONITOR_STATUS_SCHEMA = "monitor_webrtc.v2"
LEGACY_MONITOR_STATUS_SCHEMA = "monitor_webrtc.v1"
DEFAULT_MONITOR_ROOT = Path("working_data") / "monitor_webrtc"
UGREEN_USB_VENDOR_ID = "0c45"
UGREEN_USB_PRODUCT_ID = "2283"
VIDEO_CLOCK_RATE = 90_000
MAX_SDP_BYTES = 256 * 1024
DEFAULT_STUN_PORT = 3478
HEARTBEAT_STALE_S = 5.0
PEER_CONNECT_TIMEOUT_S = 15.0
MEDIA_FRAME_STALE_S = 5.0
CAMERA_IDLE_RELEASE_S = 15.0
# aiortc 1.14 packetizes VP8 into payloads up to 1300 bytes before adding RTP,
# SRTP, UDP, and IP overhead. Those datagrams exceed Tailscale's 1280-byte
# interface MTU. The large chunks are then lost while each frame's short tail
# packet can still arrive, which looks in Chrome exactly like a connected peer
# with advancing packets but zero complete or decoded frames.
VP8_PACKET_MAX_BYTES = 1100


def configure_vp8_packet_size(
    max_payload_bytes: int = VP8_PACKET_MAX_BYTES,
) -> int:
    """Constrain aiortc VP8 payloads to leave room for transport overhead."""

    if max_payload_bytes <= 0:
        raise ValueError("max_payload_bytes must be positive")
    configured = getattr(aiortc_vpx, "PACKET_MAX", None)
    if not isinstance(configured, int) or configured <= 0:
        raise RuntimeError(
            "The installed aiortc VP8 packet-size control is unavailable."
        )
    configured = min(configured, max_payload_bytes)
    aiortc_vpx.PACKET_MAX = configured
    return configured


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
    stun_port: int = DEFAULT_STUN_PORT,
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
        "--stun-port",
        str(stun_port),
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


def _timestamp_age_s(value: Any, *, now: datetime | None = None) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max(0.0, (current - parsed).total_seconds())


def monitor_status_health(
    status: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Return whether a persisted monitor worker is safe to reuse."""

    if status is None:
        return False, "Monitor worker has not published status."
    if status.get("schema_version") != MONITOR_STATUS_SCHEMA:
        return False, "Legacy monitor status is not reusable; restart the service."
    heartbeat_age = _timestamp_age_s(status.get("heartbeat_at"), now=now)
    if heartbeat_age is None:
        return False, "Monitor heartbeat is missing or malformed."
    if heartbeat_age > HEARTBEAT_STALE_S:
        return False, f"Monitor heartbeat is stale ({heartbeat_age:.1f}s old)."
    error_reason = status.get("error_reason") or status.get("error")
    if status.get("status") == "failed":
        return False, str(error_reason or "Monitor worker failed.")
    if status.get("signaling_ready") is not True:
        return False, str(error_reason or "Monitor signaling is not ready.")
    pending_age = _timestamp_age_s(status.get("peer_connect_started_at"), now=now)
    if (
        int(status.get("peer_count") or 0) > int(status.get("connected_peer_count") or 0)
        and pending_age is not None
        and pending_age > PEER_CONNECT_TIMEOUT_S
    ):
        return False, f"A monitor peer did not connect within {PEER_CONNECT_TIMEOUT_S:.0f}s."
    if int(status.get("connected_peer_count") or 0) > 0:
        media_age = _timestamp_age_s(status.get("last_media_frame_at"), now=now)
        if media_age is None:
            connected_age = _timestamp_age_s(status.get("peer_connected_at"), now=now)
            if connected_age is not None and connected_age > MEDIA_FRAME_STALE_S:
                return False, "No monitor media frame arrived after peer connection."
        elif media_age > MEDIA_FRAME_STALE_S:
            return False, f"Monitor media frames are stale ({media_age:.1f}s old)."
    return True, None


class MonitorStatusWriter:
    def __init__(self, monitor_root: str | Path) -> None:
        self.root = Path(monitor_root)
        self.path = self.root / MONITOR_STATUS_NAME
        self._lock = threading.RLock()
        self._value: dict[str, Any] = {
            "schema_version": MONITOR_STATUS_SCHEMA,
            "generated_at": utc_now_iso(),
            "transport": "webrtc",
            "status": "starting",
            "signaling_ready": False,
            "signaling_port": None,
            "stun_port": None,
            "peer_count": 0,
            "connected_peer_count": 0,
            "peer_connect_started_at": None,
            "peer_connected_at": None,
            "camera_open": False,
            "capture_frame_count": 0,
            "media_frame_count": 0,
            "frame_count": 0,
            "last_camera_frame_at": None,
            "last_media_frame_at": None,
            "heartbeat_at": utc_now_iso(),
            "selected_node": None,
            "vp8_packet_max_bytes": VP8_PACKET_MAX_BYTES,
            "error": None,
            "error_reason": None,
        }
        self.write()

    @property
    def value(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._value)

    def update(self, **changes: Any) -> None:
        with self._lock:
            self._value.update(changes)
            self.write()

    def write(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._value["generated_at"] = utc_now_iso()
            atomic_write_json(self.path, self._value)

    def heartbeat(self, **changes: Any) -> None:
        changes["heartbeat_at"] = utc_now_iso()
        self.update(**changes)


class StunBindingProtocol(asyncio.DatagramProtocol):
    """Minimal RFC 5389 binding responder for directly routed lab clients."""

    def __init__(self) -> None:
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            request = stun.parse_message(data)
        except ValueError:
            return
        if (
            request.message_method != stun.Method.BINDING
            or request.message_class != stun.Class.REQUEST
        ):
            return
        response = stun.Message(
            message_method=stun.Method.BINDING,
            message_class=stun.Class.RESPONSE,
            transaction_id=request.transaction_id,
            attributes={"XOR-MAPPED-ADDRESS": addr},
        )
        if self.transport is not None:
            self.transport.sendto(bytes(response), addr)


async def start_stun_binding_responder(
    port: int = DEFAULT_STUN_PORT,
    *,
    host: str = "0.0.0.0",
) -> tuple[asyncio.DatagramTransport, StunBindingProtocol, int]:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        StunBindingProtocol,
        local_addr=(host, int(port)),
    )
    sockname = transport.get_extra_info("sockname")
    return transport, protocol, int(sockname[1])


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
        on_capture_frame: Callable[[int], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.capture = capture
        self.fps = fps
        self.selected_path = selected_path
        self.frame_count = 0
        self._failure_count = 0
        self._on_frame = on_frame
        self._on_capture_frame = on_capture_frame
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
                if self._on_capture_frame is not None:
                    self._on_capture_frame(self.frame_count + 1)
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
        track: VideoStreamTrack | None = None,
        *,
        track_factory: Callable[[], VideoStreamTrack] | None = None,
        on_peers_changed: Callable[[int, int], None] | None = None,
        on_camera_open_changed: Callable[[bool], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        peer_connect_timeout_s: float = PEER_CONNECT_TIMEOUT_S,
        camera_idle_release_s: float = CAMERA_IDLE_RELEASE_S,
    ) -> None:
        if track is None and track_factory is None:
            raise ValueError("track or track_factory is required")
        configure_vp8_packet_size()
        self.track = track
        self._track_factory = track_factory
        self.relay = MediaRelay() if track is not None else None
        self._peers: set[RTCPeerConnection] = set()
        self._peer_started: dict[RTCPeerConnection, float] = {}
        self._on_peers_changed = on_peers_changed
        self._on_camera_open_changed = on_camera_open_changed
        self._on_error = on_error
        self._peer_connect_timeout_s = peer_connect_timeout_s
        self._camera_idle_release_s = camera_idle_release_s
        self._without_connected_since: float | None = None
        self._maintenance_task: asyncio.Task[Any] | None = None
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
        self._maintenance_task = asyncio.create_task(self._maintain())
        return int(sock.getsockname()[1])

    async def _ensure_track(self) -> VideoStreamTrack:
        if self.track is not None and self.track.readyState == "live":
            return self.track
        if self._track_factory is None:
            raise RuntimeError("Monitor media track is no longer available.")
        # Camera creation occurs in the dedicated managed worker, never in the
        # Flask request process.  Keeping it here also avoids an executor whose
        # threads could outlive worker shutdown while a V4L2 open is pending.
        self.track = self._track_factory()
        self.relay = MediaRelay()
        self._without_connected_since = time.monotonic()
        if self._on_camera_open_changed is not None:
            self._on_camera_open_changed(True)
        return self.track

    async def _release_track(self) -> None:
        track = self.track
        if track is None or self._track_factory is None:
            return
        track.stop()
        wait_stopped = getattr(track, "wait_stopped", None)
        if callable(wait_stopped):
            try:
                await asyncio.wait_for(wait_stopped(), timeout=2)
            except TimeoutError:
                pass
        self.track = None
        self.relay = None
        self._without_connected_since = None
        if self._on_camera_open_changed is not None:
            self._on_camera_open_changed(False)

    async def _maintain(self) -> None:
        try:
            while True:
                await asyncio.sleep(1)
                now = time.monotonic()
                timed_out = [
                    peer
                    for peer, started in self._peer_started.items()
                    if peer.connectionState != "connected"
                    and now - started > self._peer_connect_timeout_s
                ]
                if timed_out and self._on_error is not None:
                    self._on_error(
                        f"WebRTC peer did not connect within "
                        f"{self._peer_connect_timeout_s:.0f} seconds."
                    )
                for peer in timed_out:
                    await self._discard_peer(peer)

                if self.connected_peer_count:
                    self._without_connected_since = None
                elif self.track is not None and self._track_factory is not None:
                    if self._without_connected_since is None:
                        self._without_connected_since = now
                    if now - self._without_connected_since >= self._camera_idle_release_s:
                        for peer in list(self._peers):
                            await self._discard_peer(peer)
                        await self._release_track()
        except asyncio.CancelledError:
            return

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
        self._peer_started[peer] = time.monotonic()
        self._notify_peers_changed()

        @peer.on("connectionstatechange")
        async def connectionstatechange() -> None:
            self._notify_peers_changed()
            if peer.connectionState == "connected":
                self._without_connected_since = None
            if peer.connectionState in {"failed", "closed"}:
                await self._discard_peer(peer)

        try:
            track = await self._ensure_track()
            assert self.relay is not None
            peer.addTrack(self.relay.subscribe(track, buffered=False))
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
        self._peer_started.pop(peer, None)
        if peer.connectionState != "closed":
            await peer.close()
        self._notify_peers_changed()

    async def stop(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None
        if self.track is not None:
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
        self.track = None
        self.relay = None
        if self._on_camera_open_changed is not None:
            self._on_camera_open_changed(False)


async def run_monitor_webrtc(
    monitor_root: str | Path,
    *,
    stop_event: asyncio.Event,
    vendor_id: str = UGREEN_USB_VENDOR_ID,
    product_id: str = UGREEN_USB_PRODUCT_ID,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    stun_port: int = DEFAULT_STUN_PORT,
) -> int:
    status = MonitorStatusWriter(monitor_root)
    server: MonitorWebRTCServer | None = None
    compatibility_track: OpenCVVideoTrack | None = None
    stun_transport: asyncio.DatagramTransport | None = None
    heartbeat_task: asyncio.Task[Any] | None = None
    fatal_event = asyncio.Event()
    fatal_error: list[str] = []
    runtime: dict[str, Any] = {
        "camera_open": False,
        "capture_frame_count": 0,
        "media_frame_count": 0,
        "last_camera_frame_at": None,
        "last_media_frame_at": None,
        "peer_count": 0,
        "connected_peer_count": 0,
        "peer_connect_started_at": None,
        "peer_connected_at": None,
        "connected_since_monotonic": None,
    }

    def on_capture_frame(frame_count: int) -> None:
        runtime["capture_frame_count"] = frame_count
        runtime["last_camera_frame_at"] = utc_now_iso()

    def on_media_frame(frame_count: int) -> None:
        runtime["media_frame_count"] = frame_count
        runtime["last_media_frame_at"] = utc_now_iso()

    def on_track_error(error: str) -> None:
        fatal_error[:] = [error]
        status.update(
            status="failed",
            signaling_ready=False,
            error=error,
            error_reason=error,
        )
        fatal_event.set()

    def on_camera_open_changed(camera_open: bool) -> None:
        runtime["camera_open"] = camera_open

    def on_peers_changed(peer_count: int, connected_count: int) -> None:
        now_iso = utc_now_iso()
        previous_connected = int(runtime["connected_peer_count"])
        runtime["peer_count"] = peer_count
        runtime["connected_peer_count"] = connected_count
        runtime["peer_connect_started_at"] = (
            now_iso if peer_count > connected_count else None
        )
        if connected_count and not previous_connected:
            runtime["peer_connected_at"] = now_iso
            runtime["connected_since_monotonic"] = time.monotonic()
        elif not connected_count:
            runtime["peer_connected_at"] = None
            runtime["connected_since_monotonic"] = None

    async def heartbeat() -> None:
        while not stop_event.is_set() and not fatal_event.is_set():
            connected_since = runtime.get("connected_since_monotonic")
            if (
                runtime["connected_peer_count"]
                and connected_since is not None
                and time.monotonic() - float(connected_since) > MEDIA_FRAME_STALE_S
            ):
                last_media_age = _timestamp_age_s(runtime["last_media_frame_at"])
                if last_media_age is None or last_media_age > MEDIA_FRAME_STALE_S:
                    on_track_error(
                        "No WebRTC media frames were delivered for five seconds "
                        "after peer connection."
                    )
            current_status = (
                "failed"
                if fatal_event.is_set()
                else "connected"
                if runtime["connected_peer_count"]
                else "ready"
            )
            status.heartbeat(
                status=current_status,
                camera_open=runtime["camera_open"],
                capture_frame_count=runtime["capture_frame_count"],
                media_frame_count=runtime["media_frame_count"],
                frame_count=runtime["media_frame_count"],
                last_camera_frame_at=runtime["last_camera_frame_at"],
                last_media_frame_at=runtime["last_media_frame_at"],
                peer_count=runtime["peer_count"],
                connected_peer_count=runtime["connected_peer_count"],
                peer_connect_started_at=runtime["peer_connect_started_at"],
                peer_connected_at=runtime["peer_connected_at"],
            )
            await asyncio.sleep(1)

    try:
        selection = select_usb_rgb_node(vendor_id, product_id)
        if "MJPG" not in selection.formats:
            raise RuntimeError(
                f"UGREEN node {selection.path} does not advertise MJPEG capture."
            )
        status.update(status="starting", selected_node=selection.as_dict())

        def track_factory() -> OpenCVVideoTrack:
            capture_base = int(runtime["capture_frame_count"])
            media_base = int(runtime["media_frame_count"])
            capture = open_v4l2_capture(
                selection.path,
                width=width,
                height=height,
                fps=fps,
                pixel_format="MJPG",
            )
            try:
                return OpenCVVideoTrack(
                    capture,
                    fps=fps,
                    selected_path=selection.path,
                    on_frame=lambda count: on_media_frame(media_base + count),
                    on_capture_frame=lambda count: on_capture_frame(
                        capture_base + count
                    ),
                    on_error=on_track_error,
                )
            except Exception:
                capture.release()
                raise

        try:
            server = MonitorWebRTCServer(
                track_factory=track_factory,
                on_peers_changed=on_peers_changed,
                on_camera_open_changed=on_camera_open_changed,
                on_error=on_track_error,
            )
        except TypeError as exc:
            # Retain compatibility with small injected server doubles used by
            # smoke/integration callers written against the v1 constructor.
            if "unexpected keyword" not in str(exc):
                raise
            compatibility_track = track_factory()
            server = MonitorWebRTCServer(
                compatibility_track,
                on_peers_changed=on_peers_changed,
            )
        port = await server.start()
        bound_stun_port: int | None = None
        if compatibility_track is None:
            stun_transport, _stun_protocol, bound_stun_port = (
                await start_stun_binding_responder(stun_port)
            )
        status.update(
            status="ready",
            signaling_ready=True,
            signaling_port=port,
            stun_port=bound_stun_port,
            peer_count=0,
            connected_peer_count=0,
            camera_open=False,
            error=None,
            error_reason=None,
        )
        heartbeat_task = asyncio.create_task(heartbeat())

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
            error_reason=f"{type(exc).__name__}: {exc}",
        )
        return 2
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if server is not None:
            await server.stop()
        if compatibility_track is not None:
            compatibility_track.stop()
        if stun_transport is not None:
            stun_transport.close()
        if status.value["status"] != "failed":
            status.update(
                status="stopped",
                signaling_ready=False,
                signaling_port=None,
                peer_count=0,
                connected_peer_count=0,
                camera_open=False,
                capture_frame_count=runtime["capture_frame_count"],
                media_frame_count=runtime["media_frame_count"],
                frame_count=runtime["media_frame_count"],
            )
