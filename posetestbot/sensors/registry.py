"""Registry of supported RGB-D sensor capture adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from posetestbot.sensors.contracts import MountingMode, SensorType


AUTO_DEVICE_IDS = {"", "auto", "default"}
REALSENSE_HARDWARE_SYNC_TRANSPORT = "realsense_inter_cam_sync"
HARDWARE_SYNC_SCOPE_DEPTH_EXPOSURE = "depth_exposure"
REALSENSE_HARDWARE_SYNC_ROLES = ("master", "subordinate")


@dataclass(frozen=True)
class HardwareSyncCapability:
    """Static hardware-trigger capability for one sensor adapter."""

    transport: str | None = None
    supported_scopes: tuple[str, ...] = ()
    supported_roles: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return bool(self.transport and self.supported_scopes and self.supported_roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "transport": self.transport,
            "supported_scopes": list(self.supported_scopes),
            "supported_roles": list(self.supported_roles),
        }


@dataclass(frozen=True)
class SensorAdapterSpec:
    """Static capabilities for one supported sensor family."""

    sensor_type: SensorType
    display_name: str
    sdk_module: str
    capture_script: str
    folder_prefix: str
    supported_resolutions: tuple[str, ...]
    live_rgb_preview_supported: bool = False
    default_resolution: str = "720p"
    mounting_modes: tuple[MountingMode, ...] = (
        MountingMode.EYE_IN_HAND,
        MountingMode.STATIC,
    )
    aligned_depth_to: str = "rgb"
    timestamp_source: str = "sensor_and_host"
    hardware_sync: HardwareSyncCapability = field(
        default_factory=HardwareSyncCapability
    )
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sensor_type"] = self.sensor_type.value
        data["supported_resolutions"] = list(self.supported_resolutions)
        data["mounting_modes"] = [mode.value for mode in self.mounting_modes]
        data["hardware_sync"] = self.hardware_sync.to_dict()
        data["notes"] = list(self.notes)
        return data


SENSOR_ADAPTERS: dict[SensorType, SensorAdapterSpec] = {
    SensorType.REALSENSE_D435: SensorAdapterSpec(
        sensor_type=SensorType.REALSENSE_D435,
        display_name="Intel RealSense D435",
        sdk_module="pyrealsense2",
        capture_script="scripts/capture_realsense_720p.py",
        folder_prefix="realsense",
        supported_resolutions=("720p",),
        live_rgb_preview_supported=True,
        hardware_sync=HardwareSyncCapability(
            transport=REALSENSE_HARDWARE_SYNC_TRANSPORT,
            supported_scopes=(HARDWARE_SYNC_SCOPE_DEPTH_EXPOSURE,),
            supported_roles=REALSENSE_HARDWARE_SYNC_ROLES,
        ),
        notes=(
            "Depth is aligned to color by the capture script.",
            "Inter-camera hardware synchronization covers depth exposure only; "
            "it does not claim synchronized RGB exposure.",
            "Current capture script is 720p-only.",
        ),
    ),
    SensorType.OAK_D_PRO: SensorAdapterSpec(
        sensor_type=SensorType.OAK_D_PRO,
        display_name="Luxonis OAK-D Pro",
        sdk_module="depthai",
        capture_script="scripts/capture_luxonis_720p.py",
        folder_prefix="luxonis",
        supported_resolutions=("720p",),
        live_rgb_preview_supported=True,
        notes=(
            "DepthAI stereo depth is aligned to RGB by the capture script.",
            "Current capture script is 720p-only.",
        ),
    ),
    SensorType.ZED_2I: SensorAdapterSpec(
        sensor_type=SensorType.ZED_2I,
        display_name="Stereolabs ZED 2i",
        sdk_module="pyzed.sl",
        capture_script="scripts/capture_zed_2i.py",
        folder_prefix="zed_2i",
        supported_resolutions=("720p", "360p"),
        notes=(
            "Captures left RGB plus depth aligned to the left RGB stream.",
            "Requires the Stereolabs ZED SDK Python module outside ordinary PyPI.",
        ),
    ),
}


def is_auto_device_id(device_id: str) -> bool:
    return device_id.strip().lower() in AUTO_DEVICE_IDS


def get_sensor_adapter(sensor_type: SensorType | str) -> SensorAdapterSpec:
    try:
        normalized = (
            sensor_type
            if isinstance(sensor_type, SensorType)
            else SensorType(sensor_type)
        )
    except ValueError as exc:
        valid = ", ".join(sensor.value for sensor in SensorType)
        raise ValueError(
            f"Unsupported sensor type {str(sensor_type)!r}; expected one of: {valid}"
        ) from exc
    try:
        return SENSOR_ADAPTERS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"No capture adapter is registered for {normalized.value}"
        ) from exc


def list_sensor_adapters() -> list[dict[str, Any]]:
    return [
        adapter.to_dict()
        for adapter in sorted(
            SENSOR_ADAPTERS.values(), key=lambda item: item.sensor_type.value
        )
    ]


def sensor_folder_name(sensor_type: SensorType | str, device_id: str) -> str:
    adapter = get_sensor_adapter(sensor_type)
    suffix = "auto" if is_auto_device_id(device_id) else device_id.strip()
    return f"{adapter.folder_prefix}_{suffix}"


def capture_script_for_sensor(sensor_type: SensorType | str, resolution: str) -> str:
    adapter = get_sensor_adapter(sensor_type)
    if resolution not in adapter.supported_resolutions:
        supported = ", ".join(adapter.supported_resolutions)
        raise ValueError(
            f"{adapter.display_name} capture planning supports {supported}; "
            f"got {resolution!r}."
        )
    return adapter.capture_script


def validate_hardware_sync_request(
    *,
    sensor_type: SensorType | str,
    hardware_sync_role: str | None,
    hardware_sync_group_id: str | None,
    hardware_sync_scope: str | None,
) -> dict[str, str] | None:
    """Validate one complete, adapter-supported hardware-sync request."""

    values = {
        "hardware_sync_role": hardware_sync_role,
        "hardware_sync_group_id": hardware_sync_group_id,
        "hardware_sync_scope": hardware_sync_scope,
    }
    if all(value is None for value in values.values()):
        return None

    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "Hardware synchronization requires role, group ID, and scope together; "
            "missing: " + ", ".join(missing)
        )
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise ValueError(
            "Hardware synchronization role, group ID, and scope must be "
            "non-empty strings"
        )

    adapter = get_sensor_adapter(sensor_type)
    capability = adapter.hardware_sync
    if not capability.supported:
        raise ValueError(
            f"{adapter.display_name} does not support hardware synchronization"
        )

    role = str(hardware_sync_role).strip().lower()
    scope = str(hardware_sync_scope).strip().lower()
    group_id = str(hardware_sync_group_id).strip()
    if role not in capability.supported_roles:
        raise ValueError(
            f"{adapter.display_name} hardware_sync_role must be one of: "
            + ", ".join(capability.supported_roles)
        )
    if scope not in capability.supported_scopes:
        raise ValueError(
            f"{adapter.display_name} hardware_sync_scope must be one of: "
            + ", ".join(capability.supported_scopes)
        )
    return {
        "role": role,
        "group_id": group_id,
        "scope": scope,
        "transport": str(capability.transport),
    }


def build_sensor_capture_command(
    *,
    sensor_type: SensorType | str,
    device_id: str,
    output_folder: str,
    fps: int,
    resolution: str,
    max_frames: int | None = None,
    warmup_frames: int | None = None,
    inverted: bool = False,
    hardware_sync_role: str | None = None,
    hardware_sync_group_id: str | None = None,
    hardware_sync_scope: str | None = None,
) -> list[str]:
    """Build the current script-backed capture command for one sensor."""

    normalized = (
        sensor_type if isinstance(sensor_type, SensorType) else SensorType(sensor_type)
    )
    if inverted and normalized != SensorType.REALSENSE_D435:
        raise ValueError("Sensor inverted=true is only supported for RealSense D435")
    if warmup_frames is not None and warmup_frames < 0:
        raise ValueError("warmup_frames must be greater than or equal to 0")
    hardware_sync = validate_hardware_sync_request(
        sensor_type=normalized,
        hardware_sync_role=hardware_sync_role,
        hardware_sync_group_id=hardware_sync_group_id,
        hardware_sync_scope=hardware_sync_scope,
    )

    script = capture_script_for_sensor(sensor_type, resolution)
    command = [
        "uv",
        "run",
        "python",
        script,
        output_folder,
        "--fps",
        str(fps),
    ]
    if max_frames and max_frames > 0:
        command.extend(["--max_frames", str(max_frames)])
    if warmup_frames and warmup_frames > 0:
        command.extend(["--warmup-frames", str(warmup_frames)])
    if not is_auto_device_id(device_id):
        command.extend(["--device", device_id])
    if hardware_sync is not None:
        command.extend(
            [
                "--hardware-sync-role",
                hardware_sync["role"],
                "--hardware-sync-group-id",
                hardware_sync["group_id"],
                "--hardware-sync-scope",
                hardware_sync["scope"],
            ]
        )
    if normalized == SensorType.REALSENSE_D435 and inverted:
        command.append("--inverted")
    if normalized == SensorType.ZED_2I:
        command.extend(["--resolution", resolution])
    return command
