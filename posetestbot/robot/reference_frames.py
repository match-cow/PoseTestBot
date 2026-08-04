"""Reference-frame provenance for robot-pose streams and static calibration.

The symbolic ``template_base`` frame used by repository artifacts is not enough
to distinguish two different Sunrise Application Data frames.  Modern robot
pose packets therefore carry the exact absolute Sunrise path used to express
the streamed flange pose.  These helpers validate and retain that identity
without pretending that path equality proves a frame has not been retaught.
"""

from __future__ import annotations

from typing import Any, Mapping


ROBOT_POSE_PACKET_SCHEMA_VERSION = "robot_pose.v1"
ROBOT_POSE_REFERENCE_SCHEMA_VERSION = "robot_pose_reference.v1"
POSE_TEMPLATE_BASE_SUNRISE_PATH = "/PoseTestBot/PoseTemplateBase"


def normalize_sunrise_reference_frame_path(value: Any) -> str:
    """Return one canonical absolute Sunrise Application Data frame path."""

    if not isinstance(value, str):
        raise ValueError("Sunrise reference frame path must be a string")
    path = value.strip()
    if not path.startswith("/") or path.endswith("/") or "//" in path:
        raise ValueError(
            "Sunrise reference frame path must be absolute, must not contain "
            "empty components, and must not end with '/'"
        )
    if any(component in {".", ".."} for component in path.split("/")):
        raise ValueError("Sunrise reference frame path must not contain . or ..")
    return path


def configured_sunrise_reference_frame_path(
    config: Mapping[str, Any],
) -> str | None:
    """Read the optional exact robot-pose reference expected by a run config."""

    frames = config.get("frames")
    robot_pose = frames.get("robot_pose") if isinstance(frames, Mapping) else None
    if not isinstance(robot_pose, Mapping):
        return None
    value = robot_pose.get("sunrise_reference_frame_path")
    if value is None:
        return None
    return normalize_sunrise_reference_frame_path(value)


def robot_pose_reference_evidence(raw_poses: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one immutable reference identity from a raw robot-pose artifact.

    Fully legacy artifacts remain loadable and return explicit unverified
    evidence.  A partially annotated or identity-changing artifact is rejected:
    it is neither a coherent v1 stream nor an unambiguous legacy stream.
    """

    if not isinstance(raw_poses, Mapping) or not raw_poses:
        raise ValueError("Raw robot pose artifact must be a non-empty JSON object")

    identities: set[tuple[str, str, str, str]] = set()
    annotated_count = 0
    legacy_count = 0
    for key, raw_record in raw_poses.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Robot pose {key!r} must be a JSON object")
        packet = raw_record.get("source_packet")
        if packet is None:
            legacy_count += 1
            continue
        if not isinstance(packet, Mapping):
            raise ValueError(f"Robot pose {key!r} source_packet must be an object")
        annotated_count += 1
        schema_version = str(packet.get("schema_version") or "")
        from_frame = str(packet.get("from_frame") or "")
        to_frame = str(packet.get("to_frame") or "")
        if schema_version != ROBOT_POSE_PACKET_SCHEMA_VERSION:
            raise ValueError(
                f"Robot pose {key!r} source packet schema must be "
                f"{ROBOT_POSE_PACKET_SCHEMA_VERSION}"
            )
        if from_frame != "robot_flange" or to_frame != "template_base":
            raise ValueError(
                f"Robot pose {key!r} source packet must map robot_flange to "
                "template_base"
            )
        path = normalize_sunrise_reference_frame_path(
            packet.get("sunrise_reference_frame_path")
        )
        identities.add((schema_version, from_frame, to_frame, path))

    if annotated_count == 0:
        return {
            "schema_version": ROBOT_POSE_REFERENCE_SCHEMA_VERSION,
            "status": "unverified",
            "reason": "legacy_robot_pose_packets_omit_sunrise_reference_frame_path",
            "pose_count": legacy_count,
        }
    if legacy_count:
        raise ValueError(
            "Raw robot pose artifact mixes v1 packets with legacy packets that omit "
            "Sunrise reference-frame provenance"
        )
    if len(identities) != 1:
        raise ValueError(
            "Raw robot pose artifact changes Sunrise reference-frame identity"
        )
    schema_version, from_frame, to_frame, path = next(iter(identities))
    return {
        "schema_version": ROBOT_POSE_REFERENCE_SCHEMA_VERSION,
        "status": "verified",
        "packet_schema_version": schema_version,
        "from": from_frame,
        "to": to_frame,
        "sunrise_reference_frame_path": path,
        "pose_count": annotated_count,
    }


def verified_sunrise_reference_frame_path(value: Any) -> str | None:
    """Return the path from verified reference evidence, else ``None``.

    The function is intentionally strict for a mapping claiming ``verified``;
    malformed profile metadata must not silently degrade into legacy status.
    """

    if not isinstance(value, Mapping) or value.get("status") != "verified":
        return None
    if value.get("schema_version") != ROBOT_POSE_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            "Verified robot-pose reference evidence has an unsupported schema"
        )
    if value.get("packet_schema_version") != ROBOT_POSE_PACKET_SCHEMA_VERSION:
        raise ValueError(
            "Verified robot-pose reference evidence must originate from robot_pose.v1"
        )
    if value.get("from") != "robot_flange" or value.get("to") != "template_base":
        raise ValueError(
            "Verified robot-pose reference evidence must map robot_flange to "
            "template_base"
        )
    return normalize_sunrise_reference_frame_path(
        value.get("sunrise_reference_frame_path")
    )
