from __future__ import annotations

import pytest

from posetestbot.robot.reference_frames import (
    POSE_TEMPLATE_BASE_SUNRISE_PATH,
    robot_pose_reference_evidence,
    verified_sunrise_reference_frame_path,
)


def test_pose_template_base_sunrise_path_is_canonical() -> None:
    assert POSE_TEMPLATE_BASE_SUNRISE_PATH == "/PoseTestBot/PoseTemplateBase"


def _record(path: str | None) -> dict:
    value = {"motion": "pose_0", "pose": {"X": 0}}
    if path is not None:
        value["source_packet"] = {
            "schema_version": "robot_pose.v1",
            "from_frame": "robot_flange",
            "to_frame": "template_base",
            "sunrise_reference_frame_path": path,
        }
    return value


def test_robot_pose_reference_evidence_retains_exact_v1_sunrise_path() -> None:
    evidence = robot_pose_reference_evidence(
        {
            "0": _record("/PoseTestBot/TemplateBase"),
            "1": _record("/PoseTestBot/TemplateBase"),
        }
    )

    assert evidence == {
        "schema_version": "robot_pose_reference.v1",
        "status": "verified",
        "packet_schema_version": "robot_pose.v1",
        "from": "robot_flange",
        "to": "template_base",
        "sunrise_reference_frame_path": "/PoseTestBot/TemplateBase",
        "pose_count": 2,
    }
    assert (
        verified_sunrise_reference_frame_path(evidence) == "/PoseTestBot/TemplateBase"
    )


def test_robot_pose_reference_evidence_keeps_fully_legacy_artifact_loadable() -> None:
    evidence = robot_pose_reference_evidence({"0": _record(None)})

    assert evidence["status"] == "unverified"
    assert evidence["reason"].startswith("legacy_robot_pose_packets")
    assert verified_sunrise_reference_frame_path(evidence) is None


@pytest.mark.parametrize(
    "poses, message",
    [
        (
            {
                "0": _record("/PoseTestBot/TemplateBase"),
                "1": _record("/PoseTestBot/PoseTemplateBase"),
            },
            "changes Sunrise reference-frame identity",
        ),
        (
            {"0": _record("/PoseTestBot/TemplateBase"), "1": _record(None)},
            "mixes v1 packets with legacy packets",
        ),
    ],
)
def test_robot_pose_reference_evidence_rejects_ambiguous_streams(
    poses: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        robot_pose_reference_evidence(poses)
