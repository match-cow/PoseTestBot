from __future__ import annotations

import math
import re
from pathlib import Path


JAVA_PATH = Path("iiwa/PoseTestBotSingleFrameStaticCameraCalibrationApplication.java")


def test_iiwa_application_names_describe_their_runtime_roles() -> None:
    applications = {path.name for path in Path("iiwa").glob("*Application.java")}

    assert applications == {
        "PoseTestBotFullCaptureApplication.java",
        "PoseTestBotNineFrameCalibrationApplication.java",
        "PoseTestBotSingleFrameStaticCameraCalibrationApplication.java",
    }
    assert Path("iiwa/PoseTestBotPoseStreamTask.java").is_file()
    for path in [
        *(Path("iiwa").glob("*Application.java")),
        Path("iiwa/PoseTestBotPoseStreamTask.java"),
    ]:
        java = path.read_text()
        assert f"public class {path.stem}" in java


def test_iiwa_application_initialization_never_commands_motion() -> None:
    for path in Path("iiwa").glob("*Application.java"):
        java = path.read_text()
        initialize_body = java.split("public void initialize()", 1)[1].split(
            "private ObjectFrame requiredFrame", 1
        )[0]
        assert "robot.move(" not in initialize_body, path.name


def test_static_camera_application_requires_only_the_requested_taught_center() -> None:
    java = JAVA_PATH.read_text()
    application_data_paths = set(re.findall(r'"(/PoseTestBot/[^"]+)"', java))

    assert application_data_paths == {
        "/PoseTestBot/PoseTemplateBase",
        "/PoseTestBot/PoseTemplateBase/CalibrationStatiCenter",
    }
    assert (
        "calibrationStatiCenter = requiredFrame(\n"
        "\t\t\t\tCALIBRATION_STATI_CENTER_PATH);" in java
    )
    assert "robotinfo.setBase(POSE_TEMPLATE_BASE_PATH);" in java
    assert "new Frame(" not in java
    assert re.findall(r"private ObjectFrame ([A-Za-z0-9_]+);", java) == [
        "calibrationStatiCenter"
    ]


def test_static_camera_grid_stays_inside_the_100_mm_center_envelope() -> None:
    java = JAVA_PATH.read_text()

    half_span_match = re.search(r"GRID_HALF_SPAN_MM = ([0-9.]+);", java)
    depth_match = re.search(r"DEPTH_DITHER_MM = ([0-9.]+);", java)
    limit_match = re.search(r"MAX_CENTER_TRANSLATION_MM = ([0-9.]+);", java)
    start_limit_match = re.search(r"MAX_START_TRANSLATION_MM = ([0-9.]+);", java)
    assert half_span_match is not None
    assert depth_match is not None
    assert limit_match is not None
    assert start_limit_match is not None
    half_span = float(half_span_match.group(1))
    depth = float(depth_match.group(1))
    limit = float(limit_match.group(1))
    start_limit = float(start_limit_match.group(1))

    assert half_span == 65.0
    assert math.hypot(half_span, half_span) < limit == 100.0
    assert depth == 50.0 < limit
    assert start_limit == 25.0 < limit
    assert "radiusMm > MAX_CENTER_TRANSLATION_MM" in java
    assert "validateProgramEnvelope();" in java

    grid_body = java.split("private void runRelativePlanarGrid", 1)[1].split(
        "private void captureGridPoint", 1
    )[0]
    grid_calls = re.findall(
        r"captureGridPoint\(([^,]+), ([^,]+),\s*"
        r'cartVelocityMmS, "([^"]+)"\);',
        grid_body,
    )
    assert grid_calls == [
        ("-GRID_HALF_SPAN_MM", "GRID_HALF_SPAN_MM", "grid_upper_left"),
        ("0.0", "GRID_HALF_SPAN_MM", "grid_upper_center"),
        ("GRID_HALF_SPAN_MM", "GRID_HALF_SPAN_MM", "grid_upper_right"),
        ("GRID_HALF_SPAN_MM", "0.0", "grid_middle_right"),
        ("GRID_HALF_SPAN_MM", "-GRID_HALF_SPAN_MM", "grid_lower_right"),
        ("0.0", "-GRID_HALF_SPAN_MM", "grid_lower_center"),
        ("-GRID_HALF_SPAN_MM", "-GRID_HALF_SPAN_MM", "grid_lower_left"),
        ("-GRID_HALF_SPAN_MM", "0.0", "grid_middle_left"),
    ]
    assert "captureRelativePose(-xMm, -yMm, 0.0, 0.0, 0.0, 0.0," in java

    depth_body = java.split("private void runRelativeDepthDither", 1)[1].split(
        "private void captureDepthPoint", 1
    )[0]
    assert re.findall(
        r"captureDepthPoint\(([^,]+), cartVelocityMmS,\s*" r'"([^"]+)"\);',
        depth_body,
    ) == [
        ("DEPTH_DITHER_MM", "depth_plus"),
        ("-DEPTH_DITHER_MM", "depth_minus"),
    ]


def test_static_camera_program_waits_for_start_before_any_robot_motion() -> None:
    java = JAVA_PATH.read_text()
    run_body = java.split("public void run()", 1)[1].split(
        "private void runCapture", 1
    )[0]
    capture_body = java.split("private void runCapture", 1)[1].split(
        "private void runRelativePlanarGrid", 1
    )[0]

    assert run_body.index("waitForStartCommand()") < run_body.index(
        "runCapture(command)"
    )
    assert "robot.move(" not in run_body
    assert capture_body.index("requireCurrentPositionNearCenter();") < (
        capture_body.index("poseStream.configure(")
    )
    assert capture_body.index("poseStream.configure(") < capture_body.index(
        'moveToCenter("capture start anchor")'
    )
    assert capture_body.index('moveToCenter("capture start anchor")') < (
        capture_body.index("runRelativePlanarGrid(cartVelocityMmS)")
    )
    assert capture_body.index('moveToCenter("capture end anchor")') < (
        capture_body.index("poseStream.finishCapture();")
    )
    assert "robot.getCurrentCartesianPosition(" in java
    assert "radiusMm > MAX_START_TRANSLATION_MM" in java


def test_static_camera_relative_motion_and_pose_stream_contracts_are_preserved() -> (
    None
):
    java = JAVA_PATH.read_text()
    relative_motion_body = java.split("private void captureRelativePose", 1)[1].split(
        "private void settleAtCurrentPose", 1
    )[0]

    assert "Transformation.ofDeg(" in java
    assert "linRel(offset, calibrationStatiCenter)" in java
    assert "new Frame(" not in java
    assert "command.runId,\n\t\t\t\tPOSE_TEMPLATE_BASE_PATH);" in java
    assert "poseStream.startMotion(motionName);" in java
    assert "sentPoseCount = poseStream.stopMotion();" in java
    assert 'poseStream.sendCurrentPose(motionName + "_settled")' in java
    assert "SETTLE_TIME_MS = 1500" in java
    assert "CAPTURE_VELOCITY_SCALE = 0.60" in java
    assert "RELATIVE_MOTION_JOINT_VEL_REL = 0.03" in java
    assert ".setJointAccelerationRel(SMOOTH_MOTION_JOINT_ACCEL_REL)" in java
    assert ".setJointJerkRel(SMOOTH_MOTION_JOINT_JERK_REL)" in java
    assert "moveAsync(" not in java
    assert ".isFinished()" not in java
    assert relative_motion_body.index("requireInsideCenterEnvelope(") < (
        relative_motion_body.index("linRel(offset, calibrationStatiCenter)")
    )


def test_static_camera_program_adds_depth_and_multi_axis_orientation_diversity() -> (
    None
):
    java = JAVA_PATH.read_text()
    orientation_body = java.split("private void runRelativeOrientationDither", 1)[
        1
    ].split("private void captureOrientationPoint", 1)[0]

    assert "captureDepthPoint(DEPTH_DITHER_MM" in java
    assert "captureDepthPoint(-DEPTH_DITHER_MM" in java
    assert orientation_body.count("captureOrientationPoint(") == 6
    assert "ORIENTATION_DITHER_DEG = 10.0" in java
    assert "-alphaDeg, -betaDeg, -gammaDeg," in java
