from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from posetestbot.calibration.teaching_plan import load_teaching_plan


def test_headless_teaching_plot_contains_metric_contract_and_all_labels(tmp_path: Path) -> None:
    svg_path = tmp_path / "teaching.svg"
    png_path = tmp_path / "teaching.png"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plot_iiwa_calibration_teaching_plan.py",
            "--svg",
            str(svg_path),
            "--png",
            str(png_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert svg_path.stat().st_size > 100_000
    assert png_path.stat().st_size > 100_000
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    second_svg = tmp_path / "teaching-second.svg"
    second_png = tmp_path / "teaching-second.png"
    second = subprocess.run(
        [
            sys.executable,
            "scripts/plot_iiwa_calibration_teaching_plan.py",
            "--svg",
            str(second_svg),
            "--png",
            str(second_png),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert second.returncode == 0, second.stderr
    assert second_svg.read_bytes() == svg_path.read_bytes()
    assert second_png.read_bytes() == png_path.read_bytes()

    svg = svg_path.read_text()
    plan = load_teaching_plan()
    assert "420 × 297 mm" in svg
    assert "Metric views use equal millimetre scales" in svg
    assert "CalibrationCenter anchors both phases" in svg
    assert "9 taught frames + program-relative orientation" in svg
    assert "Taught coverage frames / LIN raster" in svg
    assert "Program-only LIN_REL orientation" in svg
    assert "joint-space path not depicted" in svg
    assert "Center→A−→A+→Center→B−→B+→Center→C−→C+→Center" in svg
    assert "Teaching aid only—not reachability, redundancy, singularity, collision, or cable-clearance validation." in svg
    assert "NON-METRIC SCHEMATIC" in svg
    assert "RGB arrows are flange X/Y/Z axes" in svg
    assert "camera optical axis" not in svg
    assert "CalibrationDepth" not in svg
    for frame in plan["frames"]:
        assert frame["name"] in svg
    for motion in plan["phases"][1]["motions"]:
        assert motion["capture_label"] in svg


def test_plot_geometry_uses_exact_template_and_equal_metric_axes(tmp_path: Path) -> None:
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-geometry")
    namespace = runpy.run_path("scripts/plot_iiwa_calibration_teaching_plan.py")
    plan = load_teaching_plan()

    corners = namespace["_template_corners"](plan)
    assert np.ptp(corners[:, 0]) == pytest.approx(420.0)
    assert np.ptp(corners[:, 1]) == pytest.approx(297.0)
    assert np.ptp(corners[:, 2]) == pytest.approx(0.0)

    figure = namespace["build_figure"](plan)
    isometric, raster = figure.axes[:2]
    isometric_ranges = [
        np.ptp(isometric.get_xlim3d()),
        np.ptp(isometric.get_ylim3d()),
        np.ptp(isometric.get_zlim3d()),
    ]
    assert isometric_ranges[0] == pytest.approx(isometric_ranges[1])
    assert isometric_ranges[1] == pytest.approx(isometric_ranges[2])
    assert raster.get_aspect() in {1, 1.0, "equal"}
    namespace["plt"].close(figure)
