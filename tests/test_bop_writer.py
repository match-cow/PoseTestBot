from __future__ import annotations

import math
from pathlib import Path

from posetestbot.bop.writer import model_geometry_info


def test_large_model_diameter_is_exact_not_aabb_diagonal(tmp_path: Path) -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),
        (0.0, 0.0, 12.0),
    ]
    vertices.extend(vertices[index % 4] for index in range(4_997))
    path = tmp_path / "large_tetrahedron.ply"
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "element face 0",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{x} {y} {z}" for x, y, z in vertices),
        "",
    ]
    path.write_text("\n".join(lines))

    info = model_geometry_info(path)

    assert math.isclose(float(info["diameter"]), math.sqrt(160.0))
    assert not math.isclose(float(info["diameter"]), 13.0)
    geometry = info["posetestbot_geometry"]
    assert geometry["diameter_method"] == "exact_convex_hull_vertex_pairwise"
    assert geometry["vertex_count"] == 5_001
    assert geometry["convex_hull_vertex_count"] == 4
