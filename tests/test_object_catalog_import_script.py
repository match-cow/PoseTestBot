from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_import_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_object_catalog_import.py"
    spec = importlib.util.spec_from_file_location("run_object_catalog_import", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_refuses_request_folder_outside_managed_upload_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_script = load_import_script()
    working = tmp_path / "working_data"
    outside = tmp_path / "operator-owned-request"
    outside.mkdir()
    request_path = outside / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "name": "Fixture",
                "cad_path": (outside / "fixture.stl").as_posix(),
                "catalog_root": (working / "object_catalog").as_posix(),
                "cleanup_request_folder": True,
            }
        )
    )
    imports: list[dict] = []
    removals: list[Path] = []
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_object_catalog_import.py", "--request", request_path.as_posix()],
    )
    monkeypatch.setattr(
        import_script,
        "import_catalog_object",
        lambda **kwargs: imports.append(kwargs) or {},
    )
    monkeypatch.setattr(
        import_script.shutil,
        "rmtree",
        lambda path, **_kwargs: removals.append(Path(path)),
    )

    with pytest.raises(ValueError, match="catalog_upload|managed.*upload"):
        import_script.main()

    assert imports == []
    assert removals == []
    assert request_path.is_file()


@pytest.mark.parametrize("analysis_fails", [False, True])
def test_managed_import_prepares_preview_without_rolling_back_valid_catalogue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    analysis_fails: bool,
) -> None:
    import_script = load_import_script()
    working = tmp_path / "working_data"
    request_folder = (
        working
        / "jobs"
        / "workpiece_catalog_requests"
        / "catalog_upload"
        / ("a" * 32)
    )
    request_folder.mkdir(parents=True)
    request_path = request_folder / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "name": "Fixture",
                "cad_path": (request_folder / "fixture.stl").as_posix(),
                "catalog_root": (working / "object_catalog").as_posix(),
                "cleanup_request_folder": True,
            }
        )
    )
    catalog_uuid = "11111111-1111-4111-8111-111111111111"
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_object_catalog_import.py", "--request", request_path.as_posix()],
    )
    monkeypatch.setattr(
        import_script,
        "import_catalog_object",
        lambda **_kwargs: {"catalog_uuid": catalog_uuid},
    )

    if analysis_fails:
        def analyze(*_args, **_kwargs):
            raise ValueError("no printable stable pose")
    else:
        def analyze(*_args, **_kwargs):
            return {
                "orientations": [{"orientation_id": "orientation-1"}],
                "source": {"canonical_ply_sha256": "1" * 64},
            }
    monkeypatch.setattr(import_script, "analyze_catalog_orientations", analyze)

    import_script.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["catalog_uuid"] == catalog_uuid
    assert payload["orientation_analysis"]["status"] == (
        "unavailable" if analysis_fails else "ready"
    )
    assert not request_folder.exists()


def test_managed_import_removes_large_staging_folder_when_import_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_script = load_import_script()
    working = tmp_path / "working_data"
    request_folder = (
        working
        / "jobs"
        / "workpiece_catalog_requests"
        / "catalog_upload"
        / ("b" * 32)
    )
    request_folder.mkdir(parents=True)
    cad_path = request_folder / "fixture.stl"
    cad_path.write_bytes(b"invalid cad")
    request_path = request_folder / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "name": "Broken fixture",
                "cad_path": cad_path.as_posix(),
                "catalog_root": (working / "object_catalog").as_posix(),
                "cleanup_request_folder": True,
            }
        )
    )
    monkeypatch.setenv("POSETESTBOT_WORKING_DATA_ROOT", working.as_posix())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_object_catalog_import.py", "--request", request_path.as_posix()],
    )

    def fail_import(**_kwargs):
        raise ValueError("invalid CAD")

    monkeypatch.setattr(import_script, "import_catalog_object", fail_import)

    with pytest.raises(ValueError, match="invalid CAD"):
        import_script.main()

    assert not request_folder.exists()
