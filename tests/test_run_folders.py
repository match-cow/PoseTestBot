from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import posetestbot.run_folders as run_folders_module
import posetestbot.web.routes.run_folders as run_folders_routes_module
import posetestbot.web.routes.ui as ui_routes_module
from posetestbot.jobs.runner import JobRecord
from posetestbot.pipeline.run_config import (
    SensorRunConfig,
    create_run_config,
    load_run_config_for_run_root,
    write_run_config,
)
from posetestbot.run_folders import (
    LOCATION_FILE,
    build_run_folder_inventory,
    delete_run_folder,
    move_run_folder,
    resolve_direct_run_folder,
    run_identity,
    write_run_folder_inventory,
)
from posetestbot.web.app import create_app


def _write_run(path: Path, *, with_object: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = create_run_config(
        run_root=path,
        sensors=(
            SensorRunConfig(
                sensor_type="realsense_d435",
                device_id="123",
                display_name="Wrist D435",
                mounting_mode="eye_in_hand",
            ),
        ),
        sequence_id="sync_aruco",
    )
    write_run_config(path, config)
    if with_object:
        (path / "object_instances.json").write_text(
            json.dumps(
                {
                    "schema_version": "object_instances.v1",
                    "template_uuid": "template-1",
                    "instances": [
                        {"name": "Clamp"},
                        {"name": "Clamp"},
                        {"name": "Bracket"},
                    ],
                }
            )
        )


def test_inventory_sizes_and_summarizes_without_following_symlinks(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    run = storage / "run-a"
    outside = tmp_path / "outside"
    _write_run(run, with_object=True)
    outside.mkdir()
    (outside / "large.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    (run / "outside-link").symlink_to(outside, target_is_directory=True)
    raw = run / "realsense_123"
    (raw / "rgb").mkdir(parents=True)
    (raw / "rgb" / "000001.png").write_bytes(b"rgb")
    synchronized = run / "processed" / "synchronized" / "realsense_123"
    synchronized.mkdir(parents=True)
    (synchronized / "sync_report.json").write_text("{}")
    bop = run / "bop"
    bop.mkdir()
    (bop / "bop_export_manifest.json").write_text("{}")

    value = build_run_folder_inventory([storage])

    assert value["schema_version"] == "run_folder_inventory.v1"
    assert len(value["runs"]) == 1
    record = value["runs"][0]
    assert record["path"] == run.as_posix()
    assert record["size_bytes"] < 2 * 1024 * 1024
    assert record["symlink_count"] == 1
    assert record["scan_complete"] is True
    assert record["config"] == {
        "valid": True,
        "error": None,
        "run_name": "run-a",
        "sequence": "sync_aruco",
        "plan_only": True,
    }
    assert record["contents"]["sensor_count"] == 1
    assert record["contents"]["enabled_sensor_count"] == 1
    assert record["contents"]["sensors"][0]["name"] == "Wrist D435"
    assert record["contents"]["object_count"] == 3
    assert record["contents"]["object_names"] == ["Clamp", "Bracket"]
    assert record["contents"]["template_uuid"] == "template-1"
    assert record["contents"]["evidence"]["raw_capture"] is True
    assert record["contents"]["evidence"]["synchronized"] is True
    assert record["contents"]["evidence"]["bop_export"] is True
    assert set(record["breakdown"]) >= {"raw_capture", "processed", "bop", "other"}


def test_general_run_discovery_hides_move_quarantine_and_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "storage"
    visible = storage / "visible-run"
    hidden = storage / ".posetestbot_run_move_staging_transaction"
    _write_run(visible)
    _write_run(hidden)
    monkeypatch.setattr(
        ui_routes_module,
        "web_run_roots",
        lambda: (storage.resolve(),),
    )

    records = ui_routes_module.discover_web_runs()

    assert [item["path"] for item in records] == [visible.as_posix()]


def test_move_preserves_path_bound_config_via_alias_and_supports_move_back(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    first_identity = run_identity(source)

    moved = move_run_folder(
        source,
        second_root,
        expected_identity=first_identity,
        allowed_roots=[first_root, second_root],
    )
    destination = second_root / "run-a"

    assert moved["destination_run_root"] == destination.as_posix()
    assert source.is_symlink()
    assert source.resolve() == destination
    assert destination.is_dir() and not destination.is_symlink()
    assert load_run_config_for_run_root(destination)["run_name"] == "run-a"
    location = json.loads((destination / LOCATION_FILE).read_text())
    assert location["original_path"] == source.as_posix()
    assert location["aliases"] == [source.as_posix()]
    assert len(location["history"]) == 1

    moved_back = move_run_folder(
        destination,
        first_root,
        expected_identity=run_identity(destination),
        allowed_roots=[first_root, second_root],
    )

    assert moved_back["destination_run_root"] == source.as_posix()
    assert source.is_dir() and not source.is_symlink()
    assert destination.is_symlink()
    assert destination.resolve() == source
    assert load_run_config_for_run_root(source)["run_name"] == "run-a"
    location = json.loads((source / LOCATION_FILE).read_text())
    assert location["original_path"] == source.as_posix()
    assert location["aliases"] == [destination.as_posix()]
    assert len(location["history"]) == 2


def test_move_retargets_all_recorded_aliases_across_three_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    third_root = tmp_path / "third"
    first_root.mkdir()
    second_root.mkdir()
    third_root.mkdir()
    first = first_root / "run-a"
    _write_run(first)

    move_run_folder(
        first,
        second_root,
        expected_identity=run_identity(first),
        allowed_roots=[first_root, second_root, third_root],
    )
    second = second_root / first.name
    move_run_folder(
        second,
        third_root,
        expected_identity=run_identity(second),
        allowed_roots=[first_root, second_root, third_root],
    )
    third = third_root / first.name

    assert third.is_dir() and not third.is_symlink()
    assert first.is_symlink() and first.resolve() == third
    assert second.is_symlink() and second.resolve() == third
    location = json.loads((third / LOCATION_FILE).read_text())
    assert location["aliases"] == sorted([first.as_posix(), second.as_posix()])
    assert len(location["history"]) == 2


def test_cross_device_move_back_keeps_destination_alias_until_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "run-a"
    _write_run(first)
    move_run_folder(
        first,
        second_root,
        expected_identity=run_identity(first),
        allowed_roots=[first_root, second_root],
    )
    second = second_root / first.name

    real_copy = run_folders_module._copy_tree_with_content_hash

    def verify_alias_during_copy(source: Path, destination: Path) -> str:
        assert first.is_symlink()
        assert os.readlink(first) == second.as_posix()
        return real_copy(source, destination)

    monkeypatch.setattr(
        run_folders_module,
        "_same_filesystem_mount",
        lambda _source, _destination: False,
    )
    monkeypatch.setattr(
        run_folders_module,
        "_copy_tree_with_content_hash",
        verify_alias_during_copy,
    )
    move_run_folder(
        second,
        first_root,
        expected_identity=run_identity(second),
        allowed_roots=[first_root, second_root],
    )

    assert first.is_dir() and not first.is_symlink()
    assert second.is_symlink() and second.resolve() == first


def test_same_device_different_mount_ids_select_verified_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    copied = False
    real_copy = run_folders_module._copy_tree_with_content_hash

    def mount_id(path: Path) -> int:
        return 1 if path == first_root or path.parent == first_root else 2

    def observe_copy(source_path: Path, destination_path: Path) -> str:
        nonlocal copied
        copied = True
        return real_copy(source_path, destination_path)

    monkeypatch.setattr(run_folders_module.sys, "platform", "linux")
    monkeypatch.setattr(run_folders_module, "_containing_mount_id", mount_id)
    monkeypatch.setattr(
        run_folders_module,
        "_copy_tree_with_content_hash",
        observe_copy,
    )

    move_run_folder(
        source,
        second_root,
        expected_identity=run_identity(source),
        allowed_roots=[first_root, second_root],
    )

    assert copied is True
    assert (second_root / source.name).is_dir()
    assert source.is_symlink()


def test_interrupted_cross_device_copy_rolls_back_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    destination = second_root / source.name
    def interrupt_copy(_source: Path, staging: Path) -> str:
        (staging / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt("simulated runner shutdown")

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_same_filesystem_mount",
            lambda _source, _destination: False,
        )
        patch.setattr(
            run_folders_module,
            "_copy_tree_with_content_hash",
            interrupt_copy,
        )
        with pytest.raises(KeyboardInterrupt, match="runner shutdown"):
            move_run_folder(
                source,
                second_root,
                expected_identity=run_identity(source),
                allowed_roots=[first_root, second_root],
            )

    assert not source.exists()
    assert not destination.exists()
    assert next(first_root.glob(".posetestbot_run_folder_transaction_*.json"))

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    assert source.is_dir() and not source.is_symlink()
    assert not destination.exists()
    assert inventory["maintenance"]["transactions"][0]["action"] == (
        "rolled_back_move"
    )
    assert not list(first_root.glob(".posetestbot_run_folder_transaction_*.json"))
    assert not list(second_root.glob(".posetestbot_run_move_staging_*"))
    second_inventory = write_run_folder_inventory(
        tmp_path / "inventory-second.json",
        allowed_roots=[first_root, second_root],
    )
    assert second_inventory["maintenance"]["recovered_count"] == 0
    assert second_inventory["maintenance"]["unresolved_count"] == 0


def test_interrupted_committed_move_finishes_cleanup_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    destination = second_root / source.name
    real_remove = run_folders_module._remove_validated_tree

    def interrupt_cleanup(
        path: Path,
        *,
        expected_identity: dict[str, int],
    ) -> None:
        if path.parent == first_root and path.name.startswith(
            ".posetestbot_run_move_source_"
        ):
            raise KeyboardInterrupt("simulated shutdown after commit")
        real_remove(path, expected_identity=expected_identity)

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_same_filesystem_mount",
            lambda _source, _destination: False,
        )
        patch.setattr(
            run_folders_module,
            "_remove_validated_tree",
            interrupt_cleanup,
        )
        with pytest.raises(KeyboardInterrupt, match="after commit"):
            move_run_folder(
                source,
                second_root,
                expected_identity=run_identity(source),
                allowed_roots=[first_root, second_root],
            )

    assert source.is_symlink() and source.resolve() == destination
    assert destination.is_dir() and not destination.is_symlink()
    assert next(first_root.glob(".posetestbot_run_folder_transaction_*.json"))

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    assert inventory["maintenance"]["transactions"][0]["action"] == (
        "completed_move"
    )
    assert source.is_symlink() and source.resolve() == destination
    assert destination.is_dir() and not destination.is_symlink()
    assert not list(first_root.glob(".posetestbot_run_move_source_*"))
    assert not list(first_root.glob(".posetestbot_run_folder_transaction_*.json"))


def test_precommit_move_recovery_restores_prior_location_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    third_root = tmp_path / "third"
    first_root.mkdir()
    second_root.mkdir()
    third_root.mkdir()
    first = first_root / "run-a"
    _write_run(first)
    move_run_folder(
        first,
        second_root,
        expected_identity=run_identity(first),
        allowed_roots=[first_root, second_root, third_root],
    )
    second = second_root / first.name
    third = third_root / first.name
    prior_location = json.loads((second / LOCATION_FILE).read_text())
    real_update = run_folders_module._update_transaction

    def interrupt_before_commit(
        path: Path,
        value: dict,
        *,
        phase: str,
        **updates,
    ) -> None:
        if phase == "committed":
            raise KeyboardInterrupt("simulated shutdown before commit journal")
        real_update(path, value, phase=phase, **updates)

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_update_transaction",
            interrupt_before_commit,
        )
        with pytest.raises(KeyboardInterrupt, match="commit journal"):
            move_run_folder(
                second,
                third_root,
                expected_identity=run_identity(second),
                allowed_roots=[first_root, second_root, third_root],
            )

    assert not third.exists()
    assert not second.exists()
    assert first.is_symlink()
    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root, third_root],
    )

    assert inventory["maintenance"]["transactions"][0]["action"] == (
        "rolled_back_move"
    )
    assert second.is_dir() and not second.is_symlink()
    assert not third.exists()
    assert first.is_symlink() and first.resolve() == second
    assert json.loads((second / LOCATION_FILE).read_text()) == prior_location

    retried = move_run_folder(
        second,
        third_root,
        expected_identity=run_identity(second),
        allowed_roots=[first_root, second_root, third_root],
    )
    assert retried["status"] == "moved"
    assert third.is_dir() and second.is_symlink()


def test_move_commits_hidden_candidate_then_interruption_rolls_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    destination = second_root / source.name
    _write_run(source)
    real_verified = run_folders_module._verified_tree_evidence
    real_update = run_folders_module._update_transaction
    hidden_verified = False

    def verify_while_hidden(path: Path):
        nonlocal hidden_verified
        if (
            path.parent == second_root
            and path.name.startswith(".posetestbot_run_move_staging_")
            and (path / LOCATION_FILE).is_file()
        ):
            hidden_verified = True
            assert not os.path.lexists(source)
            assert not os.path.lexists(destination)
            with pytest.raises(FileNotFoundError):
                (source / "concurrent-source-write.txt").write_text("write")
            with pytest.raises(FileNotFoundError):
                (destination / "concurrent-destination-write.txt").write_text(
                    "write"
                )
        return real_verified(path)

    def interrupt_after_public_verification(
        path: Path,
        value: dict,
        *,
        phase: str,
        **updates,
    ) -> None:
        if phase == "destination_verified":
            raise KeyboardInterrupt("simulated post-commit interruption")
        real_update(path, value, phase=phase, **updates)

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_same_filesystem_mount",
            lambda _source, _destination: False,
        )
        patch.setattr(
            run_folders_module,
            "_verified_tree_evidence",
            verify_while_hidden,
        )
        patch.setattr(
            run_folders_module,
            "_update_transaction",
            interrupt_after_public_verification,
        )
        with pytest.raises(KeyboardInterrupt, match="post-commit"):
            move_run_folder(
                source,
                second_root,
                expected_identity=run_identity(source),
                allowed_roots=[first_root, second_root],
            )

    assert hidden_verified is True
    assert destination.is_dir() and not destination.is_symlink()
    assert not os.path.lexists(source)
    journal = next(first_root.glob(".posetestbot_run_folder_transaction_*.json"))
    quarantine = next(first_root.glob(".posetestbot_run_move_source_*"))
    assert quarantine.is_dir()

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    assert inventory["maintenance"]["transactions"][0]["action"] == (
        "completed_move"
    )
    assert source.is_symlink() and source.resolve() == destination
    assert destination.is_dir()
    assert not journal.exists()
    assert not quarantine.exists()


def test_copy_fsyncs_files_and_directories_after_metadata_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"payload")
    destination.mkdir()
    events: list[tuple[str, str]] = []
    real_copystat = run_folders_module.shutil.copystat
    real_fsync = run_folders_module.os.fsync

    def record_copystat(
        source_path: str | Path,
        destination_path: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        real_copystat(
            source_path,
            destination_path,
            follow_symlinks=follow_symlinks,
        )
        events.append(("copystat", Path(destination_path).as_posix()))

    def record_fsync(descriptor: int) -> None:
        try:
            resolved = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            resolved = Path("<unknown>")
        events.append(("fsync", resolved.as_posix()))
        real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(run_folders_module.shutil, "copystat", record_copystat)
        patch.setattr(run_folders_module.os, "fsync", record_fsync)
        run_folders_module._copy_tree_with_content_hash(source, destination)

    copied_paths = (
        destination / "nested" / "payload.bin",
        destination / "nested",
        destination,
    )
    for copied in copied_paths:
        copystat_index = events.index(("copystat", copied.as_posix()))
        fsync_indices = [
            index
            for index, event in enumerate(events)
            if event == ("fsync", copied.as_posix())
        ]
        assert fsync_indices
        assert max(fsync_indices) > copystat_index


def test_committed_recovery_preserves_quarantine_if_destination_bytes_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    (source / "payload.bin").write_bytes(b"good")
    destination = second_root / source.name
    real_update = run_folders_module._update_transaction

    def interrupt_before_destination_verdict(
        path: Path,
        value: dict,
        *,
        phase: str,
        **updates,
    ) -> None:
        if phase == "destination_verified":
            raise KeyboardInterrupt("simulated shutdown before verdict")
        real_update(path, value, phase=phase, **updates)

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_same_filesystem_mount",
            lambda _source, _destination: False,
        )
        patch.setattr(
            run_folders_module,
            "_update_transaction",
            interrupt_before_destination_verdict,
        )
        with pytest.raises(KeyboardInterrupt, match="before verdict"):
            move_run_folder(
                source,
                second_root,
                expected_identity=run_identity(source),
                allowed_roots=[first_root, second_root],
            )

    quarantine = next(first_root.glob(".posetestbot_run_move_source_*"))
    journal = next(first_root.glob(".posetestbot_run_folder_transaction_*.json"))
    (destination / "payload.bin").write_bytes(b"evil")

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    maintenance = inventory["maintenance"]
    assert maintenance["recovered_count"] == 0
    assert maintenance["unresolved_count"] == 1
    assert "content no longer matches" in maintenance["unresolved"][0]["error"]
    assert maintenance["unresolved"][0]["remnant_bytes"] > 0
    assert journal.is_file()
    assert quarantine.is_dir()
    assert destination.is_dir()
    assert (destination / "payload.bin").read_bytes() == b"evil"


def test_recovery_refuses_destination_root_replacement_and_retains_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    destination = second_root / source.name

    def interrupt_copy(_source: Path, staging: Path) -> str:
        (staging / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt("simulated destination outage")

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module,
            "_same_filesystem_mount",
            lambda _source, _destination: False,
        )
        patch.setattr(
            run_folders_module,
            "_copy_tree_with_content_hash",
            interrupt_copy,
        )
        with pytest.raises(KeyboardInterrupt, match="destination outage"):
            move_run_folder(
                source,
                second_root,
                expected_identity=run_identity(source),
                allowed_roots=[first_root, second_root],
            )

    hidden_root = tmp_path / "second-offline"
    second_root.rename(hidden_root)
    second_root.mkdir()
    quarantine = next(first_root.glob(".posetestbot_run_move_source_*"))
    journal = next(first_root.glob(".posetestbot_run_folder_transaction_*.json"))

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    maintenance = inventory["maintenance"]
    assert maintenance["recovered_count"] == 0
    assert maintenance["unresolved_count"] == 1
    assert "identity changed" in maintenance["unresolved"][0]["error"]
    assert journal.is_file()
    assert quarantine.is_dir()
    assert not source.exists()
    hidden_staging = next(hidden_root.glob(".posetestbot_run_move_staging_*"))
    assert (hidden_staging / "partial.bin").read_bytes() == b"partial"
    assert not destination.exists()
    assert not list(second_root.glob(".posetestbot_run_move_staging_*"))


def test_delete_removes_run_and_only_verified_compatibility_aliases(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    move_run_folder(
        source,
        second_root,
        expected_identity=run_identity(source),
        allowed_roots=[first_root, second_root],
    )
    destination = second_root / "run-a"
    unrelated = first_root / "unrelated"
    unrelated.symlink_to(tmp_path)
    unrecorded = second_root / "unrecorded-run-alias"
    unrecorded.symlink_to(destination, target_is_directory=True)

    result = delete_run_folder(
        destination,
        expected_identity=run_identity(destination),
        allowed_roots=[first_root, second_root],
    )

    assert result["status"] == "deleted"
    assert not destination.exists()
    assert not source.exists() and not source.is_symlink()
    assert unrelated.is_symlink()
    assert unrecorded.is_symlink()


def test_delete_fails_closed_at_nested_filesystem_boundary_before_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    source = storage / "run-a"
    _write_run(source)
    foreign = source / "foreign-mount"
    foreign.mkdir()
    (foreign / "evidence.bin").write_bytes(b"preserve")
    identity = run_identity(source)
    real_lstat = Path.lstat

    def foreign_device(path: Path, *args, **kwargs):
        metadata = real_lstat(path, *args, **kwargs)
        if path.name != foreign.name:
            return metadata
        values = list(metadata)
        values[2] = int(metadata.st_dev) + 1
        return os.stat_result(values)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", foreign_device)
        with pytest.raises(ValueError, match="filesystem boundary"):
            delete_run_folder(
                source,
                expected_identity=identity,
                allowed_roots=[storage],
            )

    assert source.is_dir() and not source.is_symlink()
    assert (foreign / "evidence.bin").read_bytes() == b"preserve"
    assert not list(storage.glob(".posetestbot_run_folder_transaction_*.json"))
    assert not list(storage.glob(".posetestbot_run_move_source_*"))


def test_delete_fails_closed_for_same_device_nested_bind_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    source = storage / "run-a"
    _write_run(source)
    bind_mount = source / "same-device-bind"
    bind_mount.mkdir()

    monkeypatch.setattr(
        run_folders_module,
        "_nested_mount_points",
        lambda path: (bind_mount,) if path == source else (),
    )

    with pytest.raises(ValueError, match="nested mountpoint"):
        delete_run_folder(
            source,
            expected_identity=run_identity(source),
            allowed_roots=[storage],
        )

    assert source.is_dir() and bind_mount.is_dir()
    assert not list(storage.glob(".posetestbot_run_folder_transaction_*.json"))
    assert not list(storage.glob(".posetestbot_run_move_source_*"))


def test_invalid_journal_is_reported_without_inferring_hidden_trees(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    orphan = first_root / ".posetestbot_run_move_source_orphan"
    orphan.mkdir()
    (orphan / "evidence.bin").write_bytes(b"do not infer")
    journal = (
        first_root
        / ".posetestbot_run_folder_transaction_00000000000000000000000000000000.json"
    )
    journal.write_text("{}")

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    maintenance = inventory["maintenance"]
    assert maintenance["recovered_count"] == 0
    assert maintenance["unresolved_count"] == 1
    assert maintenance["unresolved"][0]["transaction_id"] == "0" * 32
    assert maintenance["unresolved"][0]["remnant_bytes"] is None
    assert (orphan / "evidence.bin").read_bytes() == b"do not infer"
    with pytest.raises(RuntimeError, match="Unresolved run-folder storage"):
        move_run_folder(
            source,
            second_root,
            expected_identity=run_identity(source),
            allowed_roots=[first_root, second_root],
        )
    assert source.is_dir() and not source.is_symlink()
    assert journal.is_file()


def test_interrupted_confirmed_delete_resumes_during_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    move_run_folder(
        source,
        second_root,
        expected_identity=run_identity(source),
        allowed_roots=[first_root, second_root],
    )
    destination = second_root / "run-a"

    def interrupt_partial_delete(path: Path) -> None:
        assert path.parent == second_root
        assert path.name.startswith(".posetestbot_run_move_source_")
        (path / "run_config.json").unlink()
        raise KeyboardInterrupt("simulated shutdown during recursive delete")

    with monkeypatch.context() as patch:
        patch.setattr(
            run_folders_module.shutil,
            "rmtree",
            interrupt_partial_delete,
        )
        with pytest.raises(KeyboardInterrupt, match="recursive delete"):
            delete_run_folder(
                destination,
                expected_identity=run_identity(destination),
                allowed_roots=[first_root, second_root],
            )

    assert not destination.exists()
    assert source.is_symlink()
    journal = next(second_root.glob(".posetestbot_run_folder_transaction_*.json"))
    assert journal.is_file()
    quarantine = next(second_root.glob(".posetestbot_run_move_source_*"))
    assert quarantine.is_dir()
    assert not (quarantine / "run_config.json").exists()
    assert not destination.exists()

    inventory = write_run_folder_inventory(
        tmp_path / "inventory.json",
        allowed_roots=[first_root, second_root],
    )

    assert inventory["maintenance"]["recovered_count"] == 1
    assert inventory["maintenance"]["transactions"][0]["action"] == "resumed_delete"
    assert not destination.exists()
    assert not source.exists() and not source.is_symlink()
    assert not journal.exists()
    second_inventory = write_run_folder_inventory(
        tmp_path / "inventory-second.json",
        allowed_roots=[first_root, second_root],
    )
    assert second_inventory["maintenance"]["recovered_count"] == 0
    assert second_inventory["maintenance"]["unresolved_count"] == 0


def test_operations_reject_nested_symlink_collision_and_stale_identity(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    nested = first_root / "nested" / "run"
    _write_run(nested)
    alias = first_root / "run-link"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="direct child"):
        resolve_direct_run_folder(
            nested, allowed_roots=[first_root, second_root]
        )
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_direct_run_folder(alias, allowed_roots=[first_root, second_root])
    with pytest.raises(RuntimeError, match="identity changed"):
        move_run_folder(
            source,
            second_root,
            expected_identity={"device": 0, "inode": 1},
            allowed_roots=[first_root, second_root],
        )

    collision = second_root / source.name
    _write_run(collision)
    with pytest.raises(FileExistsError, match="already exists"):
        move_run_folder(
            source,
            second_root,
            expected_identity=run_identity(source),
            allowed_roots=[first_root, second_root],
        )

    collision.replace(second_root / "retired-collision")
    collision.symlink_to(source, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        move_run_folder(
            source,
            second_root,
            expected_identity=run_identity(source),
            allowed_roots=[first_root, second_root],
        )


class _FakeRunner:
    def __init__(self, job_root: Path):
        self.job_root = job_root
        self.job_root.mkdir(parents=True)
        self.submissions: list[dict] = []
        self.jobs: list[JobRecord] = []

    def list(self, *, include_services: bool = True):
        return list(self.jobs)

    def submit(self, **values):
        self.submissions.append(values)
        job = JobRecord(
            id=f"job-{len(self.submissions)}",
            name=values["name"],
            command=list(values["command"]),
            cwd=Path(values["cwd"]).as_posix(),
            status="queued",
            created_at=f"2026-07-29T00:00:0{len(self.submissions)}+00:00",
            log_path=(self.job_root / f"job-{len(self.submissions)}.log").as_posix(),
            resources=sorted(values.get("resources", [])),
            parameters=dict(values.get("parameters", {})),
            scope_kind=values["scope_kind"],
            run_root=(
                Path(values["run_root"]).resolve().as_posix()
                if values.get("run_root") is not None
                else None
            ),
        )
        self.jobs.append(job)
        return job


def test_cached_unresolved_journal_fingerprint_does_not_force_refresh_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    invalid_journal = (
        first_root
        / ".posetestbot_run_folder_transaction_"
        "00000000000000000000000000000000.json"
    )
    invalid_journal.write_text("{}")
    test_roots = (first_root.resolve(), second_root.resolve())
    monkeypatch.setattr(
        run_folders_routes_module,
        "web_run_roots",
        lambda: test_roots,
    )
    runner = _FakeRunner(tmp_path / "jobs")
    cached = write_run_folder_inventory(
        runner.job_root / "run_folder_inventory.json",
        allowed_roots=test_roots,
    )
    assert cached["maintenance"]["unresolved_count"] == 1
    client = create_app(job_runner=runner).test_client()

    first = client.get("/ui/run-folders").get_json()
    second = client.get("/ui/run-folders").get_json()

    assert first["inventory_state"] == "ready"
    assert second["inventory_state"] == "ready"
    assert first["maintenance"]["journal_fingerprint"] == (
        second["maintenance"]["journal_fingerprint"]
    )
    assert runner.submissions == []


def test_cached_destination_root_replacement_is_stale_and_cannot_be_blessed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    test_roots = (first_root.resolve(), second_root.resolve())
    monkeypatch.setenv(
        "POSETESTBOT_WEB_RUN_ROOTS",
        f"{first_root}{os.pathsep}{second_root}",
    )
    monkeypatch.setattr(
        run_folders_routes_module,
        "web_run_roots",
        lambda: test_roots,
    )
    runner = _FakeRunner(tmp_path / "jobs")
    write_run_folder_inventory(
        runner.job_root / "run_folder_inventory.json",
        allowed_roots=test_roots,
    )
    client = create_app(job_runner=runner).test_client()
    original = client.get("/ui/run-folders").get_json()
    root_record = next(
        item
        for item in original["roots"]
        if item["path"] == second_root.as_posix()
    )
    expected_destination_identity = root_record["identity"]

    replaced = tmp_path / "second-replaced"
    second_root.rename(replaced)
    second_root.mkdir()

    stale = client.get("/ui/run-folders").get_json()
    stale_root = next(
        item
        for item in stale["roots"]
        if item["path"] == second_root.as_posix()
    )
    assert stale["inventory_state"] == "stale"
    assert stale_root["identity"] is None
    assert expected_destination_identity != run_identity(second_root)

    blocked = client.post(
        "/ui/run-folders/move",
        json={
            "run_root": source.as_posix(),
            "destination_root": second_root.as_posix(),
            "expected_identity": run_identity(source),
            "expected_destination_root_identity": (
                expected_destination_identity
            ),
        },
    )
    assert blocked.status_code == 409, blocked.get_json()
    assert "identity changed" in blocked.get_json()["output"]
    blocked_live_identity = client.post(
        "/ui/run-folders/move",
        json={
            "run_root": source.as_posix(),
            "destination_root": second_root.as_posix(),
            "expected_identity": run_identity(source),
            "expected_destination_root_identity": run_identity(second_root),
        },
    )
    assert blocked_live_identity.status_code == 409
    assert "refresh inventory" in blocked_live_identity.get_json()["output"]
    assert runner.submissions == []


def test_api_returns_cached_inventory_and_queues_scoped_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    monkeypatch.setenv(
        "POSETESTBOT_WEB_RUN_ROOTS",
        f"{first_root}{os.pathsep}{second_root}",
    )
    test_roots = (first_root.resolve(), second_root.resolve())
    monkeypatch.setattr(
        run_folders_routes_module,
        "web_run_roots",
        lambda: test_roots,
    )
    runner = _FakeRunner(tmp_path / "jobs")
    write_run_folder_inventory(
        runner.job_root / "run_folder_inventory.json",
        allowed_roots=test_roots,
    )
    client = create_app(job_runner=runner).test_client()

    inventory = client.get("/ui/run-folders")
    assert inventory.status_code == 200
    payload = inventory.get_json()
    assert payload["schema_version"] == "run_folder_inventory.v1"
    assert payload["inventory_state"] == "ready"
    assert payload["operation_job"] is None
    assert any(item["path"] == source.as_posix() for item in payload["runs"])
    root_records = {item["path"]: item for item in payload["roots"]}
    assert root_records[second_root.as_posix()]["identity"] == run_identity(
        second_root
    )

    cache_path = runner.job_root / "run_folder_inventory.json"
    malformed = json.loads(cache_path.read_text())
    malformed["runs"][0]["identity"]["device"] = "not-an-integer"
    cache_path.write_text(json.dumps(malformed))
    assert client.get("/ui/run-folders").get_json()["inventory_state"] == "stale"
    write_run_folder_inventory(cache_path, allowed_roots=test_roots)

    run_folders_module._new_transaction(
        operation="move",
        source=source,
        expected_identity=run_identity(source),
        aliases=[],
        destination_root=second_root,
    )
    assert client.get("/ui/run-folders").get_json()["inventory_state"] == "stale"
    recovered_cache = write_run_folder_inventory(
        cache_path,
        allowed_roots=test_roots,
    )
    assert recovered_cache["maintenance"]["transactions"][0]["action"] == (
        "rolled_back_move"
    )

    refresh = client.post("/ui/run-folders/refresh")
    assert refresh.status_code == 202
    assert runner.submissions[-1]["scope_kind"] == "global"
    assert runner.submissions[-1]["resources"] == [
        "disk_io",
        "run_folder_storage",
    ]
    assert runner.submissions[-1]["parameters"]["cancelable"] is False

    runner.jobs.clear()
    identity = run_identity(source)
    missing_destination_identity = client.post(
        "/ui/run-folders/move",
        json={
            "run_root": source.as_posix(),
            "destination_root": second_root.as_posix(),
            "expected_identity": identity,
        },
    )
    assert missing_destination_identity.status_code == 400
    move = client.post(
        "/ui/run-folders/move",
        json={
            "run_root": source.as_posix(),
            "destination_root": second_root.as_posix(),
            "expected_identity": identity,
            "expected_destination_root_identity": run_identity(second_root),
        },
    )
    assert move.status_code == 202
    moved = move.get_json()
    assert moved["source_run_root"] == source.as_posix()
    assert moved["destination_run_root"] == (second_root / source.name).as_posix()
    assert moved["compatibility_alias"] == source.as_posix()
    submission = runner.submissions[-1]
    assert submission["scope_kind"] == "run"
    assert submission["run_root"] == source
    assert submission["resources"] == ["disk_io", "run_folder_storage"]
    assert "--expected-destination-device" in submission["command"]
    assert "--expected-destination-inode" in submission["command"]
    active_operation = client.get("/ui/run-folders").get_json()["operation_job"]
    assert active_operation["id"] == moved["job_id"]
    assert active_operation["parameters"]["run_folder_operation"] == "move"

    runner.jobs.clear()
    refused = client.delete(
        "/ui/run-folders",
        json={
            "run_root": source.as_posix(),
            "confirm": False,
            "expected_identity": identity,
        },
    )
    assert refused.status_code == 400
    refused_string = client.delete(
        "/ui/run-folders",
        json={
            "run_root": source.as_posix(),
            "confirm": "true",
            "expected_identity": identity,
        },
    )
    assert refused_string.status_code == 400
    deleted = client.delete(
        "/ui/run-folders",
        json={
            "run_root": source.as_posix(),
            "confirm": True,
            "expected_identity": identity,
        },
    )
    assert deleted.status_code == 202
    assert runner.submissions[-1]["name"] == "run_folder_delete"
    assert runner.submissions[-1]["resources"] == ["disk_io", "run_folder_storage"]
    assert "--confirm-delete" in runner.submissions[-1]["command"]


def test_api_rejects_active_run_job_and_symlink_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    source = first_root / "run-a"
    _write_run(source)
    alias = first_root / "alias"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.setenv(
        "POSETESTBOT_WEB_RUN_ROOTS",
        f"{first_root}{os.pathsep}{second_root}",
    )
    monkeypatch.setattr(
        run_folders_routes_module,
        "web_run_roots",
        lambda: (first_root.resolve(), second_root.resolve()),
    )
    runner = _FakeRunner(tmp_path / "jobs")
    write_run_folder_inventory(
        runner.job_root / "run_folder_inventory.json",
        allowed_roots=[first_root, second_root],
    )
    active = runner.submit(
        name="active",
        command=[sys.executable, "-c", "pass"],
        cwd=tmp_path,
        resources=[],
        scope_kind="run",
        run_root=source,
        parameters={},
    )
    active.status = "running"
    client = create_app(job_runner=runner).test_client()
    identity = run_identity(source)

    blocked = client.delete(
        "/ui/run-folders",
        json={
            "run_root": source.as_posix(),
            "confirm": True,
            "expected_identity": identity,
        },
    )
    assert blocked.status_code == 409
    assert "active background work" in blocked.get_json()["output"]

    runner.jobs.clear()
    symlinked = client.delete(
        "/ui/run-folders",
        json={
            "run_root": alias.as_posix(),
            "confirm": True,
            "expected_identity": identity,
        },
    )
    assert symlinked.status_code == 400
    assert "symbolic link" in symlinked.get_json()["output"]

    bridge = tmp_path / "bridge"
    bridge.symlink_to(first_root, target_is_directory=True)
    nested_through_link = client.delete(
        "/ui/run-folders",
        json={
            "run_root": (bridge / source.name).as_posix(),
            "confirm": True,
            "expected_identity": identity,
        },
    )
    assert nested_through_link.status_code == 400
    assert "direct child" in nested_through_link.get_json()["output"]
