from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from posetestbot.jobs.runner import (
    CANCELED,
    CANCELING,
    FAILED,
    QUEUED,
    SUCCEEDED,
    JobRecord,
    LocalJobRunner,
    ResourceBusyError,
    SERVICE_VISIBILITY,
)


def test_local_job_runner_captures_successful_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="echo",
        command=[sys.executable, "-c", "print('hello from job')"],
        parameters={"purpose": "unit-test"},
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == SUCCEEDED
    assert finished.returncode == 0
    assert finished.parameters == {"purpose": "unit-test"}
    assert "hello from job" in runner.log_text(job.id)
    assert (tmp_path / "jobs" / job.id / "job.json").is_file()


def test_local_job_runner_records_failed_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="fail",
        command=[sys.executable, "-c", "print('nope'); raise SystemExit(7)"],
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == FAILED
    assert finished.returncode == 7
    assert "nope" in runner.log_text(job.id)


def test_local_job_runner_bounds_large_unbroken_output(tmp_path: Path) -> None:
    runner = LocalJobRunner(
        tmp_path / "jobs",
        max_log_bytes=4096,
        max_tail_line_chars=128,
    )

    job = runner.submit(
        name="large-output",
        command=[
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 100_000)",
        ],
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == SUCCEEDED
    log_path = Path(finished.log_path)
    assert log_path.stat().st_size <= 4096
    assert "job log truncated" in runner.log_text(job.id)
    assert all(len(line) <= 150 for line in finished.tail)
    assert any("line truncated" in line for line in finished.tail)
    assert (log_path.parent / "job.json").stat().st_size < 10_000


def test_local_job_runner_bounds_legacy_persisted_tail(tmp_path: Path) -> None:
    job_root = tmp_path / "jobs"
    job_dir = job_root / "legacy-large-tail"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "legacy-large-tail",
                "name": "legacy",
                "command": [sys.executable, "-c", "pass"],
                "cwd": None,
                "status": SUCCEEDED,
                "created_at": "2026-07-22T00:00:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "tail": ["discarded", "y" * 1000],
            }
        )
    )

    loaded = LocalJobRunner(
        job_root,
        tail_limit=1,
        max_tail_line_chars=128,
    ).get("legacy-large-tail")

    assert len(loaded.tail) == 1
    assert len(loaded.tail[0]) <= 150
    assert loaded.tail[0].endswith("… [line truncated]")


def test_local_job_runner_bounds_total_tail_size(tmp_path: Path) -> None:
    runner = LocalJobRunner(
        tmp_path / "jobs",
        tail_limit=200,
        max_tail_line_chars=128,
        max_tail_chars=512,
    )

    job = runner.submit(
        name="many-lines",
        command=[
            sys.executable,
            "-c",
            "print(('z' * 100 + '\\n') * 100, end='')",
        ],
    )
    finished = runner.wait(job.id, timeout=5)

    assert finished.status == SUCCEEDED
    assert sum(len(line) for line in finished.tail) <= 512
    assert len(finished.tail) < 100


def test_local_job_runner_throttles_tail_metadata_writes_for_output_flood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    original_persist = runner._persist_job
    persist_calls = 0

    def counted_persist(job: JobRecord) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(job)

    monkeypatch.setattr(runner, "_persist_job", counted_persist)
    job = runner.submit(
        name="many-fast-lines",
        command=[
            sys.executable,
            "-c",
            "print(('line\\n') * 20_000, end='')",
        ],
    )
    finished = runner.wait(job.id, timeout=10)

    assert finished.status == SUCCEEDED
    assert persist_calls < 100
    persisted = json.loads((Path(finished.log_path).parent / "job.json").read_text())
    assert persisted["status"] == SUCCEEDED
    assert persisted["tail"][-1] == "Command completed successfully."


def test_local_job_runner_can_cancel_running_command(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
    )
    deadline = time.time() + 5
    while runner.get(job.id).started_at is None and time.time() < deadline:
        time.sleep(0.01)

    canceled = runner.cancel(job.id)
    finished = runner.wait(job.id, timeout=5)

    assert canceled.status in {CANCELING, CANCELED}
    assert finished.status == CANCELED


def test_local_job_runner_cancels_child_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child_survived.txt"
    runner = LocalJobRunner(tmp_path / "jobs")
    script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import pathlib, time; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('alive')\"]); "
        "time.sleep(10)"
    )

    job = runner.submit(name="parent", command=[sys.executable, "-c", script])
    deadline = time.time() + 5
    while runner.get(job.id).started_at is None and time.time() < deadline:
        time.sleep(0.01)

    runner.cancel(job.id)
    finished = runner.wait(job.id, timeout=5)
    time.sleep(1.2)

    assert finished.status == CANCELED
    assert not marker.exists()


def test_local_job_runner_reloads_persisted_history(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="echo",
        command=[sys.executable, "-c", "print('persisted')"],
        resources=["camera"],
    )
    finished = runner.wait(job.id, timeout=5)
    reloaded = LocalJobRunner(tmp_path / "jobs")

    loaded = reloaded.get(finished.id)
    assert loaded.status == SUCCEEDED
    assert loaded.resources == ["camera"]
    assert "persisted" in reloaded.log_text(finished.id)


def test_local_job_runner_marks_interrupted_jobs_failed_on_reload(tmp_path: Path) -> None:
    job_root = tmp_path / "jobs"
    job_dir = job_root / "orphaned"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "orphaned",
                "name": "sleep",
                "command": [sys.executable, "-c", "import time; time.sleep(10)"],
                "cwd": None,
                "status": QUEUED,
                "created_at": "2026-06-16T00:00:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "resources": ["robot"],
                "parameters": {"capture": True},
            }
        )
    )

    reloaded = LocalJobRunner(job_root)
    loaded = reloaded.get("orphaned")

    assert loaded.status == FAILED
    assert loaded.message == "Job runner restarted before this job completed."
    assert loaded.parameters == {"capture": True}


def test_local_job_runner_stops_verified_orphaned_process_group_on_reload(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Process-group recovery uses Linux process metadata")

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    process_start_time = LocalJobRunner._read_process_start_time(process.pid)
    if process_start_time is None:
        process.kill()
        process.wait()
        pytest.skip("Linux /proc process start metadata is unavailable")

    job_root = tmp_path / "jobs"
    job_dir = job_root / "orphaned"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "orphaned",
                "name": "sleep",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": None,
                "status": "running",
                "created_at": "2026-07-10T00:00:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "process_pid": process.pid,
                "process_group_id": os.getpgid(process.pid),
                "process_start_time": process_start_time,
                "runner_pid": 999_999_999,
                "runner_start_time": 1,
            }
        )
    )

    try:
        reloaded = LocalJobRunner(job_root)
        loaded = reloaded.get("orphaned")
        process.wait(timeout=5)

        assert loaded.status == FAILED
        assert "orphaned process group was stopped" in loaded.message
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_local_job_runner_stops_verified_orphan_from_legacy_terminal_job(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Process-group recovery uses Linux process metadata")

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    process_start_time = LocalJobRunner._read_process_start_time(process.pid)
    if process_start_time is None:
        process.kill()
        process.wait()
        pytest.skip("Linux /proc process start metadata is unavailable")

    job_root = tmp_path / "jobs"
    job_dir = job_root / "legacy-terminal-orphan"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "legacy-terminal-orphan",
                "name": "sensor-preview:test",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": None,
                "status": FAILED,
                "created_at": "2026-07-10T00:00:00+00:00",
                "ended_at": "2026-07-10T00:01:00+00:00",
                "log_path": (job_dir / "log.txt").as_posix(),
                "message": "Job runner restarted before this job completed.",
                "process_pid": process.pid,
                "process_group_id": os.getpgid(process.pid),
                "process_start_time": process_start_time,
                "runner_pid": 999_999_999,
                "runner_start_time": 1,
            }
        )
    )

    try:
        reloaded = LocalJobRunner(job_root)
        loaded = reloaded.get("legacy-terminal-orphan")
        process.wait(timeout=5)

        assert loaded.status == FAILED
        assert loaded.message == "Job runner restarted before this job completed."
        assert "verified orphaned process group" in loaded.tail[-1]
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_shutdown_cancels_only_jobs_owned_by_this_runner(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
    )
    deadline = time.time() + 5
    while job.id not in runner._processes and time.time() < deadline:
        time.sleep(0.01)

    runner.shutdown(timeout=5)

    assert runner.wait(job.id, timeout=5).status == CANCELED


def test_local_job_runner_rejects_busy_resources(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")

    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["robot"],
    )
    try:
        with pytest.raises(ResourceBusyError, match="robot held by job"):
            runner.submit(
                name="other",
                command=[sys.executable, "-c", "print('blocked')"],
                resources=["robot"],
            )
        assert runner.resource_holders()["robot"] == job.id
    finally:
        runner.cancel(job.id)


def test_local_job_runner_applies_hierarchical_resource_conflicts(
    tmp_path: Path,
) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    preview = runner.submit(
        name="preview",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["camera:realsense_d435:123"],
    )
    try:
        with pytest.raises(ResourceBusyError, match="camera conflicts with"):
            runner.submit(
                name="capture",
                command=[sys.executable, "-c", "print('blocked')"],
                resources=["camera"],
            )
        other = runner.submit(
            name="other-preview",
            command=[sys.executable, "-c", "print('allowed')"],
            resources=["camera:realsense_d435:456"],
        )
        assert runner.wait(other.id, timeout=5).status == SUCCEEDED
    finally:
        runner.cancel(preview.id)
        runner.wait(preview.id, timeout=5)


def test_canceling_job_retains_resource_until_process_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    job = runner.submit(
        name="sleep",
        command=[sys.executable, "-c", "import time; time.sleep(10)"],
        resources=["camera"],
    )
    deadline = time.time() + 5
    while job.id not in runner._processes and time.time() < deadline:
        time.sleep(0.01)
    process = runner._processes[job.id]
    terminate = runner._terminate_process_group
    monkeypatch.setattr(runner, "_terminate_process_group", lambda _process: None)

    canceled = runner.cancel(job.id)

    assert canceled.status == CANCELING
    assert runner.resource_holders()["camera"] == job.id
    with pytest.raises(ResourceBusyError):
        runner.submit(
            name="blocked",
            command=[sys.executable, "-c", "pass"],
            resources=["camera:realsense_d435:123"],
        )

    terminate(process)
    assert runner.wait(job.id, timeout=5).status == CANCELED


def test_service_visibility_filters_public_jobs_and_resources(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path / "jobs")
    service = runner.submit(
        name="managed-monitor",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        resources=["monitoring_camera:0c45:2283"],
        visibility=SERVICE_VISIBILITY,
    )
    operator = runner.submit(
        name="operator-job",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        resources=["disk_io"],
    )
    try:
        assert [job.id for job in runner.list(include_services=False)] == [operator.id]
        assert runner.resource_holders() == {"disk_io": operator.id}
        assert runner.resource_holders(include_services=True) == {
            "disk_io": operator.id,
            "monitoring_camera:0c45:2283": service.id,
        }
    finally:
        runner.shutdown()


def test_supervisor_stops_workload_descendants_after_owner_sigkill(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("Linux parent-death signaling is required")

    ready_path = tmp_path / "owner_ready.json"
    child_ready = tmp_path / "child_pid.txt"
    survived = tmp_path / "descendant_survived.txt"
    job_root = tmp_path / "jobs"
    descendant = (
        "import pathlib, time; time.sleep(3); "
        f"pathlib.Path({str(survived)!r}).write_text('alive'); time.sleep(30)"
    )
    workload = (
        "import pathlib, subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        f"pathlib.Path({str(child_ready)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    owner = (
        "import json, pathlib, sys, time; "
        "from posetestbot.jobs.runner import LocalJobRunner; "
        f"runner=LocalJobRunner(pathlib.Path({str(job_root)!r})); "
        f"job=runner.submit(name='parent-death', command=[sys.executable, '-c', {workload!r}]); "
        f"child=pathlib.Path({str(child_ready)!r}); "
        "deadline=time.time()+5; "
        "\nwhile (runner.get(job.id).process_pid is None or not child.exists()) and time.time()<deadline: time.sleep(0.02)\n"
        "record=runner.get(job.id); "
        f"pathlib.Path({str(ready_path)!r}).write_text(json.dumps({{'supervisor_pid':record.supervisor_pid,'workload_pid':record.process_pid,'child_pid':int(child.read_text())}})); "
        "time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", owner])
    try:
        deadline = time.time() + 8
        while not ready_path.is_file() and time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"owner exited early with {process.returncode}")
            time.sleep(0.02)
        identities = json.loads(ready_path.read_text())
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

        deadline = time.time() + 7
        while time.time() < deadline:
            if all(
                LocalJobRunner._read_process_start_time(int(pid)) is None
                for pid in identities.values()
            ):
                break
            time.sleep(0.05)
        time.sleep(3.2)

        assert not survived.exists()
        assert all(
            LocalJobRunner._read_process_start_time(int(pid)) is None
            for pid in identities.values()
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for pid in (
            json.loads(ready_path.read_text()).values()
            if ready_path.is_file()
            else []
        ):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
