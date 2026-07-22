"""Small process-backed job runner for local PoseTestBot commands."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from posetestbot.io.manifest import utc_now_iso
from posetestbot.io.atomic import atomic_write_json
from posetestbot.jobs.supervisor import read_process_start_time


QUEUED = "queued"
RUNNING = "running"
CANCELING = "canceling"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELED = "canceled"
TERMINAL_STATUSES = {SUCCEEDED, FAILED, CANCELED}
OPERATOR_VISIBILITY = "operator"
SERVICE_VISIBILITY = "service"
JOB_VISIBILITIES = {OPERATOR_VISIBILITY, SERVICE_VISIBILITY}
DEFAULT_MAX_LOG_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_TAIL_LINE_CHARS = 16 * 1024
DEFAULT_MAX_TAIL_CHARS = 256 * 1024
OUTPUT_READ_CHARS = 64 * 1024
TAIL_PERSIST_INTERVAL_SECONDS = 0.25


@dataclass
class JobRecord:
    id: str
    name: str
    command: list[str]
    cwd: str | None
    status: str
    created_at: str
    log_path: str
    started_at: str | None = None
    ended_at: str | None = None
    returncode: int | None = None
    message: str | None = None
    tail: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    process_pid: int | None = None
    process_group_id: int | None = None
    process_start_time: int | None = None
    runner_pid: int | None = None
    runner_start_time: int | None = None
    supervisor_pid: int | None = None
    supervisor_process_group_id: int | None = None
    supervisor_start_time: int | None = None
    visibility: str = OPERATOR_VISIBILITY

    def to_dict(self) -> dict:
        return asdict(self)


class LocalJobRunner:
    """Run structured command arrays in background threads and keep job logs."""

    def __init__(
        self,
        job_root: str | Path,
        *,
        tail_limit: int = 200,
        max_log_bytes: int = DEFAULT_MAX_LOG_BYTES,
        max_tail_line_chars: int = DEFAULT_MAX_TAIL_LINE_CHARS,
        max_tail_chars: int = DEFAULT_MAX_TAIL_CHARS,
    ):
        if tail_limit < 1:
            raise ValueError("tail_limit must be at least 1")
        if max_log_bytes < 1024:
            raise ValueError("max_log_bytes must be at least 1024")
        if max_tail_line_chars < 64:
            raise ValueError("max_tail_line_chars must be at least 64")
        if max_tail_chars < 64:
            raise ValueError("max_tail_chars must be at least 64")
        self.job_root = Path(job_root)
        self.tail_limit = tail_limit
        self.max_log_bytes = max_log_bytes
        self.max_tail_line_chars = max_tail_line_chars
        self.max_tail_chars = max_tail_chars
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._local_job_ids: set[str] = set()
        self._runner_pid = os.getpid()
        self._runner_start_time = self._read_process_start_time(self._runner_pid)
        self._load_persisted_jobs()

    def submit(
        self,
        *,
        name: str,
        command: list[str],
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        resources: list[str] | None = None,
        parameters: Mapping[str, object] | None = None,
        visibility: str = OPERATOR_VISIBILITY,
    ) -> JobRecord:
        if not command:
            raise ValueError("Job command must not be empty")
        if visibility not in JOB_VISIBILITIES:
            raise ValueError(
                f"visibility must be one of: {', '.join(sorted(JOB_VISIBILITIES))}"
            )

        requested_resources = sorted(set(resources or []))
        with self._lock:
            self._check_resources_available(requested_resources)
            job_id = uuid.uuid4().hex[:12]
            job_dir = self.job_root / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            job = JobRecord(
                id=job_id,
                name=name,
                command=list(command),
                cwd=Path(cwd).as_posix() if cwd is not None else None,
                status=QUEUED,
                created_at=utc_now_iso(),
                log_path=(job_dir / "log.txt").as_posix(),
                resources=requested_resources,
                parameters=dict(parameters or {}),
                runner_pid=self._runner_pid,
                runner_start_time=self._runner_start_time,
                visibility=visibility,
            )
            self._jobs[job_id] = job
            self._local_job_ids.add(job_id)
            self._persist_job(job)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, dict(env or {})),
            name=f"posetestbot-job-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._threads[job_id] = thread
        thread.start()
        return self.get(job_id)

    def resource_holders(self, *, include_services: bool = False) -> dict[str, str]:
        with self._lock:
            return self._resource_holders(include_services=include_services)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return JobRecord(**self._jobs[job_id].to_dict())
            except KeyError as exc:
                raise KeyError(f"Unknown job: {job_id}") from exc

    def list(self, *, include_services: bool = True) -> list[JobRecord]:
        with self._lock:
            return [
                JobRecord(**job.to_dict())
                for job in sorted(
                    (
                        job
                        for job in self._jobs.values()
                        if include_services or job.visibility == OPERATOR_VISIBILITY
                    ),
                    key=lambda item: item.created_at,
                    reverse=True,
                )
            ]

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout=timeout)
        return self.get(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if job.status in TERMINAL_STATUSES:
                return JobRecord(**job.to_dict())
            process = self._processes.get(job_id)
            job.status = CANCELED if job.status == QUEUED else CANCELING
            job.message = "Cancellation requested."
            if job.status == CANCELED:
                job.ended_at = utc_now_iso()
            self._append_tail(job, "Cancellation requested.")
            self._persist_job(job)

        if process is not None and process.poll() is None:
            self._terminate_process_group(process)
            with self._lock:
                job = self._jobs[job_id]
                if job.status == CANCELING and process.poll() is not None:
                    job.status = CANCELED
                    job.ended_at = utc_now_iso()
                    job.returncode = process.returncode
                    job.message = "Canceled."
                    self._persist_job(job)
        return self.get(job_id)

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop all locally owned groups, escalating once the grace period ends."""

        with self._lock:
            active_ids = [
                job.id
                for job in self._jobs.values()
                if job.id in self._local_job_ids
                and job.status not in TERMINAL_STATUSES
            ]
        processes: dict[str, subprocess.Popen] = {}
        with self._lock:
            for job_id in active_ids:
                job = self._jobs[job_id]
                job.status = CANCELED if job.status == QUEUED else CANCELING
                job.message = "Shutdown requested."
                self._append_tail(job, job.message)
                self._persist_job(job)
                process = self._processes.get(job_id)
                if process is not None and process.poll() is None:
                    processes[job_id] = process

        for process in processes.values():
            self._signal_supervisor(process, signal.SIGTERM)

        deadline = time.monotonic() + max(timeout, 0.0)
        for job_id in active_ids:
            with self._lock:
                thread = self._threads.get(job_id)
            if thread is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        for job_id, process in processes.items():
            if process.poll() is None:
                self._signal_supervisor(process, signal.SIGKILL)
                self._terminate_recorded_workload(self.get(job_id), signal.SIGKILL)
        for job_id in active_ids:
            with self._lock:
                thread = self._threads.get(job_id)
            if thread is not None:
                thread.join(timeout=1.0)

    def log_text(self, job_id: str) -> str:
        job = self.get(job_id)
        log_path = Path(job.log_path)
        if not log_path.is_file():
            return ""
        return log_path.read_text()

    def _run_job(self, job_id: str, env: dict[str, str]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status in {CANCELED, CANCELING}:
                if job.status == CANCELING:
                    job.status = CANCELED
                    job.ended_at = utc_now_iso()
                    self._persist_job(job)
                return
            job.status = RUNNING
            job.started_at = utc_now_iso()
            self._persist_job(job)

        with open(job.log_path, "ab", buffering=0) as log:
            log_bytes = log.tell()
            log_truncated = log_bytes >= self.max_log_bytes

            def write_log(value: str) -> None:
                nonlocal log_bytes, log_truncated
                if log_truncated:
                    return
                encoded = value.encode("utf-8", errors="replace")
                marker = (
                    "\n[PoseTestBot job log truncated at "
                    f"{self.max_log_bytes} bytes]\n"
                ).encode("utf-8")
                data_limit = max(0, self.max_log_bytes - len(marker))
                remaining = data_limit - log_bytes
                if len(encoded) <= remaining:
                    log.write(encoded)
                    log_bytes += len(encoded)
                    return
                if remaining > 0:
                    log.write(encoded[:remaining])
                    log_bytes += remaining
                marker_remaining = self.max_log_bytes - log_bytes
                if marker_remaining > 0:
                    log.write(marker[:marker_remaining])
                    log_bytes += min(len(marker), marker_remaining)
                log_truncated = True

            write_log(f"$ {self._format_command(job.command)}\n")
            try:
                identity_path = Path(job.log_path).parent / "supervisor.json"
                supervisor_command = [
                    sys.executable,
                    "-m",
                    "posetestbot.jobs.supervisor",
                    "--owner-pid",
                    str(self._runner_pid),
                    "--owner-start-time",
                    str(self._runner_start_time),
                    "--identity-path",
                    identity_path.as_posix(),
                    "--termination-timeout",
                    "5",
                    "--",
                    *job.command,
                ]
                process = subprocess.Popen(
                    supervisor_command,
                    cwd=job.cwd,
                    env={**os.environ, **env},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=(os.name != "nt"),
                )
            except Exception as exc:
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = FAILED
                    job.ended_at = utc_now_iso()
                    job.message = f"{type(exc).__name__}: {exc}"
                    self._append_tail(job, job.message)
                    self._persist_job(job)
                return

            with self._lock:
                self._processes[job_id] = process
                current = self._jobs[job_id]
                current.supervisor_pid = process.pid
                current.supervisor_process_group_id = (
                    os.getpgid(process.pid) if os.name != "nt" else process.pid
                )
                current.supervisor_start_time = self._read_process_start_time(process.pid)
                self._persist_job(current)
                should_terminate = self._jobs[job_id].status in {
                    CANCELED,
                    CANCELING,
                }

            if should_terminate:
                self._terminate_process_group(process)

            self._refresh_supervisor_identity(job_id, wait_s=2.0)

            assert process.stdout is not None
            pending_tail = ""
            pending_tail_truncated = False
            last_tail_persisted_at = time.monotonic()
            while True:
                fragment = process.stdout.readline(OUTPUT_READ_CHARS)
                if not fragment:
                    break
                write_log(fragment)
                room = self.max_tail_line_chars - len(pending_tail)
                if room > 0:
                    pending_tail += fragment[:room]
                if len(fragment) > room:
                    pending_tail_truncated = True
                if fragment.endswith("\n"):
                    line = pending_tail.rstrip("\r\n")
                    if pending_tail_truncated:
                        line += "… [line truncated]"
                    with self._lock:
                        current = self._jobs[job_id]
                        self._append_tail(current, line)
                        now = time.monotonic()
                        if (
                            now - last_tail_persisted_at
                            >= TAIL_PERSIST_INTERVAL_SECONDS
                        ):
                            self._persist_job(current)
                            last_tail_persisted_at = now
                    pending_tail = ""
                    pending_tail_truncated = False

            if pending_tail or pending_tail_truncated:
                line = pending_tail.rstrip("\r\n")
                if pending_tail_truncated:
                    line += "… [line truncated]"
                with self._lock:
                    current = self._jobs[job_id]
                    self._append_tail(current, line)

            returncode = process.wait()
            self._cleanup_recorded_workload(job_id, timeout_s=1.0)
            with self._lock:
                job = self._jobs[job_id]
                job.returncode = returncode
                job.ended_at = utc_now_iso()
                if job.status in {CANCELED, CANCELING}:
                    job.status = CANCELED
                    job.message = "Canceled."
                elif returncode == 0:
                    job.status = SUCCEEDED
                    job.message = "Command completed successfully."
                else:
                    job.status = FAILED
                    job.message = f"Command exited with status {returncode}."
                self._append_tail(job, job.message)
                self._persist_job(job)
                self._processes.pop(job_id, None)
                self._threads.pop(job_id, None)

    def _append_tail(self, job: JobRecord, line: str) -> None:
        job.tail.append(self._bounded_tail_line(line))
        if len(job.tail) > self.tail_limit:
            del job.tail[: len(job.tail) - self.tail_limit]
        while len(job.tail) > 1 and sum(len(item) for item in job.tail) > self.max_tail_chars:
            del job.tail[0]

    def _bounded_tail_line(self, line: str) -> str:
        limit = min(self.max_tail_line_chars, self.max_tail_chars)
        if len(line) <= limit:
            return line
        suffix = "… [line truncated]"
        return line[: max(0, limit - len(suffix))] + suffix

    def _resource_holders(self, *, include_services: bool = True) -> dict[str, str]:
        holders = {}
        for job in self._jobs.values():
            if job.status in TERMINAL_STATUSES:
                continue
            if not include_services and job.visibility == SERVICE_VISIBILITY:
                continue
            for resource in job.resources:
                holders[resource] = job.id
        return holders

    def _check_resources_available(self, resources: list[str]) -> None:
        holders = self._resource_holders(include_services=True)
        conflicts: dict[str, str] = {}
        for requested in resources:
            for held, job_id in holders.items():
                if self._resources_conflict(requested, held):
                    label = (
                        requested
                        if requested == held
                        else f"{requested} conflicts with {held}"
                    )
                    conflicts[label] = job_id
        if conflicts:
            details = ", ".join(
                f"{resource} held by job {job_id}"
                for resource, job_id in sorted(conflicts.items())
            )
            raise ResourceBusyError(f"Requested resources are busy: {details}")

    @staticmethod
    def _resources_conflict(left: str, right: str) -> bool:
        return (
            left == right
            or left.startswith(f"{right}:")
            or right.startswith(f"{left}:")
        )

    def _terminate_process_group(
        self, process: subprocess.Popen, *, timeout_s: float = 5.0
    ) -> None:
        if process.poll() is not None:
            return

        self._signal_supervisor(process, signal.SIGTERM)

        try:
            process.wait(timeout=timeout_s)
            self._cleanup_workload_for_supervisor(process, timeout_s=1.0)
            return
        except subprocess.TimeoutExpired:
            pass

        self._signal_supervisor(process, signal.SIGKILL)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass
        with self._lock:
            job_id = next(
                (
                    item_id
                    for item_id, item_process in self._processes.items()
                    if item_process is process
                ),
                None,
            )
        if job_id is not None:
            self._cleanup_recorded_workload(job_id, timeout_s=0.0)

    def _cleanup_workload_for_supervisor(
        self,
        process: subprocess.Popen,
        *,
        timeout_s: float,
    ) -> None:
        with self._lock:
            job_id = next(
                (
                    item_id
                    for item_id, item_process in self._processes.items()
                    if item_process is process
                ),
                None,
            )
        if job_id is not None:
            self._cleanup_recorded_workload(job_id, timeout_s=timeout_s)

    def _cleanup_recorded_workload(self, job_id: str, *, timeout_s: float) -> None:
        self._refresh_supervisor_identity(job_id)
        job = self.get(job_id)
        if not self._persisted_process_matches(job):
            return
        self._terminate_recorded_workload(job, signal.SIGTERM)
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while self._persisted_process_matches(job) and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._persisted_process_matches(job):
            self._terminate_recorded_workload(job, signal.SIGKILL)

    @staticmethod
    def _signal_supervisor(process: subprocess.Popen, signum: int) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.terminate() if signum == signal.SIGTERM else process.kill()
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def _refresh_supervisor_identity(self, job_id: str, *, wait_s: float = 0.0) -> None:
        with self._lock:
            job = self._jobs[job_id]
            path = Path(job.log_path).parent / "supervisor.json"
        deadline = time.monotonic() + max(wait_s, 0.0)
        while True:
            try:
                with open(path, encoding="utf-8") as handle:
                    value = json.load(handle)
                workload_pid = value.get("workload_pid")
                if isinstance(workload_pid, int):
                    with self._lock:
                        job = self._jobs[job_id]
                        job.process_pid = workload_pid
                        job.process_group_id = value.get("workload_process_group_id")
                        job.process_start_time = value.get("workload_start_time")
                        self._persist_job(job)
                    return
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def _load_persisted_jobs(self) -> None:
        for path in sorted(self.job_root.glob("*/job.json")):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                job = self._job_from_dict(data)
                persisted_tail = job.tail[-self.tail_limit :]
                job.tail = []
                for line in persisted_tail:
                    self._append_tail(job, str(line))
                self._merge_supervisor_identity(job, path.parent / "supervisor.json")
            except Exception:
                continue

            owner_alive = self._job_owner_is_alive(job)
            orphan_stopped = (
                self._terminate_persisted_process_group(job)
                if not owner_alive
                else False
            )
            if job.status not in TERMINAL_STATUSES:
                if owner_alive:
                    self._jobs[job.id] = job
                    continue
                job.status = FAILED
                job.ended_at = utc_now_iso()
                job.returncode = None
                job.message = "Job runner restarted before this job completed."
                if orphan_stopped:
                    job.message += " Its orphaned process group was stopped."
                self._append_tail(job, job.message)
                self._persist_job(job)
            elif orphan_stopped:
                self._append_tail(
                    job,
                    "A verified orphaned process group left by this terminal job "
                    "was stopped.",
                )
                self._persist_job(job)
            self._jobs[job.id] = job

    @staticmethod
    def _job_from_dict(data: Mapping[str, object]) -> JobRecord:
        job_data = dict(data)
        job_data.setdefault("tail", [])
        job_data.setdefault("resources", [])
        job_data.setdefault("parameters", {})
        job_data.setdefault("process_pid", None)
        job_data.setdefault("process_group_id", None)
        job_data.setdefault("process_start_time", None)
        job_data.setdefault("runner_pid", None)
        job_data.setdefault("runner_start_time", None)
        job_data.setdefault("supervisor_pid", None)
        job_data.setdefault("supervisor_process_group_id", None)
        job_data.setdefault("supervisor_start_time", None)
        job_data.setdefault("visibility", OPERATOR_VISIBILITY)
        return JobRecord(**job_data)

    @staticmethod
    def _read_process_start_time(pid: int) -> int | None:
        """Return Linux process start ticks, used to guard against PID reuse."""

        return read_process_start_time(pid)

    @staticmethod
    def _merge_supervisor_identity(job: JobRecord, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        mappings = {
            "supervisor_pid": "supervisor_pid",
            "supervisor_process_group_id": "supervisor_process_group_id",
            "supervisor_start_time": "supervisor_start_time",
            "process_pid": "workload_pid",
            "process_group_id": "workload_process_group_id",
            "process_start_time": "workload_start_time",
        }
        for field_name, identity_name in mappings.items():
            value_item = value.get(identity_name)
            if getattr(job, field_name) is None and isinstance(value_item, int):
                setattr(job, field_name, value_item)

    @classmethod
    def _persisted_process_matches(cls, job: JobRecord) -> bool:
        pid = job.process_pid
        group_id = job.process_group_id
        start_time = job.process_start_time
        if pid is None or group_id is None or start_time is None or os.name == "nt":
            return False
        if cls._read_process_start_time(pid) != start_time:
            return False
        try:
            return os.getpgid(pid) == group_id
        except ProcessLookupError:
            return False

    @classmethod
    def _job_owner_is_alive(cls, job: JobRecord) -> bool:
        if job.runner_pid is None or job.runner_start_time is None:
            return False
        return cls._read_process_start_time(job.runner_pid) == job.runner_start_time

    @classmethod
    def _terminate_persisted_process_group(
        cls,
        job: JobRecord,
        *,
        timeout_s: float = 2.0,
    ) -> bool:
        """Stop a verified process group left by an interrupted runner."""

        stopped = False
        if cls._persisted_supervisor_matches(job):
            assert job.supervisor_process_group_id is not None
            try:
                os.killpg(job.supervisor_process_group_id, signal.SIGTERM)
                stopped = True
            except ProcessLookupError:
                pass
        if not cls._persisted_process_matches(job):
            return stopped
        assert job.process_group_id is not None
        try:
            os.killpg(job.process_group_id, signal.SIGTERM)
            stopped = True
        except ProcessLookupError:
            return stopped

        deadline = time.monotonic() + max(timeout_s, 0.0)
        while cls._persisted_process_matches(job) and time.monotonic() < deadline:
            time.sleep(0.02)
        if cls._persisted_process_matches(job):
            try:
                os.killpg(job.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return stopped

    @classmethod
    def _persisted_supervisor_matches(cls, job: JobRecord) -> bool:
        pid = job.supervisor_pid
        group_id = job.supervisor_process_group_id
        start_time = job.supervisor_start_time
        if pid is None or group_id is None or start_time is None or os.name == "nt":
            return False
        if cls._read_process_start_time(pid) != start_time:
            return False
        try:
            return os.getpgid(pid) == group_id
        except ProcessLookupError:
            return False

    @classmethod
    def _terminate_recorded_workload(cls, job: JobRecord, signum: int) -> bool:
        if not cls._persisted_process_matches(job):
            return False
        assert job.process_group_id is not None
        try:
            os.killpg(job.process_group_id, signum)
            return True
        except ProcessLookupError:
            return False

    def _persist_job(self, job: JobRecord) -> None:
        path = Path(job.log_path).parent / "job.json"
        atomic_write_json(path, job.to_dict())

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)


class ResourceBusyError(RuntimeError):
    """Raised when a job requests resources held by another active job."""
