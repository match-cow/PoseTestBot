"""Small process-backed job runner for local PoseTestBot commands."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from posetestbot.io.manifest import utc_now_iso
from posetestbot.io.atomic import atomic_write_json


QUEUED = "queued"
RUNNING = "running"
CANCELING = "canceling"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELED = "canceled"
TERMINAL_STATUSES = {SUCCEEDED, FAILED, CANCELED}


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

    def to_dict(self) -> dict:
        return asdict(self)


class LocalJobRunner:
    """Run structured command arrays in background threads and keep job logs."""

    def __init__(self, job_root: str | Path, *, tail_limit: int = 200):
        self.job_root = Path(job_root)
        self.tail_limit = tail_limit
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
    ) -> JobRecord:
        if not command:
            raise ValueError("Job command must not be empty")

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

    def resource_holders(self) -> dict[str, str]:
        with self._lock:
            return self._resource_holders()

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return JobRecord(**self._jobs[job_id].to_dict())
            except KeyError as exc:
                raise KeyError(f"Unknown job: {job_id}") from exc

    def list(self) -> list[JobRecord]:
        with self._lock:
            return [
                JobRecord(**job.to_dict())
                for job in sorted(
                    self._jobs.values(),
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
        """Cancel active jobs and wait briefly for runner threads to finish."""

        with self._lock:
            active_ids = [
                job.id
                for job in self._jobs.values()
                if job.id in self._local_job_ids
                and job.status not in TERMINAL_STATUSES
            ]
        for job_id in active_ids:
            self.cancel(job_id)

        deadline = time.monotonic() + max(timeout, 0.0)
        for job_id in active_ids:
            with self._lock:
                thread = self._threads.get(job_id)
            if thread is None:
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

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

        with open(job.log_path, "a", buffering=1) as log:
            log.write(f"$ {self._format_command(job.command)}\n")
            try:
                process = subprocess.Popen(
                    job.command,
                    cwd=job.cwd,
                    env={**os.environ, **env},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
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
                current.process_pid = process.pid
                current.process_group_id = (
                    os.getpgid(process.pid) if os.name != "nt" else process.pid
                )
                current.process_start_time = self._read_process_start_time(process.pid)
                self._persist_job(current)
                should_terminate = self._jobs[job_id].status in {
                    CANCELED,
                    CANCELING,
                }

            if should_terminate:
                self._terminate_process_group(process)

            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                with self._lock:
                    current = self._jobs[job_id]
                    self._append_tail(current, line.rstrip("\n"))
                    self._persist_job(current)

            returncode = process.wait()
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
        job.tail.append(line)
        if len(job.tail) > self.tail_limit:
            del job.tail[: len(job.tail) - self.tail_limit]

    def _resource_holders(self) -> dict[str, str]:
        holders = {}
        for job in self._jobs.values():
            if job.status in TERMINAL_STATUSES:
                continue
            for resource in job.resources:
                holders[resource] = job.id
        return holders

    def _check_resources_available(self, resources: list[str]) -> None:
        holders = self._resource_holders()
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
        self, process: subprocess.Popen, *, timeout_s: float = 2.0
    ) -> None:
        if process.poll() is not None:
            return

        if os.name == "nt":
            process.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass

    def _load_persisted_jobs(self) -> None:
        for path in sorted(self.job_root.glob("*/job.json")):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                job = self._job_from_dict(data)
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
        return JobRecord(**job_data)

    @staticmethod
    def _read_process_start_time(pid: int) -> int | None:
        """Return Linux process start ticks, used to guard against PID reuse."""

        if os.name == "nt":
            return None
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields_after_name = stat[stat.rfind(")") + 2 :].split()
            return int(fields_after_name[19])
        except (IndexError, OSError, ValueError):
            return None

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

        if not cls._persisted_process_matches(job):
            return False
        assert job.process_group_id is not None
        try:
            os.killpg(job.process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return False

        deadline = time.monotonic() + max(timeout_s, 0.0)
        while cls._persisted_process_matches(job) and time.monotonic() < deadline:
            time.sleep(0.02)
        if cls._persisted_process_matches(job):
            try:
                os.killpg(job.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return True

    def _persist_job(self, job: JobRecord) -> None:
        path = Path(job.log_path).parent / "job.json"
        atomic_write_json(path, job.to_dict())

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)


class ResourceBusyError(RuntimeError):
    """Raised when a job requests resources held by another active job."""
