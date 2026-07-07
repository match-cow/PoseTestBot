"""Small process-backed job runner for local PoseTestBot commands."""

from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from posetestbot.io.manifest import utc_now_iso


QUEUED = "queued"
RUNNING = "running"
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
        )
        with self._lock:
            self._check_resources_available(requested_resources)
            self._jobs[job_id] = job
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
            job.status = CANCELED
            job.message = "Cancellation requested."
            job.ended_at = utc_now_iso()
            self._append_tail(job, "Cancellation requested.")
            self._persist_job(job)

        if process is not None and process.poll() is None:
            self._terminate_process_group(process)
        return self.get(job_id)

    def log_text(self, job_id: str) -> str:
        job = self.get(job_id)
        log_path = Path(job.log_path)
        if not log_path.is_file():
            return ""
        return log_path.read_text()

    def _run_job(self, job_id: str, env: dict[str, str]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == CANCELED:
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
                should_terminate = self._jobs[job_id].status == CANCELED

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
                if job.status == CANCELED:
                    job.message = job.message or "Canceled."
                elif returncode == 0:
                    job.status = SUCCEEDED
                    job.message = "Command completed successfully."
                else:
                    job.status = FAILED
                    job.message = f"Command exited with status {returncode}."
                self._append_tail(job, job.message)
                self._persist_job(job)
                self._processes.pop(job_id, None)

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
        conflicts = {
            resource: holders[resource]
            for resource in resources
            if resource in holders
        }
        if conflicts:
            details = ", ".join(
                f"{resource} held by job {job_id}"
                for resource, job_id in sorted(conflicts.items())
            )
            raise ResourceBusyError(f"Requested resources are busy: {details}")

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

            if job.status not in TERMINAL_STATUSES:
                job.status = FAILED
                job.ended_at = utc_now_iso()
                job.returncode = None
                job.message = "Job runner restarted before this job completed."
                self._append_tail(job, job.message)
                self._persist_job(job)
            self._jobs[job.id] = job

    @staticmethod
    def _job_from_dict(data: Mapping[str, object]) -> JobRecord:
        job_data = dict(data)
        job_data.setdefault("tail", [])
        job_data.setdefault("resources", [])
        job_data.setdefault("parameters", {})
        return JobRecord(**job_data)

    def _persist_job(self, job: JobRecord) -> None:
        path = Path(job.log_path).parent / "job.json"
        with open(path, "w") as f:
            json.dump(job.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return " ".join(shlex.quote(part) for part in command)


class ResourceBusyError(RuntimeError):
    """Raised when a job requests resources held by another active job."""
