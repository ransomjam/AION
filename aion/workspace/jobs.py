"""The Job abstraction — traceable units of work with structured events.

Execution is **synchronous today**, but every job is a persisted record that
emits a structured event stream (queued → started → progress → completed/failed).
The record + event model is deliberately executor-agnostic: making jobs
asynchronous later means replacing the runner, not the callers or the on-disk
shape.

    run_job(project, "analyze_dataset", params, fn)  # returns a completed Job

``fn(progress)`` receives a callback ``progress(fraction, message="")`` it may
call to emit progress events. Whatever ``fn`` returns becomes the job result.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from aion.util import now_iso


class JobStatus:
    QUEUED = "queued"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"  # reserved for future asynchronous execution


class JobEventType:
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """A single unit of work and its lifecycle."""

    def __init__(self, job_type: str, project_id: str, params: dict | None = None,
                 job_id: str | None = None):
        self.id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        self.type = job_type
        self.project_id = project_id
        self.params = params or {}
        self.status = JobStatus.QUEUED
        self.progress = 0.0
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = now_iso()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.events: list[dict] = []

    def emit(self, event_type: str, **data) -> None:
        self.events.append({"type": event_type, "at": now_iso(), **data})

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "project_id": self.project_id,
            "params": self.params, "status": self.status, "progress": self.progress,
            "result": self.result, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        job = cls(d["type"], d["project_id"], d.get("params"), d["id"])
        job.status = d["status"]
        job.progress = d.get("progress", 0.0)
        job.result = d.get("result")
        job.error = d.get("error")
        job.created_at = d.get("created_at", job.created_at)
        job.started_at = d.get("started_at")
        job.finished_at = d.get("finished_at")
        job.events = d.get("events", [])
        return job


class JobStore:
    """Persists jobs as JSON under ``<project>/logs/jobs/``."""

    def __init__(self, logs_dir: Path):
        self.jobs_dir = Path(logs_dir) / "jobs"

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, job: Job) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._path(job.id).write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, job_id: str) -> Job | None:
        path = self._path(job_id)
        if not path.is_file():
            return None
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Job]:
        if not self.jobs_dir.is_dir():
            return []
        jobs = [Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
                for p in self.jobs_dir.glob("*.json")]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs


def run_job(project, job_type: str, params: dict, fn) -> Job:
    """Run ``fn`` as a tracked job on ``project`` and return the finished Job.

    Synchronous: ``fn(progress)`` runs inline, emitting events as it goes. A
    failure is captured on the job (status=failed, error set) rather than raised,
    so the caller always gets a Job record to inspect — the same contract an
    async executor would honour.
    """
    store = JobStore(project.logs_dir())
    job = Job(job_type, project.id, params)
    job.emit(JobEventType.QUEUED)
    store.save(job)

    job.status = JobStatus.STARTED
    job.started_at = now_iso()
    job.emit(JobEventType.STARTED)
    store.save(job)

    def progress(fraction: float, message: str = "") -> None:
        job.progress = max(0.0, min(1.0, float(fraction)))
        job.emit(JobEventType.PROGRESS, progress=job.progress, message=message)
        store.save(job)

    try:
        result = fn(progress)
        job.result = result if isinstance(result, dict) else {"value": result}
        job.progress = 1.0
        job.status = JobStatus.COMPLETED
        job.emit(JobEventType.COMPLETED)
    except Exception as exc:  # noqa: BLE001 - captured onto the job by design
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit(JobEventType.FAILED, error=job.error)
    finally:
        job.finished_at = now_iso()
        store.save(job)
    return job
