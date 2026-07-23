"""Workspace — AION's operating-system substrate.

The Project is the fundamental unit of the platform: every persistent artifact
(datasets, tokenizers, models, experiments, evaluations, exports, logs) belongs
to exactly one Project. This package owns that substrate:

    manifest.py   the small, extensible project identity document
    project.py    a Project handle + its directory layout
    store.py      ProjectStore: create / list / open / update / archive
    cache.py      derived-artifact cache (safely deletable, always regenerable)
    jobs.py       the Job abstraction (structured events, synchronous today)

Nothing here is domain-specific: a Project can hold text, code, image, audio,
multimodal, RL, or evaluation artifacts equally.
"""

from .cache import ProjectCache
from .experiments import ExperimentStore
from .jobs import Job, JobStatus, JobStore, run_job
from .project import Project
from .store import ProjectStore

__all__ = [
    "Project", "ProjectStore", "ProjectCache",
    "Job", "JobStatus", "JobStore", "run_job",
    "ExperimentStore",
]
