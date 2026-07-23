"""ProjectStore — create, list, open, update, and archive projects.

The workspace root resolves from (in order): an explicit constructor argument,
the ``AION_WORKSPACE`` environment variable, or ``./workspace``. Projects live
under ``<root>/projects/<id>/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from aion.util import slugify
from . import manifest as manifest_mod
from .project import Project


def default_workspace_root() -> Path:
    return Path(os.environ.get("AION_WORKSPACE", "workspace")).resolve()


class ProjectExists(Exception):
    """Raised when creating a project whose id already exists."""


class ProjectNotFound(Exception):
    """Raised when opening a project id that does not exist."""


class ProjectStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root).resolve() if root is not None else default_workspace_root()
        self.projects_dir = self.root / "projects"

    # ── create ─────────────────────────────────────────────────────────────────
    def create(self, name: str, *, description: str = "", language: str = "",
               owner: str = "", tags: list[str] | None = None) -> Project:
        project_id = slugify(name)
        root = self.projects_dir / project_id
        if root.exists():
            raise ProjectExists(f"project {project_id!r} already exists")
        manifest = manifest_mod.new_manifest(
            project_id, name, description=description, language=language,
            owner=owner, tags=tags or [])
        project = Project(root, manifest)
        project.ensure_dirs()
        project.save()
        return project

    # ── read ───────────────────────────────────────────────────────────────────
    def exists(self, project_id: str) -> bool:
        return (self.projects_dir / project_id / "manifest.json").is_file()

    def open(self, project_id: str) -> Project:
        root = self.projects_dir / project_id
        if not (root / "manifest.json").is_file():
            raise ProjectNotFound(f"no project {project_id!r}")
        return Project.load(root)

    def list(self) -> list[Project]:
        if not self.projects_dir.is_dir():
            return []
        projects = []
        for child in sorted(self.projects_dir.iterdir()):
            if (child / "manifest.json").is_file():
                projects.append(Project.load(child))
        # Most-recently-updated first — the natural home-screen ordering.
        projects.sort(key=lambda p: p.manifest.get("updated_at", ""), reverse=True)
        return projects

    # ── update ─────────────────────────────────────────────────────────────────
    _UPDATABLE = {"name", "description", "language", "owner", "tags",
                  "status", "research_notes", "defaults", "version"}

    def update(self, project_id: str, **fields) -> Project:
        project = self.open(project_id)
        unknown = set(fields) - self._UPDATABLE
        if unknown:
            raise ValueError(f"cannot update fields: {sorted(unknown)}")
        project.manifest.update(fields)
        return project.save()

    def archive(self, project_id: str) -> Project:
        return self.update(project_id, status="archived")

    def clear_default(self, project_id: str, artifact_type: str, artifact_id: str) -> Project:
        """Null ``defaults[artifact_type]`` if it currently equals ``artifact_id``.

        Called by every artifact store's delete path so the project manifest
        never references a non-existent artifact.
        """
        project = self.open(project_id)
        if manifest_mod.clear_default(project.manifest, artifact_type, artifact_id):
            project.save()
        return project
