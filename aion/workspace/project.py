"""A Project handle and its on-disk layout.

A Project is a directory under ``workspace/projects/<id>/`` containing a manifest
and a fixed set of artifact subdirectories. The subdirectory set is
**domain-neutral** — the same layout serves text, code, image, audio, multimodal,
RL, and evaluation research; nothing here assumes language models.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import manifest as manifest_mod

# The artifact directories every project has. Derived data goes ONLY in cache/;
# originals live in their typed directories and are never mutated by derived work.
SUBDIRS = (
    "data",          # source datasets (any modality)
    "tokenizers",
    "embeddings",
    "models",
    "checkpoints",
    "experiments",
    "evaluations",
    "exports",
    "cache",         # all derived artifacts, safely deletable
    "logs",          # job records + run logs
)

MANIFEST_NAME = "manifest.json"


class Project:
    """An opened project: its manifest plus helpers for its directory layout."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest_mod.normalize(manifest)

    # ── identity ───────────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def status(self) -> str:
        return self.manifest["status"]

    # ── layout ─────────────────────────────────────────────────────────────────
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def dir(self, name: str) -> Path:
        """Path to a named artifact subdirectory (created on demand)."""
        if name not in SUBDIRS:
            raise ValueError(f"unknown project subdir: {name!r}")
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def data_dir(self) -> Path: return self.dir("data")
    def cache_dir(self) -> Path: return self.dir("cache")
    def logs_dir(self) -> Path: return self.dir("logs")

    # ── persistence ────────────────────────────────────────────────────────────
    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def save(self) -> "Project":
        manifest_mod.touch(self.manifest)
        self.manifest_path().write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    @classmethod
    def load(cls, root: Path) -> "Project":
        root = Path(root)
        raw = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        return cls(root, raw)

    def summary(self) -> dict:
        """Compact record for listings (no heavy fields)."""
        m = self.manifest
        return {
            "id": m["id"], "name": m["name"], "description": m["description"],
            "status": m["status"], "language": m["language"], "tags": m["tags"],
            "created_at": m["created_at"], "updated_at": m["updated_at"],
        }
