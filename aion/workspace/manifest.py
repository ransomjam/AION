"""The project manifest — a small, deliberately extensible identity document.

Design bias (per the milestone refinement): **stability over completeness.** The
manifest carries only the fields we actually use today. It is a plain dict with a
``schema_version`` so new sections can be added later without breaking existing
projects:

- Loading an older manifest fills in any missing known fields with defaults, so
  new code reads old projects.
- Unknown fields are **preserved** on load and save, so older code never destroys
  data written by newer code.

That two-way tolerance is what lets the format evolve for years without
migrations.
"""

from __future__ import annotations

from aion.util import now_iso
from aion import __version__

SCHEMA_VERSION = 1

# Known fields and their defaults. New fields are added here; old projects that
# lack them simply get the default. Keep this small.
_DEFAULTS: dict = {
    "schema_version": SCHEMA_VERSION,
    "id": "",
    "name": "",
    "description": "",
    "created_at": "",
    "updated_at": "",
    "owner": "",
    "version": 1,
    "status": "active",           # active | archived
    "tags": [],
    "language": "",               # optional; domain-neutral (blank = unspecified)
    # defaults is the project's active artifact configuration.
    # Each key names an artifact type; the value is the artifact id or None.
    # New artifact types are added here as AION evolves.
    "defaults": {"tokenizer": None, "dataset": None, "model": None, "embedding": None},
    "research_notes": "",
    "aion_version": __version__,
}


def new_manifest(project_id: str, name: str, **fields) -> dict:
    """Create a manifest for a new project. Extra known fields may be passed in."""
    ts = now_iso()
    manifest = _with_defaults({})
    manifest.update({
        "id": project_id,
        "name": name,
        "created_at": ts,
        "updated_at": ts,
        "aion_version": __version__,
    })
    manifest.update({k: v for k, v in fields.items() if v is not None})
    return manifest


def _with_defaults(raw: dict) -> dict:
    """Return ``raw`` with all known fields present, unknown fields preserved."""
    out = dict(raw)  # keep unknown keys untouched
    for key, default in _DEFAULTS.items():
        if key not in out:
            out[key] = _copy_default(default)
    # Ensure nested 'defaults' has its known sub-keys too.
    defaults_section = dict(_DEFAULTS["defaults"])
    defaults_section.update(out.get("defaults") or {})
    out["defaults"] = defaults_section
    return out


def _copy_default(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def normalize(raw: dict) -> dict:
    """Bring a loaded manifest up to the current known shape (non-destructive)."""
    return _with_defaults(raw)


def touch(manifest: dict) -> dict:
    """Update the ``updated_at`` timestamp in place and return the manifest."""
    manifest["updated_at"] = now_iso()
    return manifest


def clear_default(manifest: dict, artifact_type: str, artifact_id: str) -> bool:
    """Null the defaults entry for ``artifact_type`` if it matches ``artifact_id``.

    Called whenever an artifact is deleted so the manifest never references a
    non-existent artifact.  Returns True if the entry was cleared, False if it
    was already None or pointed to a different artifact.
    """
    defaults = manifest.get("defaults") or {}
    if defaults.get(artifact_type) == artifact_id:
        defaults[artifact_type] = None
        manifest["defaults"] = defaults
        return True
    return False
