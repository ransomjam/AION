"""Small cross-cutting helpers shared across the platform.

Deliberately tiny and dependency-free. Anything here is used by more than one
package (workspace, datasets, app); package-specific logic does not belong here.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timezone

from aion import __version__

__all__ = [
    "now_iso", "slugify", "sha256_text", "fingerprint", "code_version",
]


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a human name into a filesystem- and URL-safe id.

    "TinyGPT (v2)" -> "tinygpt-v2". Raises if nothing usable remains, because an
    empty id would collide and break addressing.
    """
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"name {name!r} produces an empty id")
    return slug


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a string (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(*parts: object) -> str:
    """Stable short hash over the given parts — used as a cache key.

    Order matters; parts are stringified and joined with a separator that cannot
    appear ambiguously. Returns the first 16 hex chars, plenty for local keying.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def code_version() -> dict:
    """Identify the code that produced an artifact, for reproducibility.

    Always records the package version; best-effort records the git commit so an
    artifact can be traced to exact source. Git absence is not an error — the
    commit is simply ``None``.
    """
    commit = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode == 0:
            commit = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"aion_version": __version__, "git_commit": commit}
