"""Project cache — derived artifacts, safely deletable and always regenerable.

Invariants (per milestone refinement 4):

- **Originals are never touched.** The cache only ever holds *derived* data.
- **Every entry is safely deletable.** Deleting a cache entry (or the whole
  cache) loses no information — anything here can be recomputed from source
  artifacts.
- **Freshness is explicit.** Entries are keyed by a fingerprint of the exact
  inputs that determined them; a fingerprint miss means recompute, so stale data
  can never masquerade as fresh.

Layout: ``<cache>/<namespace>/<fingerprint>.json``. Each file wraps the value in
a small envelope recording how it was produced.
"""

from __future__ import annotations

import json
from pathlib import Path

from aion.util import code_version, now_iso


class ProjectCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def _path(self, namespace: str, fingerprint: str) -> Path:
        return self.cache_dir / namespace / f"{fingerprint}.json"

    def get(self, namespace: str, fingerprint: str) -> dict | None:
        """Return the cached value, or ``None`` on a miss."""
        path = self._path(namespace, fingerprint)
        if not path.is_file():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        return envelope.get("value")

    def put(self, namespace: str, fingerprint: str, value: dict) -> dict:
        """Write a value, wrapped with provenance. Returns the value."""
        path = self._path(namespace, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "namespace": namespace,
            "fingerprint": fingerprint,
            "created_at": now_iso(),
            "produced_by": code_version(),
            "value": value,
        }
        path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        return value

    def get_or_compute(self, namespace: str, fingerprint: str, compute_fn) -> dict:
        """Return the cached value for this fingerprint, computing + storing on miss.

        ``compute_fn`` is a zero-argument callable returning a JSON-able dict. It
        runs only on a miss, so recomputation cost is paid once per distinct input.
        """
        hit = self.get(namespace, fingerprint)
        if hit is not None:
            return hit
        return self.put(namespace, fingerprint, compute_fn())

    # ── deletion (always safe) ─────────────────────────────────────────────────
    def delete(self, namespace: str, fingerprint: str | None = None) -> None:
        """Delete one entry, or a whole namespace when ``fingerprint`` is None."""
        if fingerprint is not None:
            self._path(namespace, fingerprint).unlink(missing_ok=True)
            return
        ns_dir = self.cache_dir / namespace
        if ns_dir.is_dir():
            for f in ns_dir.glob("*.json"):
                f.unlink(missing_ok=True)
            if not any(ns_dir.iterdir()):
                ns_dir.rmdir()

    def clear(self) -> None:
        """Delete the entire cache. Loses nothing — all of it is regenerable."""
        if not self.cache_dir.is_dir():
            return
        for ns_dir in self.cache_dir.iterdir():
            if ns_dir.is_dir():
                for f in ns_dir.glob("*.json"):
                    f.unlink(missing_ok=True)
