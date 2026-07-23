"""Experiment store — one JSON record per training run.

A generic workspace primitive: every training run (tokenizer, embedding, model)
auto-creates an experiment record here.  The Experiment Lab (roadmap) will read
these; for now they are written and queryable.

Layout: ``projects/<p>/experiments/<experiment_id>.json``
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from aion.util import code_version, now_iso


class ExperimentStore:
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = Path(experiments_dir)

    def _path(self, experiment_id: str) -> Path:
        return self.experiments_dir / f"{experiment_id}.json"

    def record(
        self,
        *,
        experiment_type: str,
        dataset_id: str,
        dataset_fingerprint: str,
        params: dict,
        artifact_id: str,
        metrics: dict,
    ) -> dict:
        """Write and return an experiment record."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": f"exp_{uuid.uuid4().hex[:12]}",
            "type": experiment_type,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "params": params,
            "artifact_id": artifact_id,
            "metrics": metrics,
            "created_at": now_iso(),
            "produced_by": code_version(),
        }
        self._path(record["id"]).write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record

    def list(self) -> list[dict]:
        if not self.experiments_dir.is_dir():
            return []
        records = []
        for p in self.experiments_dir.glob("*.json"):
            try:
                records.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records

    def get(self, experiment_id: str) -> dict | None:
        path = self._path(experiment_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
