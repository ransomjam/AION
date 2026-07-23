"""Dataset storage — transparent files, stable ids, immutable originals.

Layout, under a project's ``data/`` directory:

    data/<dataset_id>/
        dataset.json          metadata + counters only (small, scales)
        documents/000001.txt  one raw UTF-8 document per file

Document ids are stable and monotonic; deleting a document leaves a gap and never
renumbers, because ids are references that training splits and experiments rely
on. ``dataset.json`` never inlines document content or derived statistics.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

from aion.util import code_version, fingerprint, now_iso, slugify

SCHEMA_VERSION = 1
DOC_DIGITS = 6  # 000001 .. 999999, then widens naturally


class DatasetExists(Exception):
    pass


class DatasetNotFound(Exception):
    pass


class DatasetStore:
    """Create, list, open, and delete datasets within one project's data dir."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def create(self, name: str, *, description: str = "", source: str = "",
               language: str = "") -> "Dataset":
        dataset_id = slugify(name)
        root = self.data_dir / dataset_id
        if root.exists():
            raise DatasetExists(f"dataset {dataset_id!r} already exists")
        (root / "documents").mkdir(parents=True)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "id": dataset_id, "name": name, "description": description,
            "source": source, "language": language, "version": 1,
            "created_at": now_iso(), "updated_at": now_iso(),
            "document_count": 0, "next_id": 1,
            "provenance": {"created_by": code_version()},
        }
        ds = Dataset(root, meta)
        ds._save()
        return ds

    def exists(self, dataset_id: str) -> bool:
        return (self.data_dir / dataset_id / "dataset.json").is_file()

    def open(self, dataset_id: str) -> "Dataset":
        root = self.data_dir / dataset_id
        if not (root / "dataset.json").is_file():
            raise DatasetNotFound(f"no dataset {dataset_id!r}")
        return Dataset.load(root)

    def list(self) -> list["Dataset"]:
        if not self.data_dir.is_dir():
            return []
        out = []
        for child in sorted(self.data_dir.iterdir()):
            if (child / "dataset.json").is_file():
                out.append(Dataset.load(child))
        out.sort(key=lambda d: d.meta.get("updated_at", ""), reverse=True)
        return out

    def delete(self, dataset_id: str) -> None:
        root = self.data_dir / dataset_id
        if root.is_dir():
            shutil.rmtree(root)


class Dataset:
    """One dataset: metadata plus its documents."""

    def __init__(self, root: Path, meta: dict):
        self.root = Path(root)
        self.meta = meta

    # ── persistence ────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, root: Path) -> "Dataset":
        root = Path(root)
        meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        return cls(root, meta)

    def _save(self) -> None:
        self.meta["updated_at"] = now_iso()
        (self.root / "dataset.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    def _doc_path(self, doc_id: int) -> Path:
        return self.documents_dir / f"{doc_id:0{DOC_DIGITS}d}.txt"

    # ── documents ──────────────────────────────────────────────────────────────
    def add_document(self, text: str) -> int:
        """Append a document, returning its new stable id."""
        doc_id = self.meta["next_id"]
        self._doc_path(doc_id).write_text(text, encoding="utf-8")
        self.meta["next_id"] = doc_id + 1
        self.meta["document_count"] += 1
        self._save()
        return doc_id

    def add_documents(self, texts) -> list[int]:
        ids = [self.add_document(t) for t in texts]
        return ids

    def get_document(self, doc_id: int) -> str:
        path = self._doc_path(doc_id)
        if not path.is_file():
            raise DatasetNotFound(f"no document {doc_id} in {self.id!r}")
        return path.read_text(encoding="utf-8")

    def edit_document(self, doc_id: int, text: str) -> None:
        if not self._doc_path(doc_id).is_file():
            raise DatasetNotFound(f"no document {doc_id} in {self.id!r}")
        self._doc_path(doc_id).write_text(text, encoding="utf-8")
        self._save()

    def delete_document(self, doc_id: int) -> None:
        path = self._doc_path(doc_id)
        if not path.is_file():
            raise DatasetNotFound(f"no document {doc_id} in {self.id!r}")
        path.unlink()
        self.meta["document_count"] -= 1
        self._save()  # note: next_id is NOT decremented — ids never get reused

    def document_ids(self) -> list[int]:
        if not self.documents_dir.is_dir():
            return []
        ids = [int(p.stem) for p in self.documents_dir.glob("*.txt")]
        return sorted(ids)

    def stream(self) -> Iterator[tuple[int, str]]:
        """Iterate ``(doc_id, text)`` in id order — the pipeline corpus contract."""
        for doc_id in self.document_ids():
            yield doc_id, self._doc_path(doc_id).read_text(encoding="utf-8")

    # ── import ─────────────────────────────────────────────────────────────────
    def import_file(self, path: str | Path) -> int:
        """Import a single text file as one document."""
        return self.add_document(Path(path).read_text(encoding="utf-8"))

    def import_folder(self, path: str | Path, pattern: str = "*.txt") -> list[int]:
        """Import every matching file in a folder, one document each (sorted)."""
        folder = Path(path)
        ids = []
        for f in sorted(folder.glob(pattern)):
            if f.is_file():
                ids.append(self.import_file(f))
        return ids

    # ── identity for caching ───────────────────────────────────────────────────
    def fingerprint(self) -> str:
        """A hash of the dataset's current state, for keying derived caches.

        Derived from each document's id and byte size — cheap to compute and
        changes whenever any document is added, edited, or removed.
        """
        parts = []
        for doc_id in self.document_ids():
            parts.append((doc_id, self._doc_path(doc_id).stat().st_size))
        return fingerprint(self.id, tuple(parts))

    def summary(self) -> dict:
        m = self.meta
        return {
            "id": m["id"], "name": m["name"], "description": m["description"],
            "source": m["source"], "language": m["language"],
            "document_count": m["document_count"],
            "created_at": m["created_at"], "updated_at": m["updated_at"],
        }
