"""Standard-library HTTP server for the AION application.

Zero dependencies. Serves the static frontend and a JSON API backed by
:mod:`aion.app.api`. Routing is a table, not a chain: adding a capability is one
entry in ``POST_ROUTES``. The API holds the logic; this module only parses,
dispatches, and maps exceptions to status codes.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from aion.datasets.store import DatasetExists, DatasetNotFound
from aion.workspace.store import ProjectExists, ProjectNotFound
from aion.tokenizers.store import TokenizerNotFound
from aion.embeddings.store import EmbeddingNotFound
from . import api

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 8 * 1024 * 1024

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


# ── POST route table: path -> (body -> dict) ──────────────────────────────────
POST_ROUTES = {
    # scratch labs
    "/api/tokenize": lambda b: api.tokenize_report(
        b["text"], form=b.get("form", "NFKC"),
        casefold=bool(b.get("casefold", True)), accents=bool(b.get("accents", True))),
    "/api/vocabulary": lambda b: api.vocabulary_report(
        b.get("corpus", []), min_freq=int(b.get("min_freq", 1)),
        max_size=(int(b["max_size"]) if b.get("max_size") not in (None, "") else None),
        probe=b.get("probe")),
    # projects
    "/api/projects/list": lambda b: api.list_projects(),
    "/api/projects/create": lambda b: api.create_project(
        b.get("name", ""), description=b.get("description", ""),
        language=b.get("language", ""), tags=b.get("tags")),
    "/api/projects/get": lambda b: api.get_project(b["id"]),
    "/api/projects/update": lambda b: api.update_project(b["id"], b.get("fields", {})),
    "/api/projects/archive": lambda b: api.archive_project(b["id"]),
    # datasets
    "/api/datasets/list": lambda b: api.list_datasets(b["project_id"]),
    "/api/datasets/create": lambda b: api.create_dataset(
        b["project_id"], b.get("name", ""), description=b.get("description", ""),
        source=b.get("source", ""), language=b.get("language", "")),
    "/api/datasets/get": lambda b: api.get_dataset(b["project_id"], b["dataset_id"]),
    "/api/datasets/delete": lambda b: api.delete_dataset(b["project_id"], b["dataset_id"]),
    "/api/datasets/import_text": lambda b: api.import_text(
        b["project_id"], b["dataset_id"], b.get("text", ""), split=b.get("split", "one")),
    "/api/datasets/analyze": lambda b: api.analyze_dataset(b["project_id"], b["dataset_id"]),
    "/api/datasets/search": lambda b: api.search_documents(
        b["project_id"], b["dataset_id"], b.get("query", ""),
        regex=b.get("regex", False), case_sensitive=b.get("case_sensitive", False)),
    # documents
    "/api/documents/get": lambda b: api.get_document(b["project_id"], b["dataset_id"], b["doc_id"]),
    "/api/documents/edit": lambda b: api.edit_document(
        b["project_id"], b["dataset_id"], b["doc_id"], b.get("text", "")),
    "/api/documents/delete": lambda b: api.delete_document(b["project_id"], b["dataset_id"], b["doc_id"]),
    # jobs
    "/api/jobs/list": lambda b: api.list_jobs(b["project_id"]),
    "/api/jobs/get": lambda b: api.get_job(b["project_id"], b["job_id"]),
    # tokenizers
    "/api/tokenizers/train": lambda b: api.train_tokenizer(
        b["project_id"], b["dataset_id"], b.get("name", ""),
        vocab_size=int(b.get("vocab_size", 1000)),
        description=b.get("description", "")),
    "/api/tokenizers/list": lambda b: api.list_tokenizers(b["project_id"]),
    "/api/tokenizers/get": lambda b: api.get_tokenizer(b["project_id"], b["tokenizer_id"]),
    "/api/tokenizers/delete": lambda b: api.delete_tokenizer(b["project_id"], b["tokenizer_id"]),
    "/api/tokenizers/set_default": lambda b: api.set_default_tokenizer(b["project_id"], b["tokenizer_id"]),
    "/api/tokenizers/encode": lambda b: api.encode_with_tokenizer(
        b["project_id"], b["tokenizer_id"], b.get("text", "")),
    # embeddings
    "/api/embeddings/train": lambda b: api.train_embedding(
        b["project_id"], b["dataset_id"], b.get("name", ""),
        tokenizer_id=b.get("tokenizer_id"),
        dims=int(b.get("dims", 64)),
        epochs=int(b.get("epochs", 5)),
        window=int(b.get("window", 2)),
        neg_samples=int(b.get("neg_samples", 5)),
        seed=int(b.get("seed", 42)),
        lr=float(b.get("lr", 0.025)),
        description=b.get("description", "")),
    "/api/embeddings/list": lambda b: api.list_embeddings(b["project_id"]),
    "/api/embeddings/get": lambda b: api.get_embedding(b["project_id"], b["embedding_id"]),
    "/api/embeddings/delete": lambda b: api.delete_embedding(b["project_id"], b["embedding_id"]),
    "/api/embeddings/set_default": lambda b: api.set_default_embedding(b["project_id"], b["embedding_id"]),
    "/api/embeddings/query": lambda b: api.query_embedding(
        b["project_id"], b["embedding_id"], b.get("text", ""),
        n=int(b.get("n", 10))),
    "/api/embeddings/pca": lambda b: api.project_embedding_pca(
        b["project_id"], b["embedding_id"], n=int(b.get("n", 200))),
}


class AionHandler(BaseHTTPRequestHandler):
    server_version = "AION/0.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/health":
            self._send_json(200, {"status": "ok", "version": api.labs_registry()["version"]})
        elif path == "/api/labs":
            self._send_json(200, api.labs_registry())
        else:
            self._send_json(404, {"error": f"no such resource: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        handler = POST_ROUTES.get(path)
        if handler is None:
            self._send_json(404, {"error": f"no such endpoint: {path}"})
            return
        try:
            result = handler(self._read_json())
        except (ProjectNotFound, DatasetNotFound, TokenizerNotFound, EmbeddingNotFound, KeyError) as exc:
            self._send_json(404, {"error": str(exc)})
        except (ProjectExists, DatasetExists) as exc:
            self._send_json(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace
            self._send_json(500, {"error": f"internal error: {exc}"})
        else:
            self._send_json(200, result)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative: str) -> None:
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in target.parents or not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        self._serve_file(target)

    def _serve_file(self, target: Path) -> None:
        if not target.is_file():
            self._send_json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        return


def build_server(host: str = "127.0.0.1", port: int = 7071) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), AionHandler)


def serve(host: str = "127.0.0.1", port: int = 7071) -> None:
    httpd = build_server(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"AION application running at {url}")
    print("Open it in a browser. Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AION.")
    finally:
        httpd.server_close()
