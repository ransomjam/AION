# Application — AION's operating environment

*Cross-cutting surface. Follows the [component-doc template](../../docs/templates/component-doc.md).*

## Role in the platform

The environment the whole platform is used through. Project Explorer is the home
screen; every capability is a lab reached from a data-driven navigation. Scratch
labs (Tokenizer, Vocabulary) run without a project and persist nothing; project
labs (Data Lab, and coming BPE/embeddings/training/…) operate inside the open
project. Zero dependencies (stdlib server, zero-build frontend). See ADR
[0003](../../docs/adr/0003-application-as-operating-environment.md) and
[0004](../../docs/adr/0004-workspace-and-projects.md).

**Launch:** `python -m aion.app` → `http://127.0.0.1:7071/`.

## Contract / API

Layers: `api.py` (pure `request → JSON`, workspace-root seam for tests),
`server.py` (stdlib routing table), `labs.py` (the Lab Registry), `static/`
(vanilla frontend rendering nav from the registry).

HTTP surface:
- `GET /api/health`, `GET /api/labs` (navigation source of truth)
- scratch: `POST /api/tokenize`, `/api/vocabulary`
- projects: `POST /api/projects/{list,create,get,update,archive}`
- datasets: `POST /api/datasets/{list,create,get,delete,import_text,analyze,search}`
- documents: `POST /api/documents/{get,edit,delete}`
- jobs: `POST /api/jobs/{list,get}`

## Invariants

- **Capabilities are live, never reimplemented** — `api.py` calls the real
  `tokenization`, `workspace`, and `datasets` libraries.
- **Navigation has one source** — `labs.py`; the frontend hardcodes no lab list.
- **Scratch labs never touch project storage** — they call pure functions only.
- **Errors map to status, never to stack traces** — 400 (bad input), 404
  (missing project/dataset/job), 409 (duplicate), 500 (generic).

## Design notes

- **Route table, not an if-chain** — adding an endpoint is one `POST_ROUTES`
  entry. Adding a lab is one `labs.py` entry + a panel renderer.
- **Workspace-root seam** (`api.set_workspace_root`) points the whole project
  surface at a temp dir, so the server is testable without touching real data.
- Vanilla frontend + native SVG/bar charts — the deliberate, revisitable
  no-framework bet from ADR 0003.

## Performance & scaling

Single-user local tool on `ThreadingHTTPServer`; each call costs what the
underlying library call costs (analysis is cached). Multi-user/remote/auth are
out of scope until there is a reason — a future ADR.

## Benchmarks

N/A — interactive local tool; latency is dominated by the capability invoked.

## Tests

`tests/test_api.py` (scratch labs, full project→dataset→import→analyze→search
flow, jobs), `tests/test_server.py` (live server: routes, status-code mapping).

```
python -m unittest discover -s aion -p "test_*.py" -v
```
