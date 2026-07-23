"""Capability API — pure functions from request data to JSON-able reports.

Thin adapters over the real libraries (`tokenization`, `workspace`, `datasets`).
No HTTP here; `server.py` parses requests and calls these. Errors are raised as
ordinary exceptions and mapped to status codes by the server.

A workspace-root seam (`set_workspace_root`) lets tests point the whole project
surface at a temporary directory.
"""

from __future__ import annotations

import re

from aion.tokenization import Vocabulary, normalize, tokenize
from aion.workspace import ExperimentStore, ProjectCache, ProjectStore, run_job
from aion.workspace.jobs import JobStore
from aion.datasets import (
    DatasetStore, compute_quality, compute_statistics, search_dataset,
)
from aion.tokenizers import ByteLevelBPETokenizer, TokenizerNotFound, TokenizerStore
from aion.embeddings import CBOWEmbedding, EmbeddingNotFound, EmbeddingStore
from . import labs

__all__ = [
    "labs_registry", "tokenize_report", "vocabulary_report",
    "list_projects", "create_project", "get_project", "update_project",
    "archive_project",
    "list_datasets", "create_dataset", "get_dataset", "delete_dataset",
    "import_text", "get_document", "edit_document", "delete_document",
    "analyze_dataset", "search_documents",
    "list_jobs", "get_job",
    "train_tokenizer", "list_tokenizers", "get_tokenizer", "delete_tokenizer",
    "set_default_tokenizer", "encode_with_tokenizer",
    "train_embedding", "list_embeddings", "get_embedding", "delete_embedding",
    "set_default_embedding", "query_embedding", "project_embedding_pca",
    "set_workspace_root",
]

# ── workspace seam (overridable for tests) ────────────────────────────────────
_WORKSPACE_ROOT = None


def set_workspace_root(root) -> None:
    global _WORKSPACE_ROOT
    _WORKSPACE_ROOT = root


def _store() -> ProjectStore:
    return ProjectStore(_WORKSPACE_ROOT)


def _dataset_store(project) -> DatasetStore:
    return DatasetStore(project.data_dir())


# ── navigation ────────────────────────────────────────────────────────────────
def labs_registry() -> dict:
    return labs.registry()


# ── scratch labs (no project; never touch project storage) ────────────────────
def tokenize_report(text, *, form="NFKC", casefold=True, accents=True) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    from collections import Counter
    normalized = normalize(text, form=form, casefold=casefold, accents=accents)
    tokens = tokenize(normalized, normalize_first=False)
    counts = Counter(tokens)
    explanation = [f"Applied Unicode {form} normalization."]
    if casefold:
        explanation.append("Case-folded (Unicode-aware, e.g. ß → ss).")
    if accents:
        explanation.append("Stripped accents / combining marks.")
    explanation.append("Collapsed whitespace to single spaces.")
    if normalized != text:
        explanation.append("Normalization changed the text (see the stages above).")
    explanation.append(f"Segmented into {len(tokens)} token(s), {len(counts)} unique.")
    return {
        "input": text,
        "options": {"form": form, "casefold": casefold, "accents": accents},
        "stages": [{"name": "input", "value": text},
                   {"name": "normalized", "value": normalized}],
        "tokens": tokens,
        "stats": {
            "char_count": len(text), "token_count": len(tokens),
            "unique_tokens": len(counts),
            "type_token_ratio": round(len(counts) / len(tokens), 4) if tokens else 0.0,
        },
        "frequencies": counts.most_common(),
        "explanation": explanation,
    }


def vocabulary_report(corpus, *, min_freq=1, max_size=None, probe=None) -> dict:
    docs = [d for d in corpus if isinstance(d, str) and d.strip()]
    vocab = Vocabulary.build(docs, min_freq=min_freq, max_size=max_size)
    specials = set(vocab.specials)
    entries = [
        {"id": i, "token": vocab.id_to_token(i),
         "frequency": vocab.frequency(vocab.id_to_token(i)),
         "special": vocab.id_to_token(i) in specials}
        for i in range(len(vocab))
    ]
    report = {
        "params": {"min_freq": min_freq, "max_size": max_size},
        "stats": {"documents": len(docs), "vocab_size": len(vocab),
                  "specials": list(vocab.specials),
                  "corpus_tokens": len(vocab) - len(vocab.specials)},
        "entries": entries,
    }
    if probe is not None and probe.strip():
        ids = vocab.encode(probe)
        probe_tokens = tokenize(probe)
        unk_id = vocab.token_to_id("<unk>")
        report["probe"] = {
            "text": probe, "tokens": probe_tokens, "ids": ids,
            "unknown": [t for t, i in zip(probe_tokens, ids) if i == unk_id],
            "decoded": vocab.decode(ids, skip_specials=False),
        }
    return report


# ── projects ──────────────────────────────────────────────────────────────────
def list_projects() -> dict:
    return {"projects": [p.summary() for p in _store().list()]}


def create_project(name, *, description="", language="", tags=None) -> dict:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project name is required")
    project = _store().create(name, description=description, language=language,
                              tags=tags or [])
    return {"project": project.summary()}


def get_project(project_id) -> dict:
    project = _store().open(project_id)
    return {"project": project.manifest}


def update_project(project_id, fields) -> dict:
    if not isinstance(fields, dict):
        raise ValueError("fields must be an object")
    project = _store().update(project_id, **fields)
    return {"project": project.manifest}


def archive_project(project_id) -> dict:
    return {"project": _store().archive(project_id).summary()}


# ── datasets (project-scoped) ─────────────────────────────────────────────────
def list_datasets(project_id) -> dict:
    project = _store().open(project_id)
    return {"datasets": [d.summary() for d in _dataset_store(project).list()]}


def create_dataset(project_id, name, *, description="", source="", language="") -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).create(name, description=description,
                                        source=source, language=language)
    return {"dataset": ds.summary()}


def get_dataset(project_id, dataset_id) -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    return {"dataset": ds.meta, "document_ids": ds.document_ids()}


def delete_dataset(project_id, dataset_id) -> dict:
    project = _store().open(project_id)
    _dataset_store(project).delete(dataset_id)
    return {"deleted": dataset_id}


def _split_text(text: str, mode: str) -> list[str]:
    if mode == "lines":
        return [ln for ln in text.splitlines() if ln.strip()]
    if mode == "paragraphs":
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [text] if text.strip() else []  # "one"


def import_text(project_id, dataset_id, text, *, split="one") -> dict:
    """Import pasted text as one or more documents, as a tracked job."""
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    docs = _split_text(text or "", split)

    def work(progress):
        ids = []
        for i, doc in enumerate(docs):
            ids.append(ds.add_document(doc))
            progress((i + 1) / len(docs), f"imported {i + 1}/{len(docs)}")
        return {"imported": len(ids), "doc_ids": ids}

    if not docs:
        job = run_job(project, "import_dataset", {"dataset_id": dataset_id},
                      lambda progress: {"imported": 0, "doc_ids": []})
    else:
        job = run_job(project, "import_dataset",
                      {"dataset_id": dataset_id, "split": split}, work)
    return {"job": job.to_dict(), "dataset": ds.summary()}


def get_document(project_id, dataset_id, doc_id) -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    return {"doc_id": int(doc_id), "text": ds.get_document(int(doc_id))}


def edit_document(project_id, dataset_id, doc_id, text) -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    ds.edit_document(int(doc_id), text)
    return {"doc_id": int(doc_id)}


def delete_document(project_id, dataset_id, doc_id) -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    ds.delete_document(int(doc_id))
    return {"deleted": int(doc_id)}


def analyze_dataset(project_id, dataset_id) -> dict:
    """Compute statistics + quality as a job, cached by dataset fingerprint."""
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    cache = ProjectCache(project.cache_dir())
    fp = ds.fingerprint()
    key = f"{dataset_id}-{fp}"

    def work(progress):
        progress(0.1, "statistics")
        cache.get_or_compute("dataset-stats", key, lambda: compute_statistics(ds))
        progress(0.6, "quality")
        cache.get_or_compute("dataset-quality", key, lambda: compute_quality(ds))
        progress(1.0, "done")
        return {"dataset_id": dataset_id, "fingerprint": fp}

    job = run_job(project, "analyze_dataset", {"dataset_id": dataset_id}, work)
    return {
        "job": job.to_dict(),
        "statistics": cache.get("dataset-stats", key),
        "quality": cache.get("dataset-quality", key),
    }


def search_documents(project_id, dataset_id, query, *, regex=False,
                     case_sensitive=False) -> dict:
    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    return search_dataset(ds, query, regex=bool(regex),
                          case_sensitive=bool(case_sensitive))


def _tokenizer_store(project) -> TokenizerStore:
    return TokenizerStore(project.dir("tokenizers"))


def train_tokenizer(
    project_id,
    dataset_id,
    name,
    *,
    vocab_size=1000,
    description="",
) -> dict:
    """Train a Byte-Level BPE tokenizer as a tracked job."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tokenizer name is required")
    vocab_size = int(vocab_size)

    project = _store().open(project_id)
    ds = _dataset_store(project).open(dataset_id)
    fp = ds.fingerprint()
    cache = ProjectCache(project.cache_dir())
    ts = _tokenizer_store(project)
    exp_store = ExperimentStore(project.dir("experiments"))

    def work(progress):
        # Pre-tokenized word frequencies are cached by dataset fingerprint.
        def compute_word_freq():
            from collections import Counter
            from aion.tokenization.text import normalize, tokenize
            counts = Counter()
            for _doc_id, text in ds.stream():
                for word in tokenize(normalize(text), normalize_first=False):
                    counts[word] += 1
            return {"word_freq": dict(counts)}

        progress(0.05, "pre-tokenizing corpus")
        cache.get_or_compute("word-frequencies", fp, compute_word_freq)

        # Stream corpus documents for training
        corpus = (text for _doc_id, text in ds.stream())

        tokenizer = ByteLevelBPETokenizer()
        result = tokenizer.train(
            corpus,
            vocab_size=vocab_size,
            progress_fn=lambda frac, msg: progress(0.05 + frac * 0.85, msg),
            dataset_fingerprint=fp,
        )

        progress(0.92, "saving tokenizer")
        params = {"vocab_size": vocab_size, "algorithm": tokenizer.algorithm}
        manifest = ts.save(
            tokenizer, result,
            name=name, description=description,
            dataset_id=dataset_id, dataset_fingerprint=fp,
            params=params,
        )

        progress(0.97, "recording experiment")
        exp_store.record(
            experiment_type="train_tokenizer",
            dataset_id=dataset_id,
            dataset_fingerprint=fp,
            params=params,
            artifact_id=manifest["id"],
            metrics=result.metrics,
        )

        progress(1.0, "done")
        return {"tokenizer_id": manifest["id"], "manifest": manifest}

    job = run_job(project, "train_tokenizer",
                  {"dataset_id": dataset_id, "vocab_size": vocab_size}, work)
    return {"job": job.to_dict()}


def list_tokenizers(project_id) -> dict:
    project = _store().open(project_id)
    return {"tokenizers": _tokenizer_store(project).list()}


def get_tokenizer(project_id, tokenizer_id) -> dict:
    project = _store().open(project_id)
    ts = _tokenizer_store(project)
    manifest = ts.open_manifest(tokenizer_id)
    stats = ts.statistics(tokenizer_id)
    return {"manifest": manifest, "statistics": stats}


def delete_tokenizer(project_id, tokenizer_id) -> dict:
    project = _store().open(project_id)
    _tokenizer_store(project).delete(tokenizer_id)
    _store().clear_default(project_id, "tokenizer", tokenizer_id)
    return {"deleted": tokenizer_id}


def set_default_tokenizer(project_id, tokenizer_id) -> dict:
    project = _store().open(project_id)
    ts = _tokenizer_store(project)
    # Clear previous default
    for m in ts.list():
        if m.get("status") == "default" and m["id"] != tokenizer_id:
            ts.set_status(m["id"], "experimental")
    manifest = ts.set_status(tokenizer_id, "default")
    # Record on the project manifest
    _store().update(project_id, defaults={**project.manifest["defaults"],
                                          "tokenizer": tokenizer_id})
    return {"manifest": manifest}


def encode_with_tokenizer(project_id, tokenizer_id, text) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    project = _store().open(project_id)
    tokenizer = _tokenizer_store(project).load(tokenizer_id)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    return {"text": text, "ids": ids, "token_count": len(ids), "decoded": decoded}


# ── embeddings (project-scoped) ───────────────────────────────────────────────
def _embedding_store(project) -> EmbeddingStore:
    return EmbeddingStore(project.dir("embeddings"))


def train_embedding(
    project_id,
    dataset_id,
    name,
    *,
    tokenizer_id=None,
    dims=64,
    epochs=5,
    window=2,
    neg_samples=5,
    seed=42,
    lr=0.025,
    description="",
) -> dict:
    """Train a CBOW embedding as a tracked job."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("embedding name is required")

    project = _store().open(project_id)

    resolved_tid = tokenizer_id or project.manifest.get("defaults", {}).get("tokenizer")
    if not resolved_tid:
        raise ValueError(
            "tokenizer_id is required (or set a project default tokenizer first)"
        )

    ts = _tokenizer_store(project)
    tok_manifest = ts.open_manifest(resolved_tid)
    tok_fp = tok_manifest.get("vocabulary_fingerprint", "")

    ds = _dataset_store(project).open(dataset_id)
    ds_fp = ds.fingerprint()

    es = _embedding_store(project)
    exp_store = ExperimentStore(project.dir("experiments"))

    params = {
        "dims": int(dims), "epochs": int(epochs), "window": int(window),
        "neg_samples": int(neg_samples), "seed": int(seed), "lr": float(lr),
        "algorithm": "cbow-v1",
    }

    def work(progress):
        progress(0.02, "loading tokenizer")
        tokenizer = ts.load(resolved_tid)

        corpus = (text for _doc_id, text in ds.stream())
        embedding = CBOWEmbedding()
        result = embedding.train(
            corpus,
            tokenizer=tokenizer,
            dims=int(dims),
            epochs=int(epochs),
            window=int(window),
            neg_samples=int(neg_samples),
            seed=int(seed),
            lr=float(lr),
            progress_fn=lambda frac, msg: progress(0.02 + frac * 0.88, msg),
            dataset_fingerprint=ds_fp,
            tokenizer_fingerprint=tok_fp,
        )

        progress(0.92, "saving embedding")
        manifest = es.save(
            embedding, result,
            name=name, description=description,
            tokenizer_id=resolved_tid, tokenizer_fingerprint=tok_fp,
            dataset_id=dataset_id, dataset_fingerprint=ds_fp,
            params=params,
        )

        progress(0.97, "recording experiment")
        exp_store.record(
            experiment_type="train_embedding",
            dataset_id=dataset_id,
            dataset_fingerprint=ds_fp,
            params=params,
            artifact_id=manifest["id"],
            metrics=result.metrics,
        )

        progress(1.0, "done")
        return {"embedding_id": manifest["id"], "manifest": manifest}

    job = run_job(project, "train_embedding",
                  {"dataset_id": dataset_id, **params}, work)
    return {"job": job.to_dict()}


def list_embeddings(project_id) -> dict:
    project = _store().open(project_id)
    return {"embeddings": _embedding_store(project).list()}


def get_embedding(project_id, embedding_id) -> dict:
    project = _store().open(project_id)
    es = _embedding_store(project)
    manifest = es.open_manifest(embedding_id)
    stats = es.statistics(embedding_id)
    return {"manifest": manifest, "statistics": stats}


def delete_embedding(project_id, embedding_id) -> dict:
    project = _store().open(project_id)
    _embedding_store(project).delete(embedding_id)
    _store().clear_default(project_id, "embedding", embedding_id)
    return {"deleted": embedding_id}


def set_default_embedding(project_id, embedding_id) -> dict:
    project = _store().open(project_id)
    es = _embedding_store(project)
    for m in es.list():
        if m.get("status") == "default" and m["id"] != embedding_id:
            es.set_status(m["id"], "experimental")
    manifest = es.set_status(embedding_id, "default")
    _store().update(project_id, defaults={**project.manifest["defaults"],
                                          "embedding": embedding_id})
    return {"manifest": manifest}


def query_embedding(project_id, embedding_id, text, *, n=10) -> dict:
    """Return nearest-neighbour tokens for ``text`` using a saved embedding."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    project = _store().open(project_id)
    es = _embedding_store(project)
    manifest = es.open_manifest(embedding_id)
    embedding = es.load(embedding_id)
    tokenizer = _tokenizer_store(project).load(manifest["tokenizer_id"])
    ids = tokenizer.encode(text)
    if not ids:
        return {"text": text, "results": []}
    query_id = ids[0]
    neighbours = embedding.most_similar(query_id, n=int(n))
    results = [
        {"token_id": tid, "similarity": round(sim, 4),
         "token": tokenizer.decode([tid])}
        for tid, sim in neighbours
    ]
    return {"text": text, "query_token_id": query_id, "results": results}


def project_embedding_pca(project_id, embedding_id, *, n=200) -> dict:
    """Return 2D PCA projection of the top-n tokens, cached by embedding fingerprint."""
    project = _store().open(project_id)
    es = _embedding_store(project)
    manifest = es.open_manifest(embedding_id)
    emb_fp = manifest.get("embedding_fingerprint", embedding_id)
    cache = ProjectCache(project.cache_dir())

    def compute():
        from aion.embeddings.linalg import pca2
        embedding = es.load(embedding_id)
        tokenizer = _tokenizer_store(project).load(manifest["tokenizer_id"])
        V = embedding.vocab_size
        ids = list(range(min(int(n), V)))
        matrix = [embedding.encode(i) for i in ids]
        if not matrix or len(matrix[0]) < 2:
            return {"points": []}
        coords = pca2(matrix)
        points = [
            {"token_id": i, "token": tokenizer.decode([i]),
             "x": round(x, 5), "y": round(y, 5)}
            for i, (x, y) in zip(ids, coords)
        ]
        return {"points": points}

    return cache.get_or_compute("embedding-pca", emb_fp, compute)


def list_jobs(project_id) -> dict:
    project = _store().open(project_id)
    return {"jobs": [j.to_dict() for j in JobStore(project.logs_dir()).list()]}


def get_job(project_id, job_id) -> dict:
    project = _store().open(project_id)
    job = JobStore(project.logs_dir()).get(job_id)
    if job is None:
        raise KeyError(f"no job {job_id!r}")
    return {"job": job.to_dict()}
