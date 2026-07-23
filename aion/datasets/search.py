"""Search over a dataset's documents — substring or regex, case-optional.

A linear scan today (correct and simple for current scale). When datasets grow
large enough that scans hurt, an inverted index becomes a cache namespace behind
the same function signature — callers won't change.
"""

from __future__ import annotations

import re

SNIPPET_RADIUS = 40  # characters of context shown on each side of a match
DEFAULT_MAX_RESULTS = 200


def search_dataset(dataset, query: str, *, regex: bool = False,
                   case_sensitive: bool = False,
                   max_results: int = DEFAULT_MAX_RESULTS) -> dict:
    """Find ``query`` across all documents, returning matches with snippets."""
    if not query:
        return {"query": query, "regex": regex, "case_sensitive": case_sensitive,
                "match_count": 0, "matches": [], "truncated": False}

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags)

    matches: list[dict] = []
    truncated = False
    for doc_id, text in dataset.stream():
        for m in pattern.finditer(text):
            if len(matches) >= max_results:
                truncated = True
                break
            matches.append({
                "doc_id": doc_id,
                "offset": m.start(),
                "snippet": _snippet(text, m.start(), m.end()),
            })
        if truncated:
            break

    return {
        "query": query, "regex": regex, "case_sensitive": case_sensitive,
        "match_count": len(matches), "matches": matches, "truncated": truncated,
    }


def _snippet(text: str, start: int, end: int) -> str:
    a = max(0, start - SNIPPET_RADIUS)
    b = min(len(text), end + SNIPPET_RADIUS)
    prefix = "…" if a > 0 else ""
    suffix = "…" if b < len(text) else ""
    return (prefix + text[a:b] + suffix).replace("\n", " ")
