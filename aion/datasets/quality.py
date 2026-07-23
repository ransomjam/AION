"""Dataset quality analysis — dependency-free issue detection over documents.

Flags the problems that hurt training data: empty and whitespace-only documents,
exact duplicates, extremely short/long documents, and reports the character
composition, how much of the corpus is already Unicode-normalized, and a coarse
dependency-free language hint. JSON-able output, suitable for caching.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from aion.tokenization import normalize
from aion.util import sha256_text

SHORT_CHARS = 10        # documents at or below this many chars are "very short"
LONG_CHARS = 20_000     # documents at or above this are "very long"

# Tiny closed-class word lists for a coarse en/fr hint (no external dependency).
_EN = {"the", "and", "of", "to", "in", "is", "you", "that", "it", "for", "on", "with"}
_FR = {"le", "la", "les", "de", "et", "un", "une", "que", "vous", "est", "pour", "dans"}


def compute_quality(dataset) -> dict:
    empty = []
    whitespace_only = []
    very_short = []
    very_long = []
    hashes: dict[str, list[int]] = {}
    categories: Counter[str] = Counter()
    already_normalized = 0
    lang_votes: Counter[str] = Counter()
    total = 0

    for doc_id, text in dataset.stream():
        total += 1
        if len(text) == 0:
            empty.append(doc_id)
        elif text.strip() == "":
            whitespace_only.append(doc_id)
        if len(text) <= SHORT_CHARS:
            very_short.append(doc_id)
        if len(text) >= LONG_CHARS:
            very_long.append(doc_id)

        hashes.setdefault(sha256_text(text), []).append(doc_id)
        for ch in text:
            categories[_category(ch)] += 1
        if normalize(text) == text:
            already_normalized += 1
        lang_votes[_language_hint(text)] += 1

    duplicates = [
        {"count": len(ids), "documents": ids}
        for ids in hashes.values() if len(ids) > 1
    ]

    return {
        "documents": total,
        "issues": {
            "empty": empty,
            "whitespace_only": whitespace_only,
            "very_short": very_short,
            "very_long": very_long,
            "duplicate_groups": duplicates,
            "duplicate_documents": sum(len(g["documents"]) for g in duplicates),
        },
        "character_categories": dict(categories),
        "normalization": {
            "already_normalized": already_normalized,
            "needs_normalization": total - already_normalized,
        },
        "language_hint": dict(lang_votes),
    }


def _category(ch: str) -> str:
    """Coarse character class for composition reporting."""
    if ch.isspace():
        return "whitespace"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "letter"
    if cat.startswith("N"):
        return "digit"
    if cat.startswith("P"):
        return "punctuation"
    if cat.startswith("S"):
        return "symbol"
    return "other"


def _language_hint(text: str) -> str:
    """Coarse, dependency-free language guess from closed-class word overlap.

    Not a real language identifier — a hint for triage. Returns 'en', 'fr', or
    'unknown'. Non-Latin-dominant text is reported as 'unknown' rather than
    guessed wrongly.
    """
    words = normalize(text).split()
    if not words:
        return "unknown"
    en = sum(w in _EN for w in words)
    fr = sum(w in _FR for w in words)
    if en == 0 and fr == 0:
        return "unknown"
    return "en" if en >= fr else "fr"
