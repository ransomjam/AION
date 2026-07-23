"""Dataset statistics — computed by streaming documents through the tokenizer.

One source of truth: tokenization and normalization come from
``aion.tokenization``; this module never re-implements them. The result is a
plain JSON-able dict suitable for caching and for the app's charts.
"""

from __future__ import annotations

from collections import Counter

from aion.tokenization import normalize, tokenize

TOP_TOKENS = 25
LENGTH_BINS = 12


def compute_statistics(dataset) -> dict:
    """Full statistics over ``dataset`` (anything with a ``stream()`` of docs)."""
    doc_token_lengths: list[int] = []
    doc_char_lengths: list[int] = []
    token_counts: Counter[str] = Counter()
    char_counts: Counter[str] = Counter()
    vocab_growth: list[dict] = []  # cumulative unique tokens as documents accrue
    seen_types: set[str] = set()

    total_docs = 0
    for _doc_id, text in dataset.stream():
        total_docs += 1
        tokens = tokenize(text)
        doc_token_lengths.append(len(tokens))
        doc_char_lengths.append(len(text))
        token_counts.update(tokens)
        char_counts.update(normalize(text))
        seen_types.update(tokens)
        vocab_growth.append({"documents": total_docs, "unique_tokens": len(seen_types)})

    total_tokens = sum(doc_token_lengths)
    unique_tokens = len(token_counts)

    return {
        "documents": total_docs,
        "characters": sum(doc_char_lengths),
        "tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": round(unique_tokens / total_tokens, 4) if total_tokens else 0.0,
        "document_length": _length_summary(doc_token_lengths, doc_char_lengths),
        "length_distribution": _histogram(doc_token_lengths, LENGTH_BINS),
        "top_tokens": [[tok, n] for tok, n in token_counts.most_common(TOP_TOKENS)],
        "vocabulary_growth": vocab_growth,
        "top_characters": _readable_chars(char_counts.most_common(20)),
    }


def _length_summary(token_lengths: list[int], char_lengths: list[int]) -> dict:
    if not token_lengths:
        return {"count": 0}
    return {
        "count": len(token_lengths),
        "avg_tokens": round(sum(token_lengths) / len(token_lengths), 2),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "avg_chars": round(sum(char_lengths) / len(char_lengths), 2),
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
    }


def _histogram(values: list[int], bins: int) -> list[dict]:
    """Bucket document lengths into ``bins`` ranges for the length distribution."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"range": f"{lo}", "count": len(values)}]
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return [
        {"range": f"{int(edges[i])}–{int(edges[i + 1])}", "count": counts[i]}
        for i in range(bins)
    ]


def _readable_chars(pairs: list[tuple[str, int]]) -> list[dict]:
    """Render character-frequency pairs with whitespace shown by name."""
    names = {" ": "space", "\n": "\\n", "\t": "\\t"}
    return [{"char": names.get(c, c), "count": n} for c, n in pairs]
