"""Text processing from first principles.

Before a model can learn anything from language, raw text must become a clean,
consistent sequence of discrete units. Two humans typing "Café" and "café"
(an e followed by a combining acute accent) mean the same word, but as byte or
codepoint sequences they are completely different. If we skip this step, the
model wastes capacity learning that these are "the same", or — worse — never
learns it at all.

This module does that first, unglamorous, essential job, using only the standard
library so nothing is hidden:

    normalize      -> canonical, case-folded Unicode text
    strip_accents  -> remove combining marks (an optional, lossy step)
    tokenize       -> split normalized text into word / number / punctuation units

The design goal is transparency, not raw speed. Every choice is explained where
it is made.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize", "strip_accents", "tokenize"]


def strip_accents(text: str) -> str:
    """Remove combining marks (accents, diacritics) from ``text``.

    The trick is to first decompose each character into its base letter plus its
    combining marks (NFD — Normalization Form Decomposed), then drop every
    character in Unicode category ``Mn`` ("Mark, nonspacing"). After decomposition
    "é" (é) becomes "e" + "́" (combining acute), and we keep only the "e".

    This is deliberately *lossy* — it throws away information — so ``normalize``
    leaves it off by default. It is useful when accents are noise for the task
    (e.g. matching user typos), and harmful when they carry meaning (French
    "ou"/"où", or any language where accents are not optional).
    """
    decomposed = unicodedata.normalize("NFD", text)
    kept = [ch for ch in decomposed if unicodedata.category(ch) != "Mn"]
    return "".join(kept)


def normalize(
    text: str,
    *,
    form: str = "NFKC",
    casefold: bool = True,
    accents: bool = True,
) -> str:
    """Return a canonical form of ``text`` for consistent downstream processing.

    Steps, in order:

    1. **Unicode normalization** (``form``). Unicode lets the same visible text be
       encoded in different ways; normalization picks one canonical encoding so
       equal-looking strings compare equal. We default to **NFKC**:
         - *NFC* composes characters to their canonical single-codepoint form.
         - the *K* ("compatibility") also folds visually/­semantically equivalent
           variants — e.g. the ligature "ﬁ" (ﬁ) becomes "fi", and a full-width
           digit "１" becomes "1". This is usually what we want for NLP because
           it collapses cosmetic distinctions. Pass ``form="NFC"`` to keep those
           distinctions.

    2. **Case folding** (``casefold``). ``str.casefold`` is a more aggressive,
       language-aware cousin of ``str.lower``: it maps "ß" (ß) to "ss" and
       handles scripts ``lower`` mishandles, which matters once we leave English.

    3. **Accent stripping** (``accents``). When ``True`` (the default), combining
       marks are removed via :func:`strip_accents`. Note this is lossy; disable it
       when diacritics are meaningful for your language or task.

    4. **Whitespace collapse.** Leading/trailing whitespace is removed and every
       internal run of whitespace becomes a single space, so tokenization does not
       have to reason about tabs, newlines, or double spaces.

    ``normalize`` is idempotent under a fixed set of options: normalizing an
    already-normalized string returns it unchanged. The tests rely on this.
    """
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"unknown normalization form: {form!r}")

    text = unicodedata.normalize(form, text)
    if casefold:
        text = text.casefold()
    if accents:
        text = strip_accents(text)
        # Accent stripping decomposes then filters, which can leave the string in
        # a decomposed state; re-apply the requested form to stay canonical.
        text = unicodedata.normalize(form, text)
    text = " ".join(text.split())
    return text


def tokenize(text: str, *, normalize_first: bool = True) -> list[str]:
    """Split ``text`` into a flat list of word, number, and punctuation tokens.

    This is a deterministic, rule-based tokenizer — the kind that precedes learned
    subword tokenizers like BPE (which arrives in a later module and will build on
    this one). The rules, applied character by character:

    - **Letters and digits stay together** into word/number tokens. "Runs" of
      alphanumeric characters form one token, so ``momo123`` is a single token but
      ``pay now`` is two.
    - **Each punctuation/symbol character is its own token.** Keeping "!" and "?"
      as tokens (rather than discarding them) preserves signal — punctuation
      carries meaning — and keeps the process reversible enough to inspect.
    - **Whitespace separates tokens and is itself discarded.**

    "Alphanumeric" is judged with :meth:`str.isalnum`, which is Unicode-aware:
    accented letters, non-Latin scripts, and digits from other numeral systems all
    count as word characters. That is why normalization runs first by default —
    so ``café`` and ``cafe`` tokenize the same way.

    Returns an empty list for empty or whitespace-only input.
    """
    if normalize_first:
        text = normalize(text)

    tokens: list[str] = []
    current: list[str] = []  # characters accumulating into the current word token

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for ch in text:
        if ch.isspace():
            flush()
        elif ch.isalnum():
            current.append(ch)
        else:
            # Punctuation or symbol: end any word in progress, emit this char alone.
            flush()
            tokens.append(ch)
    flush()
    return tokens
