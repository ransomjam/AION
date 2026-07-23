"""Tokenization — the platform's text-to-ids input path.

Every model AION trains consumes integer token ids, and every inference request
produces them. This subpackage owns that boundary: normalization, tokenization,
and the token<->id vocabulary. A learned subword tokenizer (BPE) will be added
here and will reuse `normalize`/`tokenize` as its pre-tokenization step.
"""

from .text import normalize, strip_accents, tokenize
from .vocabulary import Vocabulary

__all__ = ["normalize", "strip_accents", "tokenize", "Vocabulary"]
