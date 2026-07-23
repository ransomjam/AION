"""Embeddings — trained embedding artifacts for AION projects.

Every embedding is a subclass of ``Embedding`` (``base.py``).
``EmbeddingStore`` (``store.py``) handles persistence.
``CBOWEmbedding`` (``cbow.py``) is the first implementation.
``linalg.py`` provides the pure-Python linear algebra primitives.
"""

from .base import Embedding, EmbeddingResult
from .cbow import CBOWEmbedding
from .store import EmbeddingNotFound, EmbeddingStore

__all__ = [
    "Embedding", "EmbeddingResult",
    "CBOWEmbedding",
    "EmbeddingStore", "EmbeddingNotFound",
]
