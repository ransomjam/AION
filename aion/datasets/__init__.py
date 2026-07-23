"""Dataset engineering library — project-scoped, the root of every pipeline.

A dataset is a directory of raw UTF-8 documents plus a small metadata file,
living under a project's ``data/`` directory. This package owns storage and the
read-only analyses over it:

    store.py    DatasetStore + Dataset: create, import, document CRUD, stream
    stats.py    statistics (reuses aion.tokenization — one tokenizer, everywhere)
    quality.py  data-quality report
    search.py   substring + regex search

Derived output (stats, quality) is written to the project cache by the app layer,
never back into the dataset — originals stay immutable.
"""

from .store import Dataset, DatasetStore
from .stats import compute_statistics
from .quality import compute_quality
from .search import search_dataset

__all__ = [
    "Dataset", "DatasetStore", "compute_statistics", "compute_quality",
    "search_dataset",
]
