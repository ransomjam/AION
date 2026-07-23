"""Lab Registry — the single source of truth for the application's navigation.

Each laboratory registers its identity, scope, and status here; the frontend
renders navigation from this registry via ``GET /api/labs`` and never hardcodes
the list. Adding a lab is adding one entry (plus its panel renderer and, if it
has one, its API + capability functions).

Scope:
    home     the Project Explorer — the launch screen, always available
    scratch  usable without a project; must never write project storage
    project  operates inside the currently-open project; requires one

Status:
    active       implemented and usable
    coming_soon  reserved slot, shown disabled so the roadmap is visible
"""

from __future__ import annotations

from aion import __version__


class Scope:
    HOME = "home"
    SCRATCH = "scratch"
    PROJECT = "project"


class Status:
    ACTIVE = "active"
    COMING_SOON = "coming_soon"


# Ordered registry. `order` is explicit so navigation is stable and intentional.
_LABS: list[dict] = [
    {"id": "projects", "name": "Project Explorer", "icon": "📁", "order": 0,
     "scope": Scope.HOME, "status": Status.ACTIVE,
     "description": "Choose, create, and open projects. Your home base."},

    {"id": "tokenizer", "name": "Tokenizer Lab", "icon": "✂️", "order": 10,
     "scope": Scope.SCRATCH, "status": Status.ACTIVE,
     "description": "Normalize and tokenize text. Disposable scratch tool."},
    {"id": "vocabulary", "name": "Vocabulary Lab", "icon": "🔤", "order": 11,
     "scope": Scope.SCRATCH, "status": Status.ACTIVE,
     "description": "Build and inspect a vocabulary. Disposable scratch tool."},

    {"id": "data", "name": "Data Lab", "icon": "🗄️", "order": 20,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Import, inspect, search, and analyze datasets in a project."},

    {"id": "bpe", "name": "BPE Lab", "icon": "🧩", "order": 30,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Train, inspect, and visualize BPE tokenizers."},
    {"id": "embedding", "name": "Embedding Lab", "icon": "🧭", "order": 31,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Train and explore Word2Vec CBOW embeddings."},
    {"id": "training", "name": "Training Lab", "icon": "🏋️", "order": 32,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Train neural network models and watch the loss curves."},
    {"id": "attention", "name": "Attention Lab", "icon": "🔍", "order": 33,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Inspect attention weights, heatmaps, and head behaviour."},
    {"id": "transformer", "name": "Transformer Lab", "icon": "🤖", "order": 34,
     "scope": Scope.PROJECT, "status": Status.ACTIVE,
     "description": "Build and inspect transformer encoder/decoder architectures."},
    {"id": "evaluation", "name": "Evaluation Lab", "icon": "📊", "order": 35,
     "scope": Scope.PROJECT, "status": Status.COMING_SOON,
     "description": "Evaluate and compare models."},
    {"id": "inference", "name": "Inference Lab", "icon": "💬", "order": 36,
     "scope": Scope.PROJECT, "status": Status.COMING_SOON,
     "description": "Run and converse with trained models."},
]


def registry() -> dict:
    """Return the full registry for the frontend to render navigation from."""
    labs = sorted(_LABS, key=lambda l: l["order"])
    return {"platform": "AION", "version": __version__, "labs": labs}


def get(lab_id: str) -> dict | None:
    return next((l for l in _LABS if l["id"] == lab_id), None)
