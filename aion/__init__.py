"""AION — an independent AI research and engineering platform.

Top-level package for the infrastructure we use to collect data, train models,
evaluate them, and iterate. Subpackages are capabilities, not lessons:

    aion.tokenization   text -> tokens -> ids (the training/inference input path)

Further subpackages (data, training, eval, inference, registry, ...) are added
when an actual model-building need requires them, never speculatively.
"""

__version__ = "0.0.0"
