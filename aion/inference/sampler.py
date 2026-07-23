"""Samplers — select the next token from a processed logits array.

A ``Sampler`` receives logits that have already been processed by the
``LogitsProcessorList`` (temperature scaled, top-k/top-p filtered, etc.)
and returns the chosen token id plus the full probability distribution.

The sampler does NOT apply any logits processing itself.  That separation
is the approved design: processors modify distributions; samplers select tokens.

Hierarchy
---------
Sampler (ABC)
    GreedySampler       — argmax; deterministic
    StochasticSampler   — multinomial draw from softmax probabilities

``GreedySampler`` ignores the probability distribution shape and always picks
the highest-logit token.  ``StochasticSampler`` draws from the distribution,
making it the base for temperature / top-k / top-p sampling (which are all
achieved by the processors applied before this sampler is called).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


class Sampler(ABC):
    """Abstract base for all token samplers."""

    @abstractmethod
    def sample(self, logits: np.ndarray) -> tuple[int, np.ndarray]:
        """Select the next token.

        Parameters
        ----------
        logits:
            Processed logits of shape ``[vocab_size]``.  May contain ``-inf``
            for filtered tokens.

        Returns
        -------
        token_id:
            The selected token index.
        probs:
            Full probability distribution ``[vocab_size]`` after softmax.
            Filtered positions have probability 0.
        """


class GreedySampler(Sampler):
    """Deterministic argmax sampling.

    Always selects the token with the highest logit.  Ignores the shape of
    the distribution — temperature, top-k, and top-p processors have no
    effect when this sampler is used.
    """

    def sample(self, logits: np.ndarray) -> tuple[int, np.ndarray]:
        probs = _softmax(logits)
        return int(np.argmax(probs)), probs


class StochasticSampler(Sampler):
    """Multinomial sampling from the softmax distribution.

    Used for temperature, top-k, and top-p sampling.  The distribution shape
    is determined entirely by the processors applied before this sampler.

    Parameters
    ----------
    rng:
        NumPy random generator.  Pass a seeded generator for reproducibility.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def sample(self, logits: np.ndarray) -> tuple[int, np.ndarray]:
        probs = _softmax(logits)
        token_id = int(self._rng.choice(len(probs), p=probs))
        return token_id, probs
