"""Logits processors — modify the logits distribution before sampling.

A ``LogitsProcessor`` takes a raw logits array and the list of already-generated
token ids, and returns a modified logits array.  Processors are applied
sequentially in a ``LogitsProcessorList``.

This separation is the approved design decision: processors modify distributions;
samplers select tokens.  The pipeline is:

    raw logits
        → LogitsProcessorList (temperature, top-k filter, top-p filter,
                               repetition penalty, ...)
        → Sampler (greedy argmax or multinomial draw)
        → next token id

Extension points for future features:
    - Bad-word filtering: zero out forbidden token ids
    - Grammar constraints: mask tokens that violate a grammar
    - Logit biasing: add per-token bias values (OpenAI logit_bias)
    - Watermarking: bias toward a pseudo-random token subset
    - Constrained decoding: enforce structural output formats
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_NEG_INF = -1e9


class LogitsProcessor(ABC):
    """Base class for all logits processors."""

    @abstractmethod
    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        """Return modified logits.

        Parameters
        ----------
        logits:
            Raw logits array of shape ``[vocab_size]``.  Must not be mutated
            in place — return a new array or a copy.
        generated_ids:
            Token ids generated so far in this call (not including the prompt).
        """


class TemperatureProcessor(LogitsProcessor):
    """Divide logits by temperature before softmax.

    temperature < 1  → sharper distribution (more deterministic)
    temperature > 1  → flatter distribution (more random)
    temperature = 1  → identity
    """

    def __init__(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        return logits / self.temperature


class TopKProcessor(LogitsProcessor):
    """Zero out all but the top-k logits (set to -inf before softmax).

    k = 0 disables filtering (all tokens kept).
    """

    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = k

    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        if self.k == 0 or self.k >= len(logits):
            return logits
        # Keep only the top-k indices; set the rest to -inf.
        threshold = np.partition(logits, -self.k)[-self.k]
        result = logits.copy()
        result[result < threshold] = _NEG_INF
        return result


class TopPProcessor(LogitsProcessor):
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability (after softmax) exceeds p.

    p = 1.0 disables filtering (all tokens kept).
    """

    def __init__(self, p: float) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        self.p = p

    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        if self.p >= 1.0:
            return logits
        # Compute softmax probabilities for nucleus selection.
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()

        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        # Find the cutoff: first index where cumsum >= p.
        cutoff = int(np.searchsorted(cumsum, self.p)) + 1
        nucleus = sorted_idx[:cutoff]

        result = np.full_like(logits, _NEG_INF)
        result[nucleus] = logits[nucleus]
        return result


class RepetitionPenaltyProcessor(LogitsProcessor):
    """Penalise tokens that have already appeared in the generated sequence.

    Formulation from Keskar et al. (2019):
        if logit > 0: logit /= penalty
        else:         logit *= penalty

    penalty = 1.0 is identity (no effect).
    penalty > 1.0 discourages repetition.
    """

    def __init__(self, penalty: float) -> None:
        if penalty <= 0.0:
            raise ValueError(f"penalty must be > 0, got {penalty}")
        self.penalty = penalty

    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        if self.penalty == 1.0 or not generated_ids:
            return logits
        result = logits.copy()
        for token_id in set(generated_ids):
            if 0 <= token_id < len(result):
                if result[token_id] > 0:
                    result[token_id] /= self.penalty
                else:
                    result[token_id] *= self.penalty
        return result


class LogitsProcessorList:
    """Applies a sequence of ``LogitsProcessor`` instances in order.

    Processors are applied left-to-right.  The output of each processor is
    the input to the next.  An empty list is the identity.
    """

    def __init__(self, processors: list[LogitsProcessor] | None = None) -> None:
        self._processors: list[LogitsProcessor] = processors or []

    def append(self, processor: LogitsProcessor) -> None:
        self._processors.append(processor)

    def __call__(self, logits: np.ndarray, generated_ids: list[int]) -> np.ndarray:
        for proc in self._processors:
            logits = proc(logits, generated_ids)
        return logits

    def __len__(self) -> int:
        return len(self._processors)
