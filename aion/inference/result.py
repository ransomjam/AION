"""Generation output data structures.

TokenStep
    Everything recorded at a single generation step: the chosen token,
    its probability, the top-N candidates, and the distribution entropy.

GenerationMetrics
    Aggregate statistics over the full generation call.

GenerationResult
    The complete output of a ``Generator.generate()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenStep:
    """Data recorded at one generation step.

    Parameters
    ----------
    step:
        0-based step index.
    token_id:
        The selected token id.
    token_str:
        The decoded string for this token.
    prob:
        Probability of the selected token under the post-processing distribution.
    entropy:
        Shannon entropy of the full distribution: ``-sum(p * log(p + eps))``.
    top_candidates:
        List of ``{token_id, token_str, prob}`` dicts for the top-N candidates
        (N = ``GenerationConfig.top_candidates``).  Sorted by probability
        descending.
    elapsed_ms:
        Wall-clock time for this step in milliseconds.
    """
    step: int
    token_id: int
    token_str: str
    prob: float
    entropy: float
    top_candidates: list[dict]   # [{token_id, token_str, prob}]
    elapsed_ms: float


@dataclass
class GenerationMetrics:
    """Aggregate statistics for a completed generation call."""
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    generation_time_s: float
    tokens_per_sec: float
    mean_entropy: float
    mean_top1_prob: float
    stopped_by: str              # "max_new_tokens" | "stop_sequence" | "eos" | "max_context"


@dataclass
class GenerationResult:
    """Complete output of a ``Generator.generate()`` call.

    Parameters
    ----------
    prompt:
        The original prompt string (or empty string if a token array was passed).
    generated_text:
        The decoded generated text (not including the prompt).
    full_text:
        prompt + generated_text.
    generated_ids:
        List of generated token ids.
    steps:
        Per-step data (one ``TokenStep`` per generated token).
    metrics:
        Aggregate generation statistics.
    config:
        The ``GenerationConfig`` used for this call (as a dict for
        serializability).
    """
    prompt: str
    generated_text: str
    full_text: str
    generated_ids: list[int]
    steps: list[TokenStep]
    metrics: GenerationMetrics
    config: dict
