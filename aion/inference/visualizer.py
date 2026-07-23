"""GenerationVisualizer — analysis and visualization data for generation results.

All methods accept ``GenerationResult`` (or plain arrays) and return plain
Python dicts/lists.  No NumPy arrays in the output.  No rendering logic.

Follows the same pattern as ``AttentionVisualizer``: the backend computes,
the frontend renders.
"""

from __future__ import annotations

import math

import numpy as np

from .result import GenerationResult


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.sum(probs * np.log(probs + eps)))


class GenerationVisualizer:
    """Produce JSON-serializable visualization data from generation results."""

    # ── per-step token probabilities ──────────────────────────────────────────

    def token_probabilities(self, result: GenerationResult) -> list[dict]:
        """Per-step chosen-token probability.

        Returns
        -------
        list of ``{step, token_id, token_str, prob, entropy}`` dicts,
        one per generated token.
        """
        return [
            {
                "step": s.step,
                "token_id": s.token_id,
                "token_str": s.token_str,
                "prob": s.prob,
                "entropy": s.entropy,
            }
            for s in result.steps
        ]

    # ── entropy curve ─────────────────────────────────────────────────────────

    def entropy_curve(self, result: GenerationResult) -> list[float]:
        """Per-step Shannon entropy of the post-processing distribution."""
        return [s.entropy for s in result.steps]

    # ── top-k distribution at a given step ───────────────────────────────────

    def top_k_distribution(
        self,
        result: GenerationResult,
        step: int,
        k: int | None = None,
    ) -> list[dict]:
        """Top-k candidate distribution at a given generation step.

        Parameters
        ----------
        result:
            A completed ``GenerationResult``.
        step:
            0-based step index.
        k:
            Number of candidates to return.  Defaults to the number stored
            in the step (``GenerationConfig.top_candidates``).

        Returns
        -------
        list of ``{token_id, token_str, prob}`` dicts, sorted by prob desc.
        """
        if step < 0 or step >= len(result.steps):
            raise IndexError(f"step {step} out of range [0, {len(result.steps)})")
        candidates = result.steps[step].top_candidates
        if k is not None:
            candidates = candidates[:k]
        return list(candidates)

    # ── top-p (nucleus) distribution at a given step ──────────────────────────

    def top_p_distribution(
        self,
        result: GenerationResult,
        step: int,
        p: float = 0.9,
    ) -> list[dict]:
        """Nucleus distribution at a given step.

        Returns the smallest set of candidates whose cumulative probability
        exceeds ``p``, with cumulative probability annotated.

        Returns
        -------
        list of ``{token_id, token_str, prob, cumulative}`` dicts,
        sorted by prob desc.
        """
        if step < 0 or step >= len(result.steps):
            raise IndexError(f"step {step} out of range [0, {len(result.steps)})")
        candidates = result.steps[step].top_candidates
        cumulative = 0.0
        nucleus = []
        for c in candidates:
            cumulative += c["prob"]
            nucleus.append({**c, "cumulative": round(cumulative, 8)})
            if cumulative >= p:
                break
        return nucleus

    # ── generation timeline ───────────────────────────────────────────────────

    def generation_timeline(self, result: GenerationResult) -> list[dict]:
        """Token-by-token timing.

        Returns
        -------
        list of ``{step, token_id, token_str, elapsed_ms}`` dicts.
        """
        return [
            {
                "step": s.step,
                "token_id": s.token_id,
                "token_str": s.token_str,
                "elapsed_ms": s.elapsed_ms,
            }
            for s in result.steps
        ]

    # ── temperature effect ────────────────────────────────────────────────────

    def temperature_effect(
        self,
        logits: np.ndarray,
        temperatures: list[float] | None = None,
    ) -> list[dict]:
        """Show how temperature reshapes a logits distribution.

        Parameters
        ----------
        logits:
            Raw logits array of shape ``[vocab_size]``.
        temperatures:
            List of temperature values to evaluate.  Defaults to
            ``[0.1, 0.5, 1.0, 1.5, 2.0]``.

        Returns
        -------
        list of ``{temperature, entropy, top1_prob, top1_token_id}`` dicts.
        """
        if temperatures is None:
            temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
        logits = np.asarray(logits, dtype=float)
        results = []
        for t in temperatures:
            scaled = logits / t
            probs = _softmax(scaled)
            results.append({
                "temperature": t,
                "entropy": round(_entropy(probs), 6),
                "top1_prob": round(float(probs.max()), 6),
                "top1_token_id": int(np.argmax(probs)),
            })
        return results

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self, result: GenerationResult) -> dict:
        """High-level summary of a generation result.

        Returns
        -------
        dict with keys: ``prompt_tokens``, ``generated_tokens``,
        ``tokens_per_sec``, ``mean_entropy``, ``mean_top1_prob``,
        ``stopped_by``, ``generated_text``.
        """
        m = result.metrics
        return {
            "prompt_tokens": m.prompt_tokens,
            "generated_tokens": m.generated_tokens,
            "tokens_per_sec": m.tokens_per_sec,
            "mean_entropy": m.mean_entropy,
            "mean_top1_prob": m.mean_top1_prob,
            "stopped_by": m.stopped_by,
            "generated_text": result.generated_text,
        }
