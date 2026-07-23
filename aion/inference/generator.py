"""Generator — the main inference engine for decoder-only language models.

The generation loop is implemented once here.  Sampling strategy, logits
processing, and stopping conditions are all injected — the loop never
branches on strategy type.

Pipeline per step
-----------------
1. Forward pass: context → logits [1, seq, vocab]
2. Extract last-position logits: logits[0, -1, :]
3. LogitsProcessorList: temperature → top-k → top-p → repetition penalty
4. Sampler: greedy argmax or multinomial draw
5. Record TokenStep
6. Call StreamingCallback.on_token
7. Check StoppingCriteriaList → break if any fires
8. Append token to context; apply sliding window if needed

Context management
------------------
The context is a plain np.ndarray [1, current_len] grown by one token per
step.  When it reaches max_context_len, the oldest tokens are dropped from
the left (sliding window).  This is the correct extension point for KV cache:
replace the slice with incremental key/value reuse without changing the
generate() interface.

Sampler construction
--------------------
Generator builds the processor list and sampler from GenerationConfig
internally.  For testing or advanced use, processors and sampler can be
injected directly via the constructor.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .config import GenerationConfig
from .processors import (
    LogitsProcessorList,
    RepetitionPenaltyProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
)
from .result import GenerationMetrics, GenerationResult, TokenStep
from .sampler import GreedySampler, Sampler, StochasticSampler
from .stopping import (
    EosTokenCriteria,
    MaxNewTokensCriteria,
    StopSequenceCriteria,
    StoppingCriteriaList,
)
from .streaming import StreamingCallback


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.sum(probs * np.log(probs + eps)))


def _build_processors(cfg: GenerationConfig) -> LogitsProcessorList:
    """Construct the logits processor pipeline from a GenerationConfig."""
    pl = LogitsProcessorList()
    # Ordering: temperature → top-k → top-p → repetition penalty
    if cfg.strategy != "greedy" and cfg.temperature != 1.0:
        pl.append(TemperatureProcessor(cfg.temperature))
    if cfg.top_k > 0:
        pl.append(TopKProcessor(cfg.top_k))
    if cfg.top_p < 1.0:
        pl.append(TopPProcessor(cfg.top_p))
    if cfg.repetition_penalty != 1.0:
        pl.append(RepetitionPenaltyProcessor(cfg.repetition_penalty))
    return pl


def _build_sampler(cfg: GenerationConfig, rng: np.random.Generator) -> Sampler:
    """Construct the sampler from a GenerationConfig."""
    if cfg.strategy == "greedy":
        return GreedySampler()
    return StochasticSampler(rng=rng)


def _build_stopping(cfg: GenerationConfig, eos_id: int | None) -> StoppingCriteriaList:
    """Construct the stopping criteria list from a GenerationConfig."""
    sl = StoppingCriteriaList()
    sl.append(MaxNewTokensCriteria(cfg.max_new_tokens))
    if cfg.stop_on_eos and eos_id is not None:
        sl.append(EosTokenCriteria(eos_id))
    if cfg.stop_sequences:
        sl.append(StopSequenceCriteria(cfg.stop_sequences))
    return sl


class Generator:
    """Inference engine for a ``GPTModel``.

    Parameters
    ----------
    model:
        A trained ``GPTModel`` instance.  Set to eval mode before generating.
    tokenizer:
        A ``Tokenizer`` instance used to encode prompts and decode tokens.
    eos_token_id:
        Token id for end-of-sequence.  Used by ``EosTokenCriteria`` when
        ``GenerationConfig.stop_on_eos=True``.  If None, EOS stopping is
        disabled regardless of config.
    rng:
        NumPy random generator for stochastic sampling.
    processors:
        Optional pre-built ``LogitsProcessorList``.  If provided, overrides
        the processor pipeline built from ``GenerationConfig``.
    sampler:
        Optional pre-built ``Sampler``.  If provided, overrides the sampler
        built from ``GenerationConfig``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        processors: LogitsProcessorList | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self._rng = rng or np.random.default_rng()
        self._processors = processors   # None → built per call from config
        self._sampler = sampler         # None → built per call from config

    def generate(
        self,
        prompt: str | np.ndarray,
        config: GenerationConfig | None = None,
        *,
        callback: StreamingCallback | None = None,
    ) -> GenerationResult:
        """Run the generation loop and return a ``GenerationResult``.

        Parameters
        ----------
        prompt:
            Either a string (encoded via the tokenizer) or a pre-encoded
            integer array of shape ``[seq]`` or ``[1, seq]``.
        config:
            ``GenerationConfig`` controlling all generation behaviour.
            Defaults to ``GenerationConfig()`` (greedy, 128 tokens).
        callback:
            Optional ``StreamingCallback`` called after each token.
        """
        if config is None:
            config = GenerationConfig()
        config.validate()

        # ── encode prompt ─────────────────────────────────────────────────────
        if isinstance(prompt, str):
            prompt_str = prompt
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_str = ""
            arr = np.asarray(prompt, dtype=np.int32)
            prompt_ids = arr.flatten().tolist()

        # ── build pipeline (use injected or build from config) ────────────────
        processors = self._processors if self._processors is not None \
            else _build_processors(config)
        sampler = self._sampler if self._sampler is not None \
            else _build_sampler(config, self._rng)
        stopping = _build_stopping(config, self.eos_token_id)

        # ── context window: truncate prompt if needed ─────────────────────────
        budget = config.max_context_len - config.max_new_tokens
        if budget > 0 and len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]   # keep most recent tokens

        context = np.array(prompt_ids, dtype=np.int32)[np.newaxis, :]  # [1, seq]
        prompt_token_count = len(prompt_ids)

        # ── generation loop ───────────────────────────────────────────────────
        self.model.eval()
        generated_ids: list[int] = []
        steps: list[TokenStep] = []
        stopped_by = "max_new_tokens"
        t_start = time.monotonic()

        for step_idx in range(config.max_new_tokens):
            t_step = time.monotonic()

            # Forward pass — no gradient needed
            logits_tensor, _ = self.model(context)
            logits_np = logits_tensor.data[0, -1, :].copy()  # [vocab_size]

            # Logits processing
            logits_np = processors(logits_np, generated_ids)

            # Sampling
            token_id, probs = sampler.sample(logits_np)

            # Decode token
            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                token_str = f"<{token_id}>"

            # Top-N candidates
            top_n = min(config.top_candidates, len(probs))
            top_idx = np.argpartition(probs, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
            candidates = []
            for tid in top_idx:
                try:
                    tstr = self.tokenizer.decode([int(tid)])
                except Exception:
                    tstr = f"<{int(tid)}>"
                candidates.append({
                    "token_id": int(tid),
                    "token_str": tstr,
                    "prob": round(float(probs[tid]), 8),
                })

            elapsed_ms = (time.monotonic() - t_step) * 1000.0
            step_record = TokenStep(
                step=step_idx,
                token_id=token_id,
                token_str=token_str,
                prob=round(float(probs[token_id]), 8),
                entropy=round(_entropy(probs), 6),
                top_candidates=candidates,
                elapsed_ms=round(elapsed_ms, 3),
            )
            steps.append(step_record)
            generated_ids.append(token_id)

            # Streaming callback
            if callback is not None:
                callback.on_token(token_id, token_str, step_record)

            # Decode generated so far for stop-sequence checking
            try:
                decoded_so_far = self.tokenizer.decode(generated_ids)
            except Exception:
                decoded_so_far = ""

            # Stopping criteria
            if stopping(generated_ids, token_id, decoded_so_far):
                if config.stop_on_eos and token_id == self.eos_token_id:
                    stopped_by = "eos"
                elif config.stop_sequences and any(
                    decoded_so_far.endswith(s) for s in config.stop_sequences
                ):
                    stopped_by = "stop_sequence"
                else:
                    stopped_by = "max_new_tokens"
                break

            # Append to context and apply sliding window
            context = np.concatenate(
                [context, np.array([[token_id]], dtype=np.int32)], axis=1
            )
            if context.shape[1] > config.max_context_len:
                context = context[:, -config.max_context_len:]
                stopped_by = "max_context"

        # ── assemble result ───────────────────────────────────────────────────
        generation_time = time.monotonic() - t_start
        n_generated = len(generated_ids)

        try:
            generated_text = self.tokenizer.decode(generated_ids)
        except Exception:
            generated_text = "".join(s.token_str for s in steps)

        mean_entropy = float(np.mean([s.entropy for s in steps])) if steps else 0.0
        mean_top1 = float(np.mean([s.prob for s in steps])) if steps else 0.0
        tps = n_generated / generation_time if generation_time > 0 else 0.0

        metrics = GenerationMetrics(
            prompt_tokens=prompt_token_count,
            generated_tokens=n_generated,
            total_tokens=prompt_token_count + n_generated,
            generation_time_s=round(generation_time, 4),
            tokens_per_sec=round(tps, 2),
            mean_entropy=round(mean_entropy, 6),
            mean_top1_prob=round(mean_top1, 6),
            stopped_by=stopped_by,
        )

        result = GenerationResult(
            prompt=prompt_str,
            generated_text=generated_text,
            full_text=prompt_str + generated_text,
            generated_ids=generated_ids,
            steps=steps,
            metrics=metrics,
            config=config.to_dict(),
        )

        if callback is not None:
            callback.on_complete(result)

        return result
