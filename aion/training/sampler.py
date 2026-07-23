"""SampleGenerator — generate text samples from a GPTModel during training.

Wraps the inference ``Generator`` so that TrainingProject can call it at
epoch boundaries without depending on the inference package directly.

SampleResult
    prompt          str
    generated_text  str
    full_text       str
    tokens          int
    elapsed_s       float

SampleGenerator.generate_all(prompts) -> list[SampleResult]
    Runs each prompt through the model and returns the results.
    Called from a TrainingCallback at on_epoch_end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SampleResult:
    prompt: str
    generated_text: str
    full_text: str
    tokens: int
    elapsed_s: float


class SampleGenerator:
    """Generate text samples from a GPTModel.

    Parameters
    ----------
    model:
        A ``GPTModel`` instance.
    tokenizer:
        A tokenizer with ``encode(text) -> list[int]`` and
        ``decode(ids) -> str``.
    max_new_tokens:
        Maximum tokens to generate per prompt.
    strategy:
        Sampling strategy passed to ``GenerationConfig``.
        ``"greedy"`` is deterministic and reproducible.
    temperature:
        Sampling temperature (ignored for greedy).
    eos_token_id:
        Token id for end-of-sequence stopping.  ``None`` disables EOS stop.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int = 64,
        strategy: str = "greedy",
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.strategy = strategy
        self.temperature = temperature
        self.eos_token_id = eos_token_id

    def generate(self, prompt: str) -> SampleResult:
        """Generate text for a single prompt."""
        from aion.inference.config import GenerationConfig
        from aion.inference.generator import Generator

        cfg = GenerationConfig(
            strategy=self.strategy,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stop_on_eos=self.eos_token_id is not None,
        )
        gen = Generator(
            self.model,
            self.tokenizer,
            eos_token_id=self.eos_token_id,
        )
        t0 = time.monotonic()
        result = gen.generate(prompt, cfg)
        elapsed = round(time.monotonic() - t0, 3)
        return SampleResult(
            prompt=prompt,
            generated_text=result.generated_text,
            full_text=result.full_text,
            tokens=result.metrics.generated_tokens,
            elapsed_s=elapsed,
        )

    def generate_all(self, prompts: list[str]) -> list[SampleResult]:
        """Generate samples for every prompt in the list."""
        return [self.generate(p) for p in prompts]
