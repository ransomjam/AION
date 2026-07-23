"""GenerationConfig — serializable configuration for the inference pipeline.

All generation behaviour is controlled through this dataclass.  It is the
single source of truth for a generation call: strategy, sampling parameters,
stopping conditions, and output options.

``Generator`` constructs the logits-processor pipeline and sampler from this
config internally.  For testing or advanced use, samplers and processors can
also be injected directly into ``Generator``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GenerationConfig:
    # ── sampling strategy ─────────────────────────────────────────────────────
    strategy: str = "greedy"
    # "greedy" | "temperature" | "top_k" | "top_p"
    # When top_k or top_p are set alongside temperature, the ordering is:
    #   temperature → top_k → top_p → sample

    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled (keep all tokens)

    # ── logits processing ─────────────────────────────────────────────────────
    repetition_penalty: float = 1.0   # 1.0 = disabled

    # ── stopping ──────────────────────────────────────────────────────────────
    max_new_tokens: int = 128
    max_context_len: int = 512        # total context window (prompt + generated)
    stop_sequences: list[str] = field(default_factory=list)
    stop_on_eos: bool = True

    # ── output options ────────────────────────────────────────────────────────
    top_candidates: int = 20          # candidates stored per TokenStep

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        if self.strategy not in ("greedy", "temperature", "top_k", "top_p"):
            raise ValueError(
                f"strategy must be one of 'greedy', 'temperature', 'top_k', 'top_p'; "
                f"got {self.strategy!r}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.max_context_len <= 0:
            raise ValueError(f"max_context_len must be > 0, got {self.max_context_len}")
        if self.top_candidates <= 0:
            raise ValueError(f"top_candidates must be > 0, got {self.top_candidates}")
