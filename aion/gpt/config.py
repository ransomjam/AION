"""GPTConfig — hyperparameter container for GPT-style language models.

Standalone dataclass: not a subclass of DecoderConfig.  A bridge method
``to_decoder_config()`` converts to ``DecoderConfig`` when the transformer
stack needs to be constructed via the transformer config hierarchy.

Fields mirror DecoderConfig but add language-model-specific knobs:
``vocab_size``, ``max_seq_len``, and ``tie_weights``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    max_seq_len: int = 512
    activation: str = "gelu"
    pre_norm: bool = True
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {self.activation!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    def to_decoder_config(self):
        """Return a ``DecoderConfig`` with matching architectural fields."""
        from aion.transformer.config import DecoderConfig
        return DecoderConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            pre_norm=self.pre_norm,
            max_seq_len=self.max_seq_len,
        )
