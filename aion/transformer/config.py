"""Configuration objects for transformer architectures.

``TransformerConfig`` is the canonical hyperparameter container.  It encodes
every architectural choice as a typed, named field with a sensible default.
Future model families (GPT, BERT, ViT, multimodal) subclass or compose these
configs without modifying the core transformer implementation.

Design principles
-----------------
- Configs are plain dataclasses — no logic, no Module dependencies.
- Every field has a default so configs can be constructed incrementally.
- ``to_dict()`` / ``from_dict()`` round-trip through JSON for storage in
  ``architecture.json`` alongside ``weights.npz``.
- ``EncoderConfig`` and ``DecoderConfig`` are thin specialisations that set
  sensible defaults for their respective architectures and expose only the
  fields relevant to each.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TransformerConfig:
    """Hyperparameters for a full encoder-decoder Transformer.

    Parameters
    ----------
    d_model:
        Model dimensionality.  All sub-layers operate in this space.
    n_heads:
        Number of attention heads.  Must divide ``d_model`` evenly.
    n_encoder_layers:
        Number of encoder blocks stacked.
    n_decoder_layers:
        Number of decoder blocks stacked.
    d_ff:
        Feed-forward inner dimensionality.  Typically ``4 * d_model``.
    dropout:
        Dropout probability applied after attention and FFN sub-layers.
        Set to ``0.0`` to disable.
    activation:
        FFN activation function.  ``"relu"`` or ``"gelu"``.
    pre_norm:
        If ``True`` (default), apply LayerNorm before each sub-layer
        (pre-norm / RMS-norm style).  If ``False``, apply after (original
        Vaswani post-norm).
    max_seq_len:
        Maximum sequence length the positional encoding is precomputed for.
    """
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    activation: str = "relu"
    pre_norm: bool = True
    max_seq_len: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        """Raise ``ValueError`` for any invalid combination of fields."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {self.activation!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class EncoderConfig:
    """Hyperparameters for an encoder-only Transformer (e.g. BERT-style).

    Identical fields to ``TransformerConfig`` minus decoder-specific ones.
    ``n_decoder_layers`` is always 0 for encoder-only models.
    """
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    activation: str = "relu"
    pre_norm: bool = True
    max_seq_len: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {self.activation!r}"
            )

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to a full ``TransformerConfig`` with zero decoder layers."""
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_layers,
            n_decoder_layers=0,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            pre_norm=self.pre_norm,
            max_seq_len=self.max_seq_len,
        )


@dataclass
class DecoderConfig:
    """Hyperparameters for a decoder-only Transformer (e.g. GPT-style).

    Decoder-only models use encoder blocks with a causal mask — there is no
    cross-attention because there is no encoder output.  ``n_encoder_layers``
    is always 0 for decoder-only models.
    """
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.0
    activation: str = "gelu"
    pre_norm: bool = True
    max_seq_len: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {self.activation!r}"
            )

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to a full ``TransformerConfig`` with zero encoder layers."""
        return TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=0,
            n_decoder_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            pre_norm=self.pre_norm,
            max_seq_len=self.max_seq_len,
        )
