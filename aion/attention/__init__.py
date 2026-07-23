"""aion.attention — reusable attention primitives for AION.

Every component is a first-class ``Module`` that composes with the existing
``Linear``, ``EmbeddingLayer``, ``Sequential``, ``Trainer``, and ``ModelStore``
without modification.

Public surface
--------------
AttentionMask           — base class for additive mask objects
CausalMask              — lower-triangular mask for autoregressive models
PaddingMask             — per-batch mask for variable-length sequences
SinusoidalPE            — fixed sinusoidal positional encoding (no parameters)
LearnedPE               — learned positional encoding (trainable Parameter)
RotaryPE                — RoPE interface stub (implementation deferred)
ScaledDotProductAttention — the atomic attention operation
MultiHeadAttention      — multi-head wrapper with projection parameters
AttentionVisualizer     — pure analysis tool; produces JSON-serializable data
"""

from .mask import AttentionMask, CausalMask, PaddingMask
from .positional import LearnedPE, RotaryPE, SinusoidalPE
from .attention import MultiHeadAttention, ScaledDotProductAttention
from .visualizer import AttentionVisualizer

__all__ = [
    "AttentionMask", "CausalMask", "PaddingMask",
    "SinusoidalPE", "LearnedPE", "RotaryPE",
    "ScaledDotProductAttention", "MultiHeadAttention",
    "AttentionVisualizer",
]
