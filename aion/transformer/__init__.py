"""aion.transformer — reusable Transformer framework for AION.

Every component is a first-class ``Module`` that composes with the existing
autograd engine, attention primitives, optimizers, Trainer, and stores
without modification.

Public surface
--------------
TransformerConfig       — hyperparameter container for all transformer variants
EncoderConfig           — encoder-only configuration (BERT-style)
DecoderConfig           — decoder-only configuration (GPT-style)
LayerNorm               — fused layer normalization with learned scale/shift
Dropout                 — training-time dropout; identity at inference
FeedForward             — position-wise FFN with ReLU or GELU activation
TransformerBlock        — abstract base for all transformer block variants
TransformerEncoderBlock — one encoder layer (self-attn + FFN + residuals)
TransformerDecoderBlock — one decoder layer (self-attn + cross-attn + FFN)
TransformerStack        — reusable stack of N encoder blocks + final norm
TransformerEncoder      — encoder-only stack (BERT-style or seq2seq encoder)
TransformerDecoder      — cross-attention decoder stack (seq2seq decoder)
Transformer             — full encoder-decoder model
TransformerVisualizer   — pure analysis tool; produces JSON-serializable data
TransformerStore        — save / load / list transformer models
TransformerNotFound     — raised when a model_id does not exist
"""

from .config import DecoderConfig, EncoderConfig, TransformerConfig
from .norm import LayerNorm
from .dropout import Dropout
from .ffn import FeedForward
from .block import TransformerBlock, TransformerDecoderBlock, TransformerEncoderBlock
from .stack import TransformerStack
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .model import Transformer
from .visualizer import TransformerVisualizer
from .store import TransformerNotFound, TransformerStore

__all__ = [
    "TransformerConfig", "EncoderConfig", "DecoderConfig",
    "LayerNorm", "Dropout", "FeedForward",
    "TransformerBlock", "TransformerEncoderBlock", "TransformerDecoderBlock",
    "TransformerStack",
    "TransformerEncoder", "TransformerDecoder",
    "Transformer",
    "TransformerVisualizer",
    "TransformerStore", "TransformerNotFound",
]
