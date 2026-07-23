"""CausalLanguageModelLoss — next-token prediction loss for GPT.

Reshapes [batch, seq, vocab_size] logits and [batch, seq] targets into
[batch*seq, vocab_size] and [batch*seq] respectively, then applies
CrossEntropyLoss.  Teacher forcing: position i predicts token i+1, so
callers pass x = tokens[:-1] and y = tokens[1:] (handled by BatchSampler).
"""

from __future__ import annotations

import numpy as np

from aion.nn.loss import CrossEntropyLoss
from aion.nn.module import Module
from aion.nn.tensor import Tensor


class CausalLanguageModelLoss(Module):
    """Cross-entropy loss over all positions in a sequence batch.

    Parameters
    ----------
    logits:
        Shape ``[batch, seq, vocab_size]``.
    targets:
        Integer array of shape ``[batch, seq]``.  Not a ``Tensor``.

    Returns
    -------
    Scalar ``Tensor`` — mean cross-entropy over all (batch × seq) positions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ce = CrossEntropyLoss()

    def forward(self, logits: Tensor, targets) -> Tensor:  # type: ignore[override]
        targets = np.asarray(targets, dtype=np.intp)
        batch, seq, vocab = logits.data.shape
        # Flatten to [batch*seq, vocab_size] and [batch*seq]
        from aion.nn.ops import reshape
        flat_logits = reshape(logits, (batch * seq, vocab))
        flat_targets = targets.reshape(batch * seq)
        return self._ce(flat_logits, flat_targets)
