"""Packed-sequence data utilities for GPT training.

Strategy: sequence packing (no padding).
- All token ids are concatenated into a single flat array, documents separated
  by EOS tokens.
- The flat array is sliced into fixed-length blocks of size ``block_size + 1``.
  The extra token is the label for the last position.
- Blocks may span document boundaries — the model learns to predict across them.
- ``BatchSampler`` groups block indices into batches, optionally shuffled.

This is the standard GPT pre-training data strategy (Brown et al., 2020).
"""

from __future__ import annotations

import numpy as np


class TokenizedDataset:
    """Flat packed token array sliced into fixed-length blocks.

    Parameters
    ----------
    token_ids:
        1-D array of integer token ids.  Documents should already be
        separated by EOS tokens before passing here.
    block_size:
        Context length.  Each sample is ``block_size + 1`` tokens:
        ``x = block[:block_size]``, ``y = block[1:]``.
    """

    def __init__(self, token_ids: np.ndarray, block_size: int) -> None:
        self.block_size = block_size
        tokens = np.asarray(token_ids, dtype=np.int32)
        # Trim to a multiple of (block_size + 1) so every block is full.
        n_blocks = len(tokens) // (block_size + 1)
        if n_blocks == 0:
            raise ValueError(
                f"token_ids length {len(tokens)} is too short for block_size {block_size}; "
                f"need at least {block_size + 1} tokens"
            )
        tokens = tokens[: n_blocks * (block_size + 1)]
        # Shape: [n_blocks, block_size + 1]
        self._blocks = tokens.reshape(n_blocks, block_size + 1)

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) where x = block[:-1], y = block[1:]."""
        block = self._blocks[idx]
        return block[:-1], block[1:]

    @staticmethod
    def pack(
        documents: list[list[int]],
        block_size: int,
        eos_id: int,
    ) -> "TokenizedDataset":
        """Concatenate documents with EOS separators and build a dataset.

        Parameters
        ----------
        documents:
            List of token-id lists, one per document.
        block_size:
            Context length for each training block.
        eos_id:
            Token id used as document separator.
        """
        flat: list[int] = []
        for doc in documents:
            flat.extend(doc)
            flat.append(eos_id)
        return TokenizedDataset(np.array(flat, dtype=np.int32), block_size)


class BatchSampler:
    """Yields batches of (x, y) arrays from a ``TokenizedDataset``.

    Parameters
    ----------
    dataset:
        Source ``TokenizedDataset``.
    batch_size:
        Number of blocks per batch.
    shuffle:
        If ``True``, block order is shuffled each time ``__iter__`` is called.
    rng:
        NumPy random generator for shuffling.  Pass a seeded generator for
        reproducibility.
    """

    def __init__(
        self,
        dataset: TokenizedDataset,
        batch_size: int,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = rng or np.random.default_rng()

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start: start + self.batch_size]
            xs, ys = [], []
            for i in batch_idx:
                x, y = self.dataset[int(i)]
                xs.append(x)
                ys.append(y)
            yield np.stack(xs), np.stack(ys)  # [batch, block_size]

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
