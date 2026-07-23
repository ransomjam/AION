"""StoppingCriteria — composable stopping conditions for the generation loop.

Each criterion is a callable that returns ``True`` when generation should stop.
The generator holds a ``StoppingCriteriaList`` and stops when any criterion
fires.

Built-in criteria
-----------------
MaxNewTokensCriteria    — stop after N tokens generated
EosTokenCriteria        — stop when the EOS token id is produced
StopSequenceCriteria    — stop when a stop string appears as a suffix of the
                          decoded output

Extension point: add new criteria without modifying the generation loop.
Future examples: tool-call token detection, grammar completion, confidence
threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StoppingCriteria(ABC):
    """Abstract base for all stopping criteria."""

    @abstractmethod
    def should_stop(
        self,
        generated_ids: list[int],
        last_token_id: int,
        decoded_so_far: str,
    ) -> bool:
        """Return True if generation should stop.

        Parameters
        ----------
        generated_ids:
            All token ids generated so far in this call (not the prompt).
        last_token_id:
            The most recently generated token id (same as generated_ids[-1]).
        decoded_so_far:
            The decoded string of all generated tokens so far.  Provided so
            criteria do not need a tokenizer reference.
        """


class MaxNewTokensCriteria(StoppingCriteria):
    """Stop after ``max_new_tokens`` tokens have been generated."""

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens

    def should_stop(
        self,
        generated_ids: list[int],
        last_token_id: int,
        decoded_so_far: str,
    ) -> bool:
        return len(generated_ids) >= self.max_new_tokens


class EosTokenCriteria(StoppingCriteria):
    """Stop when the EOS token id is produced."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def should_stop(
        self,
        generated_ids: list[int],
        last_token_id: int,
        decoded_so_far: str,
    ) -> bool:
        return last_token_id == self.eos_token_id


class StopSequenceCriteria(StoppingCriteria):
    """Stop when any stop string appears as a suffix of the decoded output."""

    def __init__(self, stop_sequences: list[str]) -> None:
        self.stop_sequences = stop_sequences

    def should_stop(
        self,
        generated_ids: list[int],
        last_token_id: int,
        decoded_so_far: str,
    ) -> bool:
        return any(decoded_so_far.endswith(s) for s in self.stop_sequences)


class StoppingCriteriaList:
    """Applies a list of criteria; stops when any returns True."""

    def __init__(self, criteria: list[StoppingCriteria] | None = None) -> None:
        self._criteria: list[StoppingCriteria] = criteria or []

    def append(self, criterion: StoppingCriteria) -> None:
        self._criteria.append(criterion)

    def __call__(
        self,
        generated_ids: list[int],
        last_token_id: int,
        decoded_so_far: str,
    ) -> bool:
        return any(
            c.should_stop(generated_ids, last_token_id, decoded_so_far)
            for c in self._criteria
        )

    def __len__(self) -> int:
        return len(self._criteria)
