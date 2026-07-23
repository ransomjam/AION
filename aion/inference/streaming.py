"""StreamingCallback — protocol for token-by-token streaming.

The generator calls ``on_token`` after each step and ``on_complete`` when the
loop exits.  The callback receives the decoded token string so it does not
need a tokenizer reference.

The generator is synchronous.  Streaming is achieved by the callback
side-effect, not by making the generator async.  This keeps the inference
engine pure and testable without an event loop.

The app layer implements a concrete callback that pushes tokens to the SSE
stream for the UI.  The inference package defines only the protocol.
"""

from __future__ import annotations

from .result import GenerationResult, TokenStep


class StreamingCallback:
    """Base streaming callback.  Override the methods you need.

    The default implementations are no-ops so subclasses only need to
    override the methods they care about.
    """

    def on_token(self, token_id: int, token_str: str, step: TokenStep) -> None:
        """Called immediately after each token is generated.

        Parameters
        ----------
        token_id:
            The generated token id.
        token_str:
            The decoded string for this token.
        step:
            The full ``TokenStep`` record for this position.
        """

    def on_complete(self, result: GenerationResult) -> None:
        """Called once when generation finishes (before returning to caller).

        Parameters
        ----------
        result:
            The complete ``GenerationResult``.
        """


class CollectingCallback(StreamingCallback):
    """Callback that collects tokens as they arrive.

    Useful for testing and for building a live text buffer in the UI.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.token_ids: list[int] = []
        self.result: GenerationResult | None = None

    def on_token(self, token_id: int, token_str: str, step: TokenStep) -> None:
        self.tokens.append(token_str)
        self.token_ids.append(token_id)

    def on_complete(self, result: GenerationResult) -> None:
        self.result = result

    @property
    def text(self) -> str:
        return "".join(self.tokens)
