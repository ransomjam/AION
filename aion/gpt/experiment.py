"""GPT experiment schema — records language-model training runs.

Wraps ``ExperimentStore.record`` with a typed interface that enforces the
GPT-specific fields: tokens processed, context length, validation perplexity,
parameter counts, tokenizer fingerprint, and dataset fingerprint.

Experiment type string: ``"gpt-language-model"``
"""

from __future__ import annotations

from aion.workspace.experiments import ExperimentStore

EXPERIMENT_TYPE = "gpt-language-model"


def record_gpt_experiment(
    store: ExperimentStore,
    *,
    model_id: str,
    config: dict,
    dataset_id: str,
    dataset_fingerprint: str,
    tokenizer_id: str,
    tokenizer_fingerprint: str,
    # training metrics
    epochs: int,
    tokens_processed: int,
    context_length: int,
    param_count: int,
    trainable_param_count: int,
    final_loss: float,
    loss_history: list[float],
    # optional validation metrics
    final_val_loss: float | None = None,
    final_val_perplexity: float | None = None,
    val_loss_history: list[float] | None = None,
    val_perplexity_history: list[float] | None = None,
    # optional performance metrics
    training_time_s: float | None = None,
    tokens_per_sec: list[float] | None = None,
    grad_norm_history: list[float] | None = None,
    seed: int = 0,
) -> dict:
    """Record a GPT training run in the experiment store.

    Parameters
    ----------
    store:
        ``ExperimentStore`` for the project.
    model_id:
        The artifact id returned by ``GPTStore.save``.
    config:
        ``GPTConfig.to_dict()`` snapshot.
    dataset_id / dataset_fingerprint:
        Identity of the training dataset.
    tokenizer_id / tokenizer_fingerprint:
        Identity of the tokenizer used to produce the token ids.
    epochs:
        Number of training epochs completed.
    tokens_processed:
        Total tokens seen during training (sum over all epochs × batch tokens).
    context_length:
        Block size (``GPTConfig.max_seq_len`` or the dataset block_size).
    param_count:
        Total scalar parameter count.
    trainable_param_count:
        Parameters with ``requires_grad=True`` (may differ if layers frozen).
    final_loss / loss_history:
        Training loss.
    final_val_loss / final_val_perplexity:
        Validation metrics (None if no validation set was used).

    Returns
    -------
    The experiment record dict written to disk.
    """
    metrics = {
        "epochs": epochs,
        "tokens_processed": tokens_processed,
        "context_length": context_length,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_perplexity,
        "val_loss_history": val_loss_history or [],
        "val_perplexity_history": val_perplexity_history or [],
        "training_time_s": training_time_s,
        "tokens_per_sec": tokens_per_sec or [],
        "grad_norm_history": grad_norm_history or [],
        "seed": seed,
        "tokenizer_id": tokenizer_id,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }

    params = {
        "config": config,
        "context_length": context_length,
        "tokenizer_id": tokenizer_id,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }

    return store.record(
        experiment_type=EXPERIMENT_TYPE,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        params=params,
        artifact_id=model_id,
        metrics=metrics,
    )
