"""Inference experiment schema — records generation runs in ExperimentStore.

Wraps ``ExperimentStore.record`` with a typed interface that enforces
inference-specific fields.

Experiment type string: ``"gpt-inference"``
"""

from __future__ import annotations

from aion.workspace.experiments import ExperimentStore

EXPERIMENT_TYPE = "gpt-inference"


def record_inference_experiment(
    store: ExperimentStore,
    *,
    model_id: str,
    tokenizer_id: str,
    config: dict,
    prompt_tokens: int,
    generated_tokens: int,
    generation_time_s: float,
    tokens_per_sec: float,
    mean_entropy: float,
    mean_top1_prob: float,
    stopped_by: str,
    strategy: str,
) -> dict:
    """Record a generation run in the experiment store.

    Parameters
    ----------
    store:
        ``ExperimentStore`` for the project.
    model_id:
        The GPT model artifact id.
    tokenizer_id:
        The tokenizer artifact id.
    config:
        ``GenerationConfig.to_dict()`` snapshot.
    prompt_tokens:
        Number of tokens in the prompt.
    generated_tokens:
        Number of tokens generated.
    generation_time_s:
        Wall-clock time for the generation call.
    tokens_per_sec:
        Throughput in tokens per second.
    mean_entropy:
        Mean per-step distribution entropy.
    mean_top1_prob:
        Mean probability of the chosen token per step.
    stopped_by:
        Stopping reason: ``"max_new_tokens"``, ``"eos"``, ``"stop_sequence"``,
        or ``"max_context"``.
    strategy:
        Sampling strategy string from ``GenerationConfig.strategy``.

    Returns
    -------
    The experiment record dict written to disk.
    """
    metrics = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": prompt_tokens + generated_tokens,
        "generation_time_s": generation_time_s,
        "tokens_per_sec": tokens_per_sec,
        "mean_entropy": mean_entropy,
        "mean_top1_prob": mean_top1_prob,
        "stopped_by": stopped_by,
        "strategy": strategy,
    }
    params = {
        "config": config,
        "strategy": strategy,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
    }
    return store.record(
        experiment_type=EXPERIMENT_TYPE,
        dataset_id="",
        dataset_fingerprint="",
        params=params,
        artifact_id=model_id,
        metrics=metrics,
    )
