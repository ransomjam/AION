"""TrainingProject — top-level orchestrator for a GPT training run.

Responsibilities
----------------
- Resolve dataset and tokenizer from the project.
- Build (or load from cache) the packed corpus via CorpusManager.
- Construct GPTModel and optimizer from TrainingConfig.
- Wire callbacks: SchedulerCallback, CheckpointCallback, MetricsCallback,
  SampleCallback.
- Delegate the training loop entirely to GPTTrainer.
- Save the trained model via GPTStore.
- Record the experiment via record_gpt_experiment.
- Generate and persist the ModelCard.
- Return a TrainingRunResult with all artifacts and metrics.

TrainingProject does not own the training loop.  GPTTrainer does.

Corpus cache
------------
The corpus is cached under project/cache/corpus/<fingerprint>/.
The cache is keyed by DatasetFingerprint.combined.  It is never automatically
invalidated.  project/cache is entirely disposable and may be safely cleared
by the user.

Reproducibility
---------------
Three independent RNG streams are seeded from TrainingConfig.seed:
    rng_model   model weight initialisation
    rng_data    batch shuffling
    rng_sample  sample generation (greedy by default, so unused)
RNG states are serialized to logs/<run_id>/rng_state_epoch_<n>.json after
each epoch so a run can be resumed with identical results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aion.gpt.data import BatchSampler, TokenizedDataset
from aion.gpt.experiment import record_gpt_experiment
from aion.gpt.model import GPTModel
from aion.gpt.store import GPTStore
from aion.gpt.trainer import GPTTrainer, TrainingCallback, TrainingCallbackList
from aion.nn.optim import Adam, SGD
from aion.util import now_iso
from aion.workspace.experiments import ExperimentStore

from .checkpoint import CheckpointCallback, CheckpointManager
from .config import TrainingConfig
from .corpus import CorpusManager
from .dashboard import TrainingDashboard
from .metrics import MetricsCollector
from .model_card import ModelCard
from .resume import ResumeTraining
from .sampler import SampleGenerator
from .scheduler import SchedulerCallback, build_scheduler
from .fingerprint import DatasetFingerprint


@dataclass
class TrainingRunResult:
    """Outcome of a completed TrainingProject.run() call."""
    run_id: str
    model_id: str
    model_manifest: dict
    experiment_record: dict
    model_card: dict          # ModelCardResult.data
    model_card_markdown: str
    metrics: dict
    corpus_fingerprint: str
    corpus_stats: dict
    checkpoint_dir: str
    log_path: str


# ── Internal callbacks ────────────────────────────────────────────────────────

class _MetricsCallback(TrainingCallback):
    """Records per-step and per-epoch metrics into MetricsCollector."""

    def __init__(self, collector: MetricsCollector) -> None:
        self._c = collector

    def on_step_end(self, trainer, ctx: dict) -> None:
        self._c.record_step(ctx["global_step"], ctx["epoch"], {
            "train_loss": ctx.get("loss"),
            "grad_norm": ctx.get("grad_norm"),
            "tokens": ctx.get("tokens"),
            "lr": ctx.get("lr"),
        })

    def on_epoch_end(self, trainer, ctx: dict) -> None:
        self._c.record_epoch(ctx["epoch"], {
            "mean_loss": ctx.get("mean_loss"),
            "val_loss": ctx.get("val_loss"),
            "perplexity": ctx.get("val_perplexity"),
            "tokens_per_sec": ctx.get("tokens_per_sec"),
            "elapsed_s": ctx.get("elapsed_s"),
            "global_step": ctx.get("global_step"),
            "lr": ctx.get("lr", 0.0),
            "grad_norm": 0.0,
        })


class _SampleCallback(TrainingCallback):
    """Generates text samples at the end of every N epochs."""

    def __init__(
        self,
        generator: SampleGenerator,
        prompts: list[str],
        every_n_epochs: int,
        collector: MetricsCollector,
    ) -> None:
        self._gen = generator
        self._prompts = prompts
        self._every = every_n_epochs
        self._collector = collector

    def on_epoch_end(self, trainer, ctx: dict) -> None:
        epoch = ctx.get("epoch", 0)
        if not self._prompts:
            return
        if (epoch + 1) % self._every != 0:
            return
        results = self._gen.generate_all(self._prompts)
        samples = [
            {"prompt": r.prompt, "generated": r.generated_text, "tokens": r.tokens}
            for r in results
        ]
        self._collector.record_epoch(epoch, {"samples": samples})


class _RNGCallback(TrainingCallback):
    """Serializes RNG states after each epoch for reproducible resume."""

    def __init__(
        self,
        logs_dir: Path,
        rng_model: np.random.Generator,
        rng_data: np.random.Generator,
        rng_sample: np.random.Generator,
    ) -> None:
        self._logs_dir = logs_dir
        self._rng_model = rng_model
        self._rng_data = rng_data
        self._rng_sample = rng_sample

    def on_epoch_end(self, trainer, ctx: dict) -> None:
        epoch = ctx.get("epoch", 0)
        ResumeTraining.save_rng_state(
            self._logs_dir,
            epoch + 1,
            self._rng_model,
            self._rng_data,
            self._rng_sample,
        )


# ── TrainingProject ───────────────────────────────────────────────────────────

class TrainingProject:
    """Orchestrate a complete GPT training run within a project.

    Parameters
    ----------
    project:
        An opened ``Project`` instance.
    config:
        ``TrainingConfig`` controlling all aspects of the run.
    """

    def __init__(self, project, config: TrainingConfig) -> None:
        self._project = project
        self._config = config

    def run(
        self,
        *,
        resume: bool = False,
        progress_fn=None,
    ) -> TrainingRunResult:
        """Execute the full training workflow.

        Parameters
        ----------
        resume:
            If ``True`` and a checkpoint exists for this run, restore weights
            and continue from the last saved epoch.
        progress_fn:
            Optional ``progress_fn(fraction, message)`` callback passed to
            ``GPTTrainer`` for job progress reporting.
        """
        cfg = self._config
        cfg.validate()

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        project = self._project

        # ── directories ───────────────────────────────────────────────────────
        logs_dir = project.logs_dir() / run_id
        logs_dir.mkdir(parents=True, exist_ok=True)

        # ── RNG streams ───────────────────────────────────────────────────────
        seed = cfg.seed
        rng_model = np.random.default_rng(seed)
        rng_data = np.random.default_rng(seed + 1)
        rng_sample = np.random.default_rng(seed + 2)

        # ── tokenizer ─────────────────────────────────────────────────────────
        from aion.tokenizers.store import TokenizerStore
        ts = TokenizerStore(project.dir("tokenizers"))
        tok_manifest = ts.open_manifest(cfg.tokenizer_id)
        tokenizer = ts.load(cfg.tokenizer_id)
        tok_fp = tok_manifest.get("vocabulary_fingerprint", "")
        eos_id = tokenizer.vocab_size - 1  # BPE: last token is <eos>

        # ── corpus ────────────────────────────────────────────────────────────
        corpus_mgr = CorpusManager(project.data_dir(), project.cache_dir())
        corpus = corpus_mgr.build(
            cfg.dataset_ids,
            tokenizer,
            tokenizer_id=cfg.tokenizer_id,
            tokenizer_fingerprint=tok_fp,
            eos_id=eos_id,
            split=cfg.train_split,
            progress_fn=progress_fn,
        )

        # ── datasets ──────────────────────────────────────────────────────────
        train_ds = TokenizedDataset(corpus.train_tokens, cfg.context_length)
        val_ds = (
            TokenizedDataset(corpus.val_tokens, cfg.context_length)
            if len(corpus.val_tokens) >= cfg.context_length + 1
            else None
        )

        def train_sampler():
            return BatchSampler(train_ds, cfg.batch_size, shuffle=True, rng=rng_data)

        val_sampler = (
            BatchSampler(val_ds, cfg.batch_size, shuffle=False)
            if val_ds is not None else None
        )

        # ── model ─────────────────────────────────────────────────────────────
        from aion.gpt.config import GPTConfig
        gpt_cfg = GPTConfig.from_dict({
            **cfg.gpt_config,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": cfg.context_length,
        })
        model = GPTModel(gpt_cfg, rng=rng_model)

        # ── optimizer ─────────────────────────────────────────────────────────
        if cfg.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        else:
            optimizer = SGD(model.parameters(), lr=cfg.learning_rate)

        # ── callbacks ─────────────────────────────────────────────────────────
        metrics_log = logs_dir / "metrics.jsonl"
        collector = MetricsCollector(metrics_log)

        ckpt_mgr = CheckpointManager(
            project.dir("checkpoints"),
            run_id,
            keep_last_n=cfg.keep_last_n_checkpoints,
        )

        scheduler = build_scheduler(
            cfg.scheduler,
            cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            min_lr=cfg.min_lr,
        )

        callbacks: list[TrainingCallback] = [
            SchedulerCallback(scheduler),
            _MetricsCallback(collector),
            CheckpointCallback(ckpt_mgr, every_n_epochs=cfg.checkpoint_every_n_epochs),
            _RNGCallback(logs_dir, rng_model, rng_data, rng_sample),
        ]

        if cfg.sample_prompts:
            sample_gen = SampleGenerator(
                model, tokenizer,
                max_new_tokens=cfg.sample_max_new_tokens,
                strategy="greedy",
                eos_token_id=eos_id,
            )
            callbacks.append(_SampleCallback(
                sample_gen, cfg.sample_prompts,
                cfg.sample_every_n_epochs, collector,
            ))

        # ── resume ────────────────────────────────────────────────────────────
        start_epoch = 0
        if resume:
            resumer = ResumeTraining(ckpt_mgr, logs_dir)
            if resumer.can_resume():
                state = resumer.restore(model, optimizer)
                start_epoch = state.start_epoch
                if state.rng_states:
                    ResumeTraining.restore_rng_state(
                        state.rng_states, rng_model, rng_data, rng_sample,
                    )

        # ── write config snapshot ─────────────────────────────────────────────
        import json
        config_path = logs_dir / "config.json"
        config_path.write_text(
            json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ── train ─────────────────────────────────────────────────────────────
        trainer = GPTTrainer(
            model, optimizer,
            grad_clip=cfg.grad_clip,
            callbacks=TrainingCallbackList(callbacks),
            progress_fn=progress_fn,
        )
        remaining_epochs = cfg.epochs - start_epoch
        result = trainer.train(
            train_sampler,
            epochs=remaining_epochs,
            val_sampler=val_sampler,
            seed=seed,
        )

        # ── save model ────────────────────────────────────────────────────────
        gpt_store = GPTStore(project.dir("models"))
        model_manifest = gpt_store.save(
            model, result,
            name=cfg.model_name,
            description=cfg.gpt_config.get("description", ""),
            dataset_id=",".join(cfg.dataset_ids),
            dataset_fingerprint=corpus.fingerprint.combined,
            tokenizer_id=cfg.tokenizer_id,
            tokenizer_fingerprint=tok_fp,
            params={"training_config": cfg.to_dict()},
        )
        model_id = model_manifest["id"]

        # ── experiment record ─────────────────────────────────────────────────
        exp_store = ExperimentStore(project.dir("experiments"))
        trainable = sum(
            p.data.size for p in model.parameters() if p.requires_grad
        )
        exp_record = record_gpt_experiment(
            exp_store,
            model_id=model_id,
            config=gpt_cfg.to_dict(),
            dataset_id=",".join(cfg.dataset_ids),
            dataset_fingerprint=corpus.fingerprint.combined,
            tokenizer_id=cfg.tokenizer_id,
            tokenizer_fingerprint=tok_fp,
            epochs=cfg.epochs,
            tokens_processed=result.metrics.get("tokens_processed", 0),
            context_length=cfg.context_length,
            param_count=model.param_count(),
            trainable_param_count=trainable,
            final_loss=result.metrics.get("final_loss", 0.0),
            loss_history=result.metrics.get("loss_history", []),
            final_val_loss=result.metrics.get("final_val_loss"),
            final_val_perplexity=result.metrics.get("final_val_perplexity"),
            val_loss_history=result.metrics.get("val_loss_history"),
            val_perplexity_history=result.metrics.get("val_perplexity_history"),
            training_time_s=result.metrics.get("training_time_s"),
            tokens_per_sec=result.metrics.get("tokens_per_sec"),
            grad_norm_history=result.metrics.get("grad_norm_history"),
            seed=seed,
        )

        # ── model card ────────────────────────────────────────────────────────
        card = ModelCard(
            model_name=cfg.model_name,
            gpt_config=gpt_cfg.to_dict(),
            param_count=model.param_count(),
            tokenizer_manifest=tok_manifest,
            corpus_stats=corpus.stats.to_dict(),
            corpus_fingerprint=corpus.fingerprint.combined,
            dataset_ids=cfg.dataset_ids,
            training_config=cfg.to_dict(),
            metrics=result.metrics,
            model_id=model_id,
        )
        card_result = card.generate()

        # Write model card alongside the model
        model_root = project.dir("models") / model_id
        (model_root / "model_card.md").write_text(
            card_result.markdown, encoding="utf-8"
        )
        (model_root / "model_card.json").write_text(
            json.dumps(card_result.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return TrainingRunResult(
            run_id=run_id,
            model_id=model_id,
            model_manifest=model_manifest,
            experiment_record=exp_record,
            model_card=card_result.data,
            model_card_markdown=card_result.markdown,
            metrics=result.metrics,
            corpus_fingerprint=corpus.fingerprint.combined,
            corpus_stats=corpus.stats.to_dict(),
            checkpoint_dir=str(ckpt_mgr.run_dir),
            log_path=str(metrics_log),
        )
