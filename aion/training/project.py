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
each epoch, and optimizer state is checkpointed alongside the model, so a run
can be resumed and continue from where it stopped.

Resume
------
Each fresh run() mints a unique run_id (a UUID) and records it in
checkpoints/active_runs.json, keyed by model name.  A later run(resume=True)
recovers that run_id and continues the SAME run — restoring model weights,
optimizer state, and RNG state, and continuing epoch/checkpoint numbering.
If no prior run is recorded, resume raises ResumeError rather than silently
starting over.

Two checkpoint systems coexist (see ADR 0013):

- **Step-based (crash-safe)** — active when ``checkpoint_every_n_steps`` or
  ``checkpoint_every_minutes`` is set.  Complete bundles are written atomically
  every N steps / minutes, on each epoch boundary, and on Ctrl+C, to
  ``checkpoints/<run_id>/{latest,step_<N>}/``.  ``--resume`` restores the full
  ``latest/`` bundle and continues at the exact interrupted step (mid-epoch),
  bit-identically to an uninterrupted run.
- **Epoch-based (legacy default)** — used when no step cadence is configured;
  checkpoints and resumes only at epoch boundaries via ``ResumeTraining``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("aion.training.project")

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
from .resume import ResumeError, ResumeTraining
from .sampler import SampleGenerator
from .scheduler import SchedulerCallback, build_scheduler
from .step_checkpoint import StepCheckpointManager
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

    # ── run-id registry ───────────────────────────────────────────────────────
    # The active run_id for each model name is persisted so a later --resume can
    # recover the exact run to continue.  Each fresh run still gets its own UUID;
    # the registry only remembers which UUID is the latest per model.

    def _run_registry_path(self) -> Path:
        return self._project.dir("checkpoints") / "active_runs.json"

    def _read_run_registry(self) -> dict:
        path = self._run_registry_path()
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _record_run_id(self, run_id: str) -> None:
        registry = self._read_run_registry()
        registry[self._config.model_name] = run_id
        path = self._run_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _recover_run_id(self) -> str | None:
        return self._read_run_registry().get(self._config.model_name)

    @staticmethod
    def _history_from_records(records: list[dict]) -> dict:
        """Rebuild per-epoch metric lists from prior metrics.jsonl epoch records
        so a resumed run's returned history spans the full run."""
        loss, vloss, vppl, tps, gnorm = [], [], [], [], []
        for r in sorted(records, key=lambda x: x.get("epoch", 0)):
            if r.get("mean_loss") is not None:
                loss.append(r["mean_loss"])
            if r.get("val_loss") is not None:
                vloss.append(r["val_loss"])
            if r.get("perplexity") is not None:
                vppl.append(r["perplexity"])
            if r.get("tokens_per_sec") is not None:
                tps.append(r["tokens_per_sec"])
            gnorm.append(r.get("grad_norm") or 0.0)
        return {
            "loss_history": loss,
            "val_loss_history": vloss,
            "val_perplexity_history": vppl,
            "tokens_per_sec": tps,
            "grad_norm_history": gnorm,
        }

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

        project = self._project

        # ── run id (persisted so --resume recovers the same run) ───────────────
        if resume:
            run_id = self._recover_run_id()
            if run_id is None:
                raise ResumeError(
                    f"--resume requested but no previous run is recorded for model "
                    f"{cfg.model_name!r} in project {project.id!r}. "
                    f"Start a fresh run first (run without --resume)."
                )
        else:
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            self._record_run_id(run_id)

        # ── directories ───────────────────────────────────────────────────────
        logs_dir = project.logs_dir() / run_id
        logs_dir.mkdir(parents=True, exist_ok=True)

        # ── RNG streams ───────────────────────────────────────────────────────
        seed = cfg.seed
        rng_model = np.random.default_rng(seed)
        rng_data = np.random.default_rng(seed + 1)
        rng_sample = np.random.default_rng(seed + 2)

        # ── tokenizer ─────────────────────────────────────────────────────────
        logger.info("Resolving tokenizer %s...", cfg.tokenizer_id)
        t_stage = time.monotonic()
        from aion.tokenizers.store import TokenizerStore
        ts = TokenizerStore(project.dir("tokenizers"))
        tok_manifest = ts.open_manifest(cfg.tokenizer_id)
        tokenizer = ts.load(cfg.tokenizer_id)
        tok_fp = tok_manifest.get("vocabulary_fingerprint", "")
        eos_id = tokenizer.eos_id  # public accessor — do not assume a physical id
        logger.info(
            "Tokenizer loaded (vocab=%d, eos_id=%d) in %.2fs",
            tokenizer.vocab_size, eos_id, time.monotonic() - t_stage,
        )

        # ── corpus ────────────────────────────────────────────────────────────
        # Documents are shuffled deterministically (seed = cfg.seed) before the
        # train/val split so validation is a representative mix of sources.
        # (CorpusManager.build logs its own loading/tokenizing/packing stages.)
        t_stage = time.monotonic()
        corpus_mgr = CorpusManager(project.data_dir(), project.cache_dir())
        corpus = corpus_mgr.build(
            cfg.dataset_ids,
            tokenizer,
            tokenizer_id=cfg.tokenizer_id,
            tokenizer_fingerprint=tok_fp,
            eos_id=eos_id,
            split=cfg.train_split,
            shuffle_seed=cfg.seed,
            progress_fn=progress_fn,
        )
        logger.info("Corpus ready in %.2fs", time.monotonic() - t_stage)

        # ── datasets ──────────────────────────────────────────────────────────
        logger.info("Building train/validation datasets (context_length=%d)...",
                    cfg.context_length)
        t_stage = time.monotonic()
        train_ds = TokenizedDataset(corpus.train_tokens, cfg.context_length)
        val_ds = (
            TokenizedDataset(corpus.val_tokens, cfg.context_length)
            if len(corpus.val_tokens) >= cfg.context_length + 1
            else None
        )
        logger.info(
            "Datasets built in %.2fs: %d train blocks, %d val blocks",
            time.monotonic() - t_stage, len(train_ds),
            len(val_ds) if val_ds is not None else 0,
        )

        # ── batches ───────────────────────────────────────────────────────────
        logger.info("Creating batch samplers (batch_size=%d)...", cfg.batch_size)
        def train_sampler():
            return BatchSampler(train_ds, cfg.batch_size, shuffle=True, rng=rng_data)

        val_sampler = (
            BatchSampler(val_ds, cfg.batch_size, shuffle=False)
            if val_ds is not None else None
        )
        logger.info(
            "Batch samplers ready: ~%d train steps/epoch",
            (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size,
        )

        # ── model ─────────────────────────────────────────────────────────────
        logger.info("Building model...")
        t_stage = time.monotonic()
        from aion.gpt.config import GPTConfig
        gpt_cfg = GPTConfig.from_dict({
            **cfg.gpt_config,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": cfg.context_length,
        })
        model = GPTModel(gpt_cfg, rng=rng_model)
        logger.info(
            "Model built in %.2fs: %s params",
            time.monotonic() - t_stage, f"{model.param_count():,}",
        )

        # ── optimizer ─────────────────────────────────────────────────────────
        logger.info("Initializing optimizer (%s, lr=%g)...",
                    cfg.optimizer, cfg.learning_rate)
        t_stage = time.monotonic()
        if cfg.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        else:
            optimizer = SGD(model.parameters(), lr=cfg.learning_rate)
        logger.info("Optimizer ready in %.2fs", time.monotonic() - t_stage)

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

        # ── step-based checkpointing (crash-safe) ─────────────────────────────
        step_ckpt_enabled = (
            cfg.checkpoint_every_n_steps > 0 or cfg.checkpoint_every_minutes > 0
        )
        step_mgr = None
        checkpoint_fn = None
        steps_per_epoch_est = (len(train_ds) + cfg.batch_size - 1) // cfg.batch_size
        if step_ckpt_enabled:
            step_mgr = StepCheckpointManager(
                project.dir("checkpoints"), run_id,
                keep_last_n=cfg.keep_last_n_step_checkpoints,
            )

            def checkpoint_fn(reason: str, ts: dict) -> None:
                t0 = time.monotonic()
                state = {
                    "run_id": run_id,
                    "model_name": cfg.model_name,
                    "global_step": ts["global_step"],
                    "epoch": ts["epoch"],
                    "batch_in_epoch": ts["batch_in_epoch"],
                    "total_epochs": ts["total_epochs"],
                    "reason": reason,
                    "created_at": now_iso(),
                    "tokens_processed": ts["tokens_processed"],
                    "rng": {
                        "model": rng_model.bit_generator.state,
                        "sample": rng_sample.bit_generator.state,
                        "data_epoch_start": ts["data_rng_epoch_state"],
                    },
                    "scheduler": {
                        "name": cfg.scheduler,
                        "base_lr": cfg.learning_rate,
                        "min_lr": cfg.min_lr,
                        "warmup_steps": cfg.warmup_steps,
                        "total_steps": cfg.epochs * steps_per_epoch_est,
                    },
                    "history": ts["history"],
                    "config": cfg.to_dict(),
                }
                latest = step_mgr.save(model, optimizer, state)
                dt = time.monotonic() - t0
                sep = "-" * 40
                print(f"\n{sep}\nCheckpoint saved\n\n"
                      f"Step: {ts['global_step']}\n"
                      f"Epoch: {ts['epoch']}\n"
                      f"Time: {dt:.2f} sec\n\n"
                      f"Directory:\n\n{latest}\n{sep}", flush=True)
                logger.info("Checkpoint (%s) saved at step %d in %.2fs",
                            reason, ts["global_step"], dt)

        callbacks: list[TrainingCallback] = [
            SchedulerCallback(scheduler),
            _MetricsCallback(collector),
            _RNGCallback(logs_dir, rng_model, rng_data, rng_sample),
        ]
        # Epoch-boundary checkpoints (old format) only when step-checkpointing is
        # off, to avoid two systems writing checkpoints for the same run.
        if not step_ckpt_enabled:
            callbacks.insert(
                2, CheckpointCallback(ckpt_mgr, every_n_epochs=cfg.checkpoint_every_n_epochs)
            )

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
        start_step = 0
        start_batch_in_epoch = 0
        prior_history: dict | None = None
        if resume:
            if step_mgr is not None and step_mgr.has_checkpoint():
                # Step-based resume: locate latest/ automatically and restore the
                # complete state (model, optimizer, scheduler-via-step, RNG,
                # history, global_step, epoch, batch_in_epoch).
                loaded = step_mgr.load_latest()
                StepCheckpointManager.restore_model(model, loaded)
                StepCheckpointManager.restore_optimizer(optimizer, loaded)
                rng = loaded.state.get("rng", {})
                if rng.get("model") is not None:
                    rng_model.bit_generator.state = rng["model"]
                if rng.get("sample") is not None:
                    rng_sample.bit_generator.state = rng["sample"]
                if rng.get("data_epoch_start") is not None:
                    rng_data.bit_generator.state = rng["data_epoch_start"]
                start_step = loaded.global_step
                start_epoch = loaded.epoch
                start_batch_in_epoch = loaded.batch_in_epoch
                prior_history = loaded.state.get("history")
                if start_epoch >= cfg.epochs and start_batch_in_epoch == 0:
                    raise ResumeError(
                        f"run {run_id!r} already completed all {cfg.epochs} epochs. "
                        f"Increase epochs to continue training."
                    )
                logger.info(
                    "Resuming run %s from checkpoint: step=%d epoch=%d batch_in_epoch=%d (%s)",
                    run_id, start_step, start_epoch, start_batch_in_epoch, loaded.directory,
                )
            else:
                # Epoch-based resume (backward-compatible path).
                resumer = ResumeTraining(ckpt_mgr, logs_dir)
                if not resumer.can_resume():
                    raise ResumeError(
                        f"--resume requested but run {run_id!r} for model {cfg.model_name!r} "
                        f"has no saved checkpoints to resume from."
                    )
                state = resumer.restore(model, optimizer)
                start_epoch = state.start_epoch
                start_step = state.start_step
                if state.rng_states:
                    ResumeTraining.restore_rng_state(
                        state.rng_states, rng_model, rng_data, rng_sample,
                    )
                prior_history = self._history_from_records(state.metrics_history)
                if start_epoch >= cfg.epochs:
                    raise ResumeError(
                        f"run {run_id!r} already completed {start_epoch} epoch(s), which meets "
                        f"the configured epochs={cfg.epochs}. Increase epochs to continue training."
                    )
                logger.info("Resuming run %s from epoch %d (step %d)",
                            run_id, start_epoch, start_step)

        # ── write config snapshot ─────────────────────────────────────────────
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
            log_every_n_steps=cfg.log_every_n_steps,
        )
        remaining_epochs = cfg.epochs - start_epoch
        logger.info(
            "Starting training: %d epoch(s) (%d..%d), run_id=%s",
            remaining_epochs, start_epoch + 1, cfg.epochs, run_id,
        )
        if step_ckpt_enabled:
            logger.info(
                "Step checkpointing: every %d steps%s, keep last %d",
                cfg.checkpoint_every_n_steps,
                f" / {cfg.checkpoint_every_minutes:g} min"
                if cfg.checkpoint_every_minutes else "",
                cfg.keep_last_n_step_checkpoints,
            )
        t_train = time.monotonic()
        result = trainer.train(
            train_sampler,
            epochs=remaining_epochs,
            val_sampler=val_sampler,
            seed=seed,
            start_epoch=start_epoch,
            start_step=start_step,
            total_epochs=cfg.epochs,
            history=prior_history,
            data_rng=rng_data,
            start_batch_in_epoch=start_batch_in_epoch,
            checkpoint_fn=checkpoint_fn,
            checkpoint_every_n_steps=cfg.checkpoint_every_n_steps,
            checkpoint_every_minutes=cfg.checkpoint_every_minutes,
        )
        logger.info("Training loop finished in %.1fs", time.monotonic() - t_train)

        # ── save model ────────────────────────────────────────────────────────
        logger.info("Saving model...")
        t_stage = time.monotonic()
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
        logger.info("Model saved as %s in %.2fs", model_id, time.monotonic() - t_stage)

        # ── experiment record ─────────────────────────────────────────────────
        logger.info("Recording experiment...")
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
        logger.info("Generating model card...")
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
        logger.info("Run complete: model=%s checkpoints=%s", model_id, ckpt_mgr.run_dir)

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
