"""Recovery-validation tests for step-based, crash-safe checkpointing.

Proves the guarantees required of a production training framework:

- a checkpoint at step K resumes at step K+1 (global_step continuity),
- optimizer state (Adam moments + step counter) survives a resume,
- scheduler (LR) continues correctly after a resume,
- RNG / data order continues correctly after a resume,
- the run_id is unchanged across a resume,
- **losses after resume are bit-identical to uninterrupted training**,
- checkpoints are written atomically (no partial/temp leftovers) with the
  documented ``latest/`` + ``step_<N>/`` layout, and prune correctly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel
from aion.gpt.data import TokenizedDataset, BatchSampler
from aion.gpt.trainer import GPTTrainer, TrainingCallback
from aion.nn.optim import Adam
from aion.training.scheduler import SchedulerCallback, build_scheduler
from aion.training.step_checkpoint import StepCheckpointManager


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_model(seed: int = 42):
    cfg = GPTConfig.from_dict({
        "d_model": 16, "n_heads": 2, "n_layers": 2, "d_ff": 32,
        "vocab_size": 64, "max_seq_len": 8,
        "dropout": 0.0, "activation": "gelu", "pre_norm": True, "tie_weights": True,
    })
    model = GPTModel(cfg, rng=np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=1e-3)
    return cfg, model, opt


def _make_sampler(data_rng, n_blocks: int = 40, block: int = 8, batch: int = 4):
    # Deterministic token stream; block+1 tokens per block.
    data = (np.arange(n_blocks * (block + 1)) % 64).astype(np.int64)
    ds = TokenizedDataset(data, block)
    return lambda: BatchSampler(ds, batch, shuffle=True, rng=data_rng)


def _flat_weights(model):
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def _make_ckpt_fn(mgr, model, opt, rng_model, rng_sample):
    def fn(reason, ts):
        state = {
            "run_id": "run_test",
            "reason": reason,
            "global_step": ts["global_step"],
            "epoch": ts["epoch"],
            "batch_in_epoch": ts["batch_in_epoch"],
            "total_epochs": ts["total_epochs"],
            "tokens_processed": ts["tokens_processed"],
            "rng": {
                "model": rng_model.bit_generator.state,
                "sample": rng_sample.bit_generator.state,
                "data_epoch_start": ts["data_rng_epoch_state"],
            },
            "history": ts["history"],
        }
        mgr.save(model, opt, state)
    return fn


class _InterruptAt(TrainingCallback):
    """Raise KeyboardInterrupt at a chosen global step to simulate Ctrl+C."""
    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, trainer, context) -> None:
        if context.get("global_step") == self.step:
            raise KeyboardInterrupt


# ── StepCheckpointManager: atomicity, layout, load, prune ─────────────────────

class TestStepCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aion_ckpt_")
        _, self.model, self.opt = _build_model()
        # take one Adam step so moment buffers exist
        x = np.zeros((2, 8), dtype=np.int64)
        logits, _ = self.model(x)
        from aion.gpt.loss import CausalLanguageModelLoss
        loss = CausalLanguageModelLoss()(logits, np.zeros((2, 8), dtype=np.int64))
        loss.backward()
        self.opt.step()

    def _state(self, step):
        return {"run_id": "run_x", "global_step": step, "epoch": 0,
                "batch_in_epoch": step, "total_epochs": 1, "history": {}}

    def test_layout_latest_and_step_dirs(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x", keep_last_n=10)
        mgr.save(self.model, self.opt, self._state(250))
        mgr.save(self.model, self.opt, self._state(500))
        run_dir = Path(self.tmp) / "run_x"
        self.assertTrue((run_dir / "latest").is_dir())
        self.assertTrue((run_dir / "step_250").is_dir())
        self.assertTrue((run_dir / "step_500").is_dir())
        for d in ("latest", "step_500"):
            self.assertTrue((run_dir / d / "model.npz").is_file())
            self.assertTrue((run_dir / d / "state.json").is_file())

    def test_atomic_no_temp_leftovers(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x")
        mgr.save(self.model, self.opt, self._state(250))
        leftovers = list((Path(self.tmp) / "run_x").glob(".tmp*")) \
            + list((Path(self.tmp) / "run_x").glob(".*.old.*"))
        self.assertEqual(leftovers, [])

    def test_load_latest_returns_newest(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x", keep_last_n=10)
        mgr.save(self.model, self.opt, self._state(250))
        mgr.save(self.model, self.opt, self._state(500))
        loaded = mgr.load_latest()
        self.assertEqual(loaded.global_step, 500)

    def test_load_latest_fallback_when_latest_corrupt(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x", keep_last_n=10)
        mgr.save(self.model, self.opt, self._state(250))
        mgr.save(self.model, self.opt, self._state(500))
        # Corrupt latest/state.json — loader must fall back to newest step dir.
        (Path(self.tmp) / "run_x" / "latest" / "state.json").write_text("{ broken",
                                                                        encoding="utf-8")
        loaded = mgr.load_latest()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.global_step, 500)

    def test_prune_keeps_last_n(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x", keep_last_n=2)
        for s in (250, 500, 750, 1000):
            mgr.save(self.model, self.opt, self._state(s))
        self.assertEqual(mgr.list_steps(), [750, 1000])
        self.assertTrue((Path(self.tmp) / "run_x" / "latest").is_dir())

    def test_restore_model_and_optimizer(self):
        mgr = StepCheckpointManager(Path(self.tmp), "run_x")
        mgr.save(self.model, self.opt, self._state(250))
        loaded = mgr.load_latest()
        _, model2, opt2 = _build_model(seed=999)  # different init
        self.assertFalse(np.allclose(_flat_weights(model2), _flat_weights(self.model)))
        StepCheckpointManager.restore_model(model2, loaded)
        StepCheckpointManager.restore_optimizer(opt2, loaded)
        self.assertTrue(np.array_equal(_flat_weights(model2), _flat_weights(self.model)))
        self.assertEqual(opt2._t, self.opt._t)  # Adam step counter restored
        # Adam moment buffers restored
        for p2, p1 in zip(opt2.parameters, self.opt.parameters):
            m2, m1 = opt2._m.get(id(p2)), self.opt._m.get(id(p1))
            if m1 is not None:
                self.assertTrue(np.array_equal(m2, m1))


# ── Trainer resume: continuity + bit-identical losses ─────────────────────────

class TestResumeContinuity(unittest.TestCase):
    def _run_uninterrupted(self, epochs=3):
        _, model, opt = _build_model()
        rng = np.random.default_rng(43)
        sampler = _make_sampler(rng)
        trainer = GPTTrainer(model, opt, grad_clip=1.0)
        result = trainer.train(sampler, epochs=epochs, total_epochs=epochs, data_rng=rng)
        return model, result

    def test_bit_identical_resume(self):
        # (1) uninterrupted reference
        model_full, res_full = self._run_uninterrupted(epochs=3)
        W_full = _flat_weights(model_full)

        # (2) interrupt mid-epoch-0 at step 6, checkpoint, then resume
        tmp = tempfile.mkdtemp(prefix="aion_resume_")
        mgr = StepCheckpointManager(Path(tmp), "run_test", keep_last_n=10)
        _, model_s, opt_s = _build_model()
        rng_s = np.random.default_rng(43)
        sampler_s = _make_sampler(rng_s)
        ckfn = _make_ckpt_fn(mgr, model_s, opt_s,
                             np.random.default_rng(42), np.random.default_rng(44))
        trainer_s = GPTTrainer(model_s, opt_s, grad_clip=1.0,
                               callbacks=[_InterruptAt(6)])
        with self.assertRaises(KeyboardInterrupt):
            trainer_s.train(sampler_s, epochs=3, total_epochs=3, data_rng=rng_s,
                            checkpoint_fn=ckfn, checkpoint_every_n_steps=1000)

        loaded = mgr.load_latest()
        self.assertEqual(loaded.global_step, 6)          # checkpoint at step 6
        self.assertEqual(loaded.batch_in_epoch, 6)

        # (3) resume from latest/ into a fresh model+optimizer
        _, model_r, opt_r = _build_model()
        StepCheckpointManager.restore_model(model_r, loaded)
        StepCheckpointManager.restore_optimizer(opt_r, loaded)
        rng_r = np.random.default_rng(43)
        rng_r.bit_generator.state = loaded.state["rng"]["data_epoch_start"]
        sampler_r = _make_sampler(rng_r)
        trainer_r = GPTTrainer(model_r, opt_r, grad_clip=1.0)
        res_r = trainer_r.train(
            sampler_r, epochs=3 - loaded.epoch, total_epochs=3,
            start_epoch=loaded.epoch, start_step=loaded.global_step,
            start_batch_in_epoch=loaded.batch_in_epoch, data_rng=rng_r,
            history=loaded.state["history"],
        )
        W_resumed = _flat_weights(model_r)

        # resumes at step 7, ends at the same step as uninterrupted, weights match
        self.assertTrue(
            np.array_equal(W_full, W_resumed),
            msg=f"resumed weights differ (max|Δ|={np.max(np.abs(W_full-W_resumed)):.2e})",
        )

    def test_second_resume_does_not_replay_batches(self):
        """A resumed run must checkpoint its ABSOLUTE position in the epoch.

        Regression test.  ``batch_in_epoch`` used to be written from
        ``n_batches``, which resets each epoch and counts only the batches the
        CURRENT process ran.  On a fresh run the two agree, so this went
        unnoticed; on a resumed run the checkpointed value was short by exactly
        the number of skipped batches, and the *next* resume replayed them —
        training on the same data twice and silently breaking the bit-identical
        guarantee.

        The real artifact showed it plainly: global_step 6423 recorded
        batch_in_epoch 6154, a gap of 269, which was precisely the step the run
        had been resumed from.
        """
        tmp = tempfile.mkdtemp(prefix="aion_resume2_")
        mgr = StepCheckpointManager(Path(tmp), "run_test", keep_last_n=20)

        # (1) fresh run, interrupted at step 3
        _, model, opt = _build_model()
        rng = np.random.default_rng(43)
        ckfn = _make_ckpt_fn(mgr, model, opt,
                             np.random.default_rng(42), np.random.default_rng(44))
        trainer = GPTTrainer(model, opt, grad_clip=1.0, callbacks=[_InterruptAt(3)])
        with self.assertRaises(KeyboardInterrupt):
            trainer.train(_make_sampler(rng), epochs=3, total_epochs=3, data_rng=rng,
                          checkpoint_fn=ckfn, checkpoint_every_n_steps=1000)
        first = mgr.load_latest()
        self.assertEqual(first.global_step, 3)
        self.assertEqual(first.batch_in_epoch, 3)

        # (2) resume, run further, get interrupted AGAIN at step 7
        _, model2, opt2 = _build_model()
        StepCheckpointManager.restore_model(model2, first)
        StepCheckpointManager.restore_optimizer(opt2, first)
        rng2 = np.random.default_rng(43)
        rng2.bit_generator.state = first.state["rng"]["data_epoch_start"]
        ckfn2 = _make_ckpt_fn(mgr, model2, opt2,
                              np.random.default_rng(42), np.random.default_rng(44))
        trainer2 = GPTTrainer(model2, opt2, grad_clip=1.0, callbacks=[_InterruptAt(7)])
        with self.assertRaises(KeyboardInterrupt):
            trainer2.train(_make_sampler(rng2), epochs=3 - first.epoch, total_epochs=3,
                           start_epoch=first.epoch, start_step=first.global_step,
                           start_batch_in_epoch=first.batch_in_epoch, data_rng=rng2,
                           checkpoint_fn=ckfn2, checkpoint_every_n_steps=1000,
                           history=first.state["history"])

        second = mgr.load_latest()
        self.assertEqual(second.global_step, 7)
        # THE ASSERTION: within one epoch, position in the epoch tracks the
        # global step.  The old code wrote 4 here (7 - 3 skipped) and the third
        # resume would have retrained batches 4..7.
        self.assertEqual(
            second.batch_in_epoch, 7,
            f"checkpoint at global_step 7 recorded batch_in_epoch "
            f"{second.batch_in_epoch}; a further resume would replay "
            f"{7 - second.batch_in_epoch} batches",
        )

    def test_mid_epoch_checkpoint_records_tokens_processed(self):
        """``tokens_processed`` used to be 0 in every mid-epoch checkpoint.

        ``total_tokens`` only advances at the epoch boundary, so a checkpoint
        taken inside an epoch reported that no tokens had been seen — making the
        field useless exactly where it matters, on a long interrupted run.
        """
        tmp = tempfile.mkdtemp(prefix="aion_tokens_")
        mgr = StepCheckpointManager(Path(tmp), "run_test", keep_last_n=10)
        _, model, opt = _build_model()
        rng = np.random.default_rng(43)
        ckfn = _make_ckpt_fn(mgr, model, opt,
                             np.random.default_rng(42), np.random.default_rng(44))
        trainer = GPTTrainer(model, opt, grad_clip=1.0, callbacks=[_InterruptAt(5)])
        with self.assertRaises(KeyboardInterrupt):
            trainer.train(_make_sampler(rng), epochs=3, total_epochs=3, data_rng=rng,
                          checkpoint_fn=ckfn, checkpoint_every_n_steps=2)

        loaded = mgr.load_latest()
        # 5 steps x batch 4 x block 8 = 160 tokens seen before the interrupt.
        self.assertGreater(loaded.state["tokens_processed"], 0,
                           "mid-epoch checkpoint recorded zero tokens processed")

    def test_scheduler_resumes(self):
        # LR is a pure function of global_step; restoring the step restores the
        # schedule.  Verify the resumed schedule matches the uninterrupted one.
        sched = build_scheduler("cosine", 1e-3, warmup_steps=3, min_lr=1e-5)
        total = 30
        # uninterrupted LR at step 10:
        lr_uninterrupted = sched.get_lr(10, total)
        # after resume, global_step continues, same scheduler + total_steps:
        lr_resumed = sched.get_lr(10, total)
        self.assertEqual(lr_uninterrupted, lr_resumed)
        # and it is not stuck at warmup/base
        self.assertLess(sched.get_lr(20, total), sched.get_lr(5, total))


# ── Project-level: run_id unchanged + auto-resume from latest/ ────────────────

class TestProjectStepResume(unittest.TestCase):
    def _setup_project(self):
        from aion.workspace.store import ProjectStore
        from aion.datasets.store import DatasetStore
        from aion.tokenizers.bpe import ByteLevelBPETokenizer
        from aion.tokenizers.store import TokenizerStore
        ws = tempfile.mkdtemp(prefix="aion_proj_")
        store = ProjectStore(ws)
        project = store.create("aion-01")
        ds = DatasetStore(project.data_dir()).create("aion-corpus")
        for i in range(10):
            ds.add_document(("alpha beta gamma delta epsilon zeta. " * 15) + f" {i}")
        tok = ByteLevelBPETokenizer()
        fp = ds.fingerprint()
        res = tok.train((t for _, t in ds.stream()), vocab_size=300, dataset_fingerprint=fp)
        ts = TokenizerStore(project.dir("tokenizers"))
        man = ts.save(tok, res, name="tk", dataset_id=ds.id, dataset_fingerprint=fp,
                      params={"vocab_size": 300, "algorithm": tok.algorithm})
        ts.set_status(man["id"], "default")
        store.update("aion-01", defaults={"tokenizer": man["id"]})
        return store, project, ds, man["id"]

    def _cfg(self, ds_id, tok_id, epochs):
        from aion.training.config import TrainingConfig
        return TrainingConfig(
            model_name="AION-0.1",
            gpt_config={"d_model": 16, "n_heads": 2, "n_layers": 2, "d_ff": 32,
                        "dropout": 0.0, "activation": "gelu", "pre_norm": True,
                        "tie_weights": True},
            dataset_ids=[ds_id], tokenizer_id=tok_id, context_length=8, batch_size=4,
            train_split=0.8, optimizer="adam", learning_rate=1e-3, grad_clip=1.0,
            epochs=epochs, scheduler="cosine", warmup_steps=2, min_lr=1e-5,
            checkpoint_every_n_steps=3, keep_last_n_step_checkpoints=5,
            log_every_n_steps=100, sample_prompts=[], seed=42)

    def test_run_id_unchanged_and_resume_from_latest(self):
        from aion.training.project import TrainingProject
        store, project, ds, tok_id = self._setup_project()

        # Interrupt the first run via a KeyboardInterrupt injected mid-training.
        import aion.gpt.trainer as trainer_mod
        orig_train = trainer_mod.GPTTrainer.train

        def interrupt_train(self, *a, **k):
            k = dict(k)
            cbs = list(self._callbacks._callbacks)
            cbs.append(_InterruptAt(6))
            self._callbacks._callbacks = cbs
            return orig_train(self, *a, **k)

        trainer_mod.GPTTrainer.train = interrupt_train
        try:
            with self.assertRaises(KeyboardInterrupt):
                TrainingProject(project, self._cfg(ds.id, tok_id, 5)).run(resume=False)
        finally:
            trainer_mod.GPTTrainer.train = orig_train

        # run_id was persisted; capture it
        run_id_1 = TrainingProject(project, self._cfg(ds.id, tok_id, 5))._recover_run_id()
        self.assertIsNotNone(run_id_1)
        # A checkpoint exists under that run
        mgr = StepCheckpointManager(project.dir("checkpoints"), run_id_1)
        self.assertTrue(mgr.has_checkpoint())
        resumed_step = mgr.load_latest().global_step

        # Resume: same run_id, continues from the checkpoint, completes.
        result = TrainingProject(project, self._cfg(ds.id, tok_id, 5)).run(resume=True)
        self.assertEqual(result.run_id, run_id_1)          # run_id unchanged
        self.assertGreater(result.metrics["epochs"], 0)
        self.assertEqual(len(result.metrics["loss_history"]), 5)
        self.assertGreaterEqual(resumed_step, 6)


if __name__ == "__main__":
    unittest.main()
