"""End-to-end tests for TrainingProject.

Uses a real (tiny) model, a synthetic corpus, and a real project directory
in a temp folder.  Validates that run() completes and returns a well-formed
TrainingRunResult with all expected artifacts on disk.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aion.training.config import TrainingConfig
from aion.training.project import TrainingProject, TrainingRunResult
from aion.workspace.store import ProjectStore


# ── Minimal fake tokenizer that satisfies the TokenizerStore interface ─────────

class _FakeTokenizer:
    algorithm = "bpe-byte-v1"
    vocab_size = 64
    # Special tokens occupy the fixed low ids 0-3, matching the real tokenizer.
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids)

    def save(self, model_dir: Path) -> None:
        import json
        (model_dir / "vocabulary.json").write_text(
            json.dumps({"vocab": list(range(self.vocab_size))}), encoding="utf-8"
        )
        (model_dir / "merges.json").write_text("[]", encoding="utf-8")


def _setup_project(tmp: str):
    """Create a minimal project with one dataset and one tokenizer."""
    store = ProjectStore(Path(tmp) / "workspace")
    project = store.create("test-project")

    # Dataset
    from aion.datasets.store import DatasetStore
    ds_store = DatasetStore(project.data_dir())
    ds = ds_store.create("corpus")
    # Add enough text to produce a usable corpus
    text = "the quick brown fox jumps over the lazy dog " * 200
    ds.add_document(text)

    # Tokenizer — write directly into the tokenizer store layout
    from aion.tokenizers.store import TokenizerStore
    import uuid, json as _json
    from aion.util import fingerprint, now_iso, slugify
    ts_dir = project.dir("tokenizers")
    tok = _FakeTokenizer()
    tok_id = f"fake-tok-{uuid.uuid4().hex[:8]}"
    tok_root = ts_dir / tok_id
    model_dir = tok_root / "model"
    training_dir = tok_root / "training"
    model_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    tok.save(model_dir)
    vocab_text = (model_dir / "vocabulary.json").read_text(encoding="utf-8")
    vocab_fp = fingerprint(vocab_text)
    manifest = {
        "schema_version": 1,
        "id": tok_id,
        "name": "fake-tok",
        "description": "",
        "algorithm": "bpe-byte-v1",
        "dataset_id": ds.id,
        "dataset_fingerprint": "",
        "vocab_size": tok.vocab_size,
        "merge_count": 0,
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "params": {},
        "status": "experimental",
        "created_at": now_iso(),
        "vocabulary_fingerprint": vocab_fp,
        "metrics": {},
        "produced_by": {"aion_version": "0.0", "git_commit": None},
    }
    (tok_root / "manifest.json").write_text(
        _json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (training_dir / "statistics.json").write_text("{}", encoding="utf-8")

    # Patch TokenizerStore.load to return our fake tokenizer
    from aion.tokenizers import store as ts_mod
    original_load = ts_mod.TokenizerStore.load

    def _patched_load(self, tokenizer_id):
        if tokenizer_id == tok_id:
            return tok
        return original_load(self, tokenizer_id)

    ts_mod.TokenizerStore.load = _patched_load

    return project, ds.id, tok_id, original_load, ts_mod


def _minimal_config(dataset_id: str, tokenizer_id: str) -> TrainingConfig:
    return TrainingConfig(
        model_name="test-model",
        gpt_config={
            "d_model": 16, "n_heads": 2, "n_layers": 1,
            "d_ff": 32, "dropout": 0.0, "activation": "gelu",
            "tie_weights": True, "pre_norm": True,
        },
        dataset_ids=[dataset_id],
        tokenizer_id=tokenizer_id,
        context_length=16,
        batch_size=4,
        train_split=0.9,
        optimizer="adam",
        learning_rate=1e-3,
        grad_clip=1.0,
        epochs=2,
        scheduler="constant",
        warmup_steps=0,
        min_lr=1e-5,
        checkpoint_every_n_epochs=1,
        keep_last_n_checkpoints=2,
        eval_every_n_epochs=1,
        sample_every_n_epochs=1,
        sample_prompts=[],
        sample_max_new_tokens=8,
        seed=42,
    )


class TestTrainingProject(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        result = _setup_project(self._tmp.name)
        self._project = result[0]
        self._dataset_id = result[1]
        self._tokenizer_id = result[2]
        self._original_load = result[3]
        self._ts_mod = result[4]

    def tearDown(self):
        # Restore patched method
        self._ts_mod.TokenizerStore.load = self._original_load
        self._tmp.cleanup()

    def _run(self, **overrides):
        cfg = _minimal_config(self._dataset_id, self._tokenizer_id)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        tp = TrainingProject(self._project, cfg)
        return tp.run()

    def test_run_returns_result(self):
        result = self._run()
        self.assertIsInstance(result, TrainingRunResult)

    def test_model_id_non_empty(self):
        result = self._run()
        self.assertTrue(result.model_id)

    def test_metrics_loss_history_length(self):
        result = self._run()
        self.assertEqual(len(result.metrics["loss_history"]), 2)

    def test_model_saved_to_disk(self):
        result = self._run()
        model_dir = self._project.dir("models") / result.model_id
        self.assertTrue((model_dir / "manifest.json").is_file())
        self.assertTrue((model_dir / "model" / "weights.npz").is_file())

    def test_model_card_written(self):
        result = self._run()
        model_dir = self._project.dir("models") / result.model_id
        self.assertTrue((model_dir / "model_card.md").is_file())
        self.assertTrue((model_dir / "model_card.json").is_file())

    def test_model_card_markdown_non_empty(self):
        result = self._run()
        self.assertIn("test-model", result.model_card_markdown)

    def test_experiment_record_present(self):
        result = self._run()
        self.assertIn("id", result.experiment_record)
        self.assertEqual(result.experiment_record["type"], "gpt-language-model")

    def test_corpus_fingerprint_non_empty(self):
        result = self._run()
        self.assertTrue(result.corpus_fingerprint)

    def test_corpus_stats_present(self):
        result = self._run()
        self.assertIn("n_documents", result.corpus_stats)
        self.assertGreater(result.corpus_stats["n_documents"], 0)

    def test_checkpoint_dir_exists(self):
        result = self._run()
        self.assertTrue(Path(result.checkpoint_dir).is_dir())

    def test_log_path_exists(self):
        result = self._run()
        self.assertTrue(Path(result.log_path).is_file())

    def test_config_snapshot_written(self):
        result = self._run()
        log_dir = Path(result.log_path).parent
        self.assertTrue((log_dir / "config.json").is_file())

    def test_rng_state_written(self):
        result = self._run()
        log_dir = Path(result.log_path).parent
        # At least one rng_state file should exist after 2 epochs
        rng_files = list(log_dir.glob("rng_state_epoch_*.json"))
        self.assertGreater(len(rng_files), 0)

    def test_adam_optimizer(self):
        result = self._run(optimizer="adam")
        self.assertIsInstance(result, TrainingRunResult)

    def test_sgd_optimizer(self):
        result = self._run(optimizer="sgd")
        self.assertIsInstance(result, TrainingRunResult)

    def test_cosine_scheduler(self):
        result = self._run(scheduler="cosine", warmup_steps=2)
        self.assertIsInstance(result, TrainingRunResult)

    def test_sample_prompts_empty(self):
        result = self._run(sample_prompts=[])
        self.assertIsInstance(result, TrainingRunResult)


if __name__ == "__main__":
    unittest.main()
