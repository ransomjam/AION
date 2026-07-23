import unittest
import tempfile
from pathlib import Path
from aion.gpt.config import GPTConfig
from aion.gpt.experiment import EXPERIMENT_TYPE, record_gpt_experiment
from aion.workspace.experiments import ExperimentStore


class TestRecordGPTExperiment(unittest.TestCase):

    def _store(self, tmp):
        return ExperimentStore(Path(tmp) / "experiments")

    def test_record_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cfg = GPTConfig(vocab_size=32, d_model=16, n_heads=2, n_layers=1, d_ff=32)
            record = record_gpt_experiment(
                store,
                model_id="gpt-abc123",
                config=cfg.to_dict(),
                dataset_id="ds-1",
                dataset_fingerprint="fp-ds",
                tokenizer_id="tok-1",
                tokenizer_fingerprint="fp-tok",
                epochs=3,
                tokens_processed=5000,
                context_length=64,
                param_count=10000,
                trainable_param_count=10000,
                final_loss=2.5,
                loss_history=[3.0, 2.7, 2.5],
            )
            self.assertIsInstance(record, dict)
            self.assertEqual(record["type"], EXPERIMENT_TYPE)

    def test_experiment_type_string(self):
        self.assertEqual(EXPERIMENT_TYPE, "gpt-language-model")

    def test_metrics_fields_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cfg = GPTConfig()
            record = record_gpt_experiment(
                store,
                model_id="gpt-xyz",
                config=cfg.to_dict(),
                dataset_id="ds-2",
                dataset_fingerprint="fp2",
                tokenizer_id="tok-2",
                tokenizer_fingerprint="fp-tok2",
                epochs=1,
                tokens_processed=1000,
                context_length=32,
                param_count=500,
                trainable_param_count=500,
                final_loss=3.0,
                loss_history=[3.0],
                final_val_loss=3.2,
                final_val_perplexity=24.5,
                val_loss_history=[3.2],
                val_perplexity_history=[24.5],
                training_time_s=10.5,
                tokens_per_sec=[100.0],
                grad_norm_history=[0.8],
                seed=42,
            )
            m = record["metrics"]
            self.assertEqual(m["tokens_processed"], 1000)
            self.assertEqual(m["context_length"], 32)
            self.assertAlmostEqual(m["final_val_perplexity"], 24.5)
            self.assertEqual(m["tokenizer_fingerprint"], "fp-tok2")
            self.assertEqual(m["seed"], 42)

    def test_record_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cfg = GPTConfig()
            record = record_gpt_experiment(
                store,
                model_id="gpt-persist",
                config=cfg.to_dict(),
                dataset_id="ds-3",
                dataset_fingerprint="fp3",
                tokenizer_id="tok-3",
                tokenizer_fingerprint="fp-tok3",
                epochs=1,
                tokens_processed=200,
                context_length=16,
                param_count=100,
                trainable_param_count=100,
                final_loss=2.0,
                loss_history=[2.0],
            )
            retrieved = store.get(record["id"])
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["type"], EXPERIMENT_TYPE)

    def test_params_includes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cfg = GPTConfig(vocab_size=64)
            record = record_gpt_experiment(
                store,
                model_id="gpt-cfg",
                config=cfg.to_dict(),
                dataset_id="ds-4",
                dataset_fingerprint="fp4",
                tokenizer_id="tok-4",
                tokenizer_fingerprint="fp-tok4",
                epochs=1,
                tokens_processed=100,
                context_length=8,
                param_count=50,
                trainable_param_count=50,
                final_loss=1.5,
                loss_history=[1.5],
            )
            self.assertEqual(record["params"]["config"]["vocab_size"], 64)


if __name__ == "__main__":
    unittest.main()
