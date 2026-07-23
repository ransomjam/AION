import unittest
import tempfile
from pathlib import Path
from aion.inference.experiment import EXPERIMENT_TYPE, record_inference_experiment
from aion.inference.config import GenerationConfig
from aion.workspace.experiments import ExperimentStore


class TestRecordInferenceExperiment(unittest.TestCase):

    def _store(self, tmp):
        return ExperimentStore(Path(tmp) / "experiments")

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            record = record_inference_experiment(
                store,
                model_id="gpt-abc",
                tokenizer_id="tok-1",
                config=GenerationConfig().to_dict(),
                prompt_tokens=10,
                generated_tokens=50,
                generation_time_s=0.5,
                tokens_per_sec=100.0,
                mean_entropy=2.3,
                mean_top1_prob=0.4,
                stopped_by="max_new_tokens",
                strategy="greedy",
            )
            self.assertIsInstance(record, dict)

    def test_experiment_type(self):
        self.assertEqual(EXPERIMENT_TYPE, "gpt-inference")

    def test_metrics_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            record = record_inference_experiment(
                store,
                model_id="gpt-abc",
                tokenizer_id="tok-1",
                config=GenerationConfig().to_dict(),
                prompt_tokens=5,
                generated_tokens=20,
                generation_time_s=0.2,
                tokens_per_sec=100.0,
                mean_entropy=1.5,
                mean_top1_prob=0.6,
                stopped_by="eos",
                strategy="top_p",
            )
            m = record["metrics"]
            self.assertEqual(m["prompt_tokens"], 5)
            self.assertEqual(m["generated_tokens"], 20)
            self.assertEqual(m["total_tokens"], 25)
            self.assertEqual(m["stopped_by"], "eos")
            self.assertEqual(m["strategy"], "top_p")

    def test_record_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            record = record_inference_experiment(
                store,
                model_id="gpt-xyz",
                tokenizer_id="tok-2",
                config=GenerationConfig().to_dict(),
                prompt_tokens=3,
                generated_tokens=10,
                generation_time_s=0.1,
                tokens_per_sec=100.0,
                mean_entropy=2.0,
                mean_top1_prob=0.5,
                stopped_by="max_new_tokens",
                strategy="greedy",
            )
            retrieved = store.get(record["id"])
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["type"], EXPERIMENT_TYPE)

    def test_params_includes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            cfg = GenerationConfig(max_new_tokens=64, strategy="temperature")
            record = record_inference_experiment(
                store,
                model_id="gpt-cfg",
                tokenizer_id="tok-3",
                config=cfg.to_dict(),
                prompt_tokens=2,
                generated_tokens=5,
                generation_time_s=0.05,
                tokens_per_sec=100.0,
                mean_entropy=1.8,
                mean_top1_prob=0.45,
                stopped_by="max_new_tokens",
                strategy="temperature",
            )
            self.assertEqual(record["params"]["config"]["strategy"], "temperature")
            self.assertEqual(record["params"]["strategy"], "temperature")


if __name__ == "__main__":
    unittest.main()
