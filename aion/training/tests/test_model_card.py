import unittest
from aion.training.model_card import ModelCard, ModelCardResult


def _card(**overrides):
    defaults = dict(
        model_name="AION-0.1",
        gpt_config={
            "d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 512,
            "max_seq_len": 512, "vocab_size": 1000, "activation": "gelu",
            "tie_weights": True, "pre_norm": True,
        },
        param_count=1_234_567,
        tokenizer_manifest={
            "id": "bpe-abc123", "name": "bpe-1k", "algorithm": "bpe-byte-v1",
            "vocab_size": 1000, "vocabulary_fingerprint": "abcdef1234567890",
        },
        corpus_stats={
            "n_documents": 500, "n_train_tokens": 90000, "n_val_tokens": 10000,
            "n_total_tokens": 100000, "vocab_coverage": 0.85,
            "mean_doc_length": 200.0, "median_doc_length": 180.0,
            "max_doc_length": 1000, "min_doc_length": 10,
        },
        corpus_fingerprint="fp1234567890abcd",
        dataset_ids=["ds-001", "ds-002"],
        training_config={
            "model_name": "AION-0.1", "optimizer": "adam",
            "learning_rate": 3e-4, "scheduler": "cosine",
            "warmup_steps": 100, "min_lr": 1e-5, "epochs": 10,
            "batch_size": 8, "context_length": 512, "grad_clip": 1.0,
            "seed": 42, "train_split": 0.9,
        },
        metrics={
            "final_loss": 2.5, "final_val_loss": 2.7,
            "final_val_perplexity": 14.88,
            "loss_history": [3.0, 2.8, 2.5],
            "val_loss_history": [3.2, 2.9, 2.7],
            "training_time_s": 120.5,
            "tokens_processed": 900000,
        },
        model_id="aion-0-1-abc12345",
    )
    defaults.update(overrides)
    return ModelCard(**defaults)


class TestModelCard(unittest.TestCase):

    def test_generate_returns_result(self):
        result = _card().generate()
        self.assertIsInstance(result, ModelCardResult)

    def test_markdown_is_string(self):
        result = _card().generate()
        self.assertIsInstance(result.markdown, str)

    def test_data_is_dict(self):
        result = _card().generate()
        self.assertIsInstance(result.data, dict)

    def test_markdown_contains_model_name(self):
        result = _card().generate()
        self.assertIn("AION-0.1", result.markdown)

    def test_data_schema_version(self):
        result = _card().generate()
        self.assertEqual(result.data["schema_version"], 1)

    def test_data_model_section(self):
        result = _card().generate()
        m = result.data["model"]
        self.assertEqual(m["name"], "AION-0.1")
        self.assertEqual(m["param_count"], 1_234_567)
        self.assertEqual(m["id"], "aion-0-1-abc12345")

    def test_data_architecture_fields(self):
        result = _card().generate()
        arch = result.data["architecture"]
        self.assertEqual(arch["d_model"], 128)
        self.assertEqual(arch["n_heads"], 4)
        self.assertEqual(arch["n_layers"], 2)

    def test_data_tokenizer_fields(self):
        result = _card().generate()
        tok = result.data["tokenizer"]
        self.assertEqual(tok["id"], "bpe-abc123")
        self.assertEqual(tok["vocab_size"], 1000)

    def test_data_corpus_fields(self):
        result = _card().generate()
        corpus = result.data["corpus"]
        self.assertEqual(corpus["n_documents"], 500)
        self.assertIn("ds-001", corpus["dataset_ids"])
        self.assertEqual(corpus["fingerprint"], "fp1234567890abcd")

    def test_data_training_fields(self):
        result = _card().generate()
        train = result.data["training"]
        self.assertEqual(train["optimizer"], "adam")
        self.assertAlmostEqual(train["learning_rate"], 3e-4)
        self.assertEqual(train["seed"], 42)

    def test_data_evaluation_fields(self):
        result = _card().generate()
        ev = result.data["evaluation"]
        self.assertAlmostEqual(ev["final_loss"], 2.5)
        self.assertAlmostEqual(ev["final_val_perplexity"], 14.88)

    def test_data_limitations_list(self):
        result = _card().generate()
        self.assertIsInstance(result.data["limitations"], list)
        self.assertGreater(len(result.data["limitations"]), 0)

    def test_custom_limitations(self):
        result = _card(limitations=["custom limit"]).generate()
        self.assertIn("custom limit", result.data["limitations"])

    def test_data_license(self):
        result = _card(license_name="MIT").generate()
        self.assertEqual(result.data["license"], "MIT")

    def test_markdown_contains_architecture_table(self):
        result = _card().generate()
        self.assertIn("## Architecture", result.markdown)
        self.assertIn("d_model", result.markdown)

    def test_markdown_contains_evaluation_section(self):
        result = _card().generate()
        self.assertIn("## Evaluation Metrics", result.markdown)
        self.assertIn("final_loss", result.markdown)

    def test_generated_at_present(self):
        result = _card().generate()
        self.assertIn("generated_at", result.data)
        self.assertTrue(result.data["generated_at"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
