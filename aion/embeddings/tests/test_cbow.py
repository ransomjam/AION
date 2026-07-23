"""Tests for CBOWEmbedding: training, encoding, persistence, metrics."""

import json
import tempfile
import unittest
from pathlib import Path

from aion.embeddings.base import Embedding, EmbeddingResult
from aion.embeddings.cbow import CBOWEmbedding
from aion.tokenizers.bpe import ByteLevelBPETokenizer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog played",
    "low lower lowest new newer newest",
    "the quick brown fox jumps over the lazy dog",
]

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = ByteLevelBPETokenizer()
        _TOK.train(CORPUS, vocab_size=300)
    return _TOK


def _train(dims=8, epochs=2, seed=42):
    tok = _tokenizer()
    emb = CBOWEmbedding()
    result = emb.train(
        CORPUS, tokenizer=tok,
        dims=dims, epochs=epochs, window=2,
        neg_samples=3, seed=seed,
    )
    return emb, result


class TestCBOWAbstraction(unittest.TestCase):
    def test_is_abstract(self):
        import inspect
        self.assertTrue(inspect.isabstract(Embedding))

    def test_cbow_is_concrete(self):
        import inspect
        self.assertFalse(inspect.isabstract(CBOWEmbedding))

    def test_algorithm_identifier(self):
        self.assertEqual(CBOWEmbedding.algorithm, "cbow-v1")


class TestCBOWTraining(unittest.TestCase):
    def setUp(self):
        self.emb, self.result = _train()

    def test_result_is_embedding_result(self):
        self.assertIsInstance(self.result, EmbeddingResult)

    def test_metrics_keys_present(self):
        for key in ("vocab_size", "dims", "epochs", "final_loss",
                    "loss_history", "training_time_s", "nn_coherence", "coverage"):
            self.assertIn(key, self.result.metrics, f"missing metric: {key}")

    def test_loss_history_length_equals_epochs(self):
        self.assertEqual(len(self.result.metrics["loss_history"]), 2)

    def test_loss_is_finite(self):
        import math
        self.assertTrue(math.isfinite(self.result.metrics["final_loss"]))

    def test_coverage_between_0_and_1(self):
        c = self.result.metrics["coverage"]
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_vocab_size_matches_tokenizer(self):
        self.assertEqual(self.result.metrics["vocab_size"], _tokenizer().vocab_size)

    def test_dims_property(self):
        self.assertEqual(self.emb.dims, 8)

    def test_vocab_size_property(self):
        self.assertEqual(self.emb.vocab_size, _tokenizer().vocab_size)


class TestCBOWEncoding(unittest.TestCase):
    def setUp(self):
        self.emb, _ = _train()

    def test_encode_returns_list_of_floats(self):
        vec = self.emb.encode(0)
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 8)
        self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_encode_out_of_range_returns_zeros(self):
        vec = self.emb.encode(99999)
        self.assertEqual(vec, [0.0] * 8)

    def test_most_similar_returns_n_results(self):
        results = self.emb.most_similar(0, n=5)
        self.assertLessEqual(len(results), 5)
        for tid, sim in results:
            self.assertIsInstance(tid, int)
            self.assertIsInstance(sim, float)
            self.assertGreaterEqual(sim, -1.01)
            self.assertLessEqual(sim, 1.01)

    def test_most_similar_excludes_query(self):
        results = self.emb.most_similar(0, n=5)
        ids = [r[0] for r in results]
        self.assertNotIn(0, ids)

    def test_embed_text(self):
        tok = _tokenizer()
        vec = self.emb.embed_text("the cat", tok)
        self.assertEqual(len(vec), 8)


class TestCBOWDeterminism(unittest.TestCase):
    def test_same_seed_same_vectors(self):
        emb1, _ = _train(seed=7)
        emb2, _ = _train(seed=7)
        self.assertEqual(emb1._W[0], emb2._W[0])

    def test_different_seed_different_vectors(self):
        emb1, _ = _train(seed=1)
        emb2, _ = _train(seed=2)
        self.assertNotEqual(emb1._W[0], emb2._W[0])


class TestCBOWPersistence(unittest.TestCase):
    def test_save_and_load(self):
        emb, _ = _train()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            emb.save(model_dir)
            self.assertTrue((model_dir / "vectors.json").is_file())
            loaded = CBOWEmbedding.load(model_dir)
            self.assertEqual(loaded.vocab_size, emb.vocab_size)
            self.assertEqual(loaded.dims, emb.dims)
            self.assertEqual(loaded.encode(0), emb.encode(0))

    def test_vectors_json_is_valid(self):
        emb, _ = _train()
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            emb.save(model_dir)
            data = json.loads((model_dir / "vectors.json").read_text(encoding="utf-8"))
            self.assertEqual(data["algorithm"], "cbow-v1")
            self.assertIsInstance(data["vectors"], list)
            self.assertEqual(len(data["vectors"]), emb.vocab_size)

    def test_progress_fn_called(self):
        calls = []
        tok = _tokenizer()
        emb = CBOWEmbedding()
        emb.train(CORPUS, tokenizer=tok, dims=4, epochs=2, window=1,
                  neg_samples=2, seed=0,
                  progress_fn=lambda f, m: calls.append(f))
        self.assertEqual(len(calls), 2)  # one call per epoch
        self.assertAlmostEqual(calls[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
