"""Tests for EmbeddingStore: save, list, open, load, delete, set_status."""

import tempfile
import unittest
from pathlib import Path

from aion.embeddings.cbow import CBOWEmbedding
from aion.embeddings.store import EmbeddingNotFound, EmbeddingStore
from aion.tokenizers.bpe import ByteLevelBPETokenizer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "low lower lowest",
]

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = ByteLevelBPETokenizer()
        _TOK.train(CORPUS, vocab_size=280)
    return _TOK


def _train():
    tok = _tokenizer()
    emb = CBOWEmbedding()
    result = emb.train(CORPUS, tokenizer=tok, dims=4, epochs=1,
                       window=1, neg_samples=2, seed=0)
    return emb, result


class TestEmbeddingStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = EmbeddingStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_list(self):
        emb, result = _train()
        manifest = self.store.save(emb, result, name="test-emb",
                                   tokenizer_id="tok1", dataset_id="ds1")
        self.assertEqual(manifest["name"], "test-emb")
        self.assertEqual(manifest["algorithm"], "cbow-v1")
        listed = self.store.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], manifest["id"])

    def test_open_manifest(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["id"], m["id"])
        self.assertEqual(reopened["dims"], 4)

    def test_not_found_raises(self):
        with self.assertRaises(EmbeddingNotFound):
            self.store.open_manifest("nonexistent")

    def test_load_embedding(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        loaded = self.store.load(m["id"])
        self.assertIsInstance(loaded, CBOWEmbedding)
        self.assertEqual(loaded.encode(0), emb.encode(0))

    def test_statistics_saved(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        stats = self.store.statistics(m["id"])
        self.assertIsNotNone(stats)
        self.assertIn("loss_history", stats)
        self.assertIn("final_loss", stats)

    def test_delete(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        self.store.delete(m["id"])
        self.assertFalse(self.store.exists(m["id"]))

    def test_set_status(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        updated = self.store.set_status(m["id"], "default")
        self.assertEqual(updated["status"], "default")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["status"], "default")

    def test_embedding_fingerprint_present(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        self.assertIn("embedding_fingerprint", m)
        self.assertTrue(len(m["embedding_fingerprint"]) > 0)

    def test_tokenizer_link_stored(self):
        emb, result = _train()
        m = self.store.save(emb, result, name="e1",
                            tokenizer_id="tok-abc", tokenizer_fingerprint="fp123")
        self.assertEqual(m["tokenizer_id"], "tok-abc")
        self.assertEqual(m["tokenizer_fingerprint"], "fp123")

    def test_metrics_not_include_loss_history_in_manifest(self):
        # loss_history is large; it lives in statistics.json, not the manifest
        emb, result = _train()
        m = self.store.save(emb, result, name="e1")
        self.assertNotIn("loss_history", m.get("metrics", {}))


if __name__ == "__main__":
    unittest.main()
