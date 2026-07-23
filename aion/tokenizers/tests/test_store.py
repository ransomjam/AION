"""Tests for TokenizerStore — save, list, open, delete, set_default."""

import tempfile
import unittest
from pathlib import Path

from aion.tokenizers.bpe import ByteLevelBPETokenizer
from aion.tokenizers.store import TokenizerNotFound, TokenizerStore

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "low lower lowest new newer newest",
]


def _train(vocab_size=280):
    tok = ByteLevelBPETokenizer()
    result = tok.train(CORPUS, vocab_size=vocab_size)
    return tok, result


class TestTokenizerStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TokenizerStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_list(self):
        tok, result = _train()
        manifest = self.store.save(tok, result, name="test-bpe",
                                   dataset_id="ds1", dataset_fingerprint="fp1")
        self.assertEqual(manifest["name"], "test-bpe")
        self.assertEqual(manifest["algorithm"], "bpe-byte-v1")
        listed = self.store.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], manifest["id"])

    def test_open_manifest(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["id"], m["id"])

    def test_not_found_raises(self):
        with self.assertRaises(TokenizerNotFound):
            self.store.open_manifest("nonexistent")

    def test_load_tokenizer(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        loaded = self.store.load(m["id"])
        self.assertIsInstance(loaded, ByteLevelBPETokenizer)
        self.assertEqual(loaded.encode("the cat"), tok.encode("the cat"))

    def test_statistics_saved(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        stats = self.store.statistics(m["id"])
        self.assertIsNotNone(stats)
        self.assertIn("merge_history", stats)
        self.assertIn("compression_ratio", stats)

    def test_delete(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        self.store.delete(m["id"])
        self.assertFalse(self.store.exists(m["id"]))

    def test_set_status(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        updated = self.store.set_status(m["id"], "default")
        self.assertEqual(updated["status"], "default")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["status"], "default")

    def test_vocabulary_fingerprint_present(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1")
        self.assertIn("vocabulary_fingerprint", m)
        self.assertTrue(len(m["vocabulary_fingerprint"]) > 0)

    def test_params_stored(self):
        tok, result = _train()
        m = self.store.save(tok, result, name="t1",
                            params={"vocab_size": 280, "algorithm": "bpe-byte-v1"})
        self.assertEqual(m["params"]["vocab_size"], 280)


if __name__ == "__main__":
    unittest.main()
