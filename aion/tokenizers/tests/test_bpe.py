"""Tests for the Tokenizer abstraction and ByteLevelBPETokenizer."""

import json
import tempfile
import unittest
from pathlib import Path

from aion.tokenizers.base import MergeStep, Tokenizer, TrainingResult
from aion.tokenizers.bpe import (
    ByteLevelBPETokenizer,
    _build_byte_encoder,
    _count_pairs,
    _merge_pair,
    _word_to_bytes,
    _symbols_to_str,
)


TINY_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog",
    "low lower lowest",
    "new newer newest",
]


class TestByteEncoder(unittest.TestCase):
    def test_all_256_bytes_mapped(self):
        enc = _build_byte_encoder()
        self.assertEqual(len(enc), 256)
        self.assertEqual(len(set(enc.values())), 256)

    def test_roundtrip_ascii(self):
        word = "hello"
        syms = _word_to_bytes(word)
        self.assertEqual(_symbols_to_str(syms), word)

    def test_roundtrip_unicode(self):
        word = "café"
        syms = _word_to_bytes(word)
        self.assertEqual(_symbols_to_str(syms), word)


class TestBPEHelpers(unittest.TestCase):
    def test_count_pairs(self):
        wf = {("a", "b", "c"): 2, ("a", "b"): 3}
        counts = _count_pairs(wf)
        self.assertEqual(counts[("a", "b")], 5)
        self.assertEqual(counts[("b", "c")], 2)

    def test_merge_pair(self):
        wf = {("a", "b", "c", "b"): 1}
        merged = _merge_pair(wf, ("a", "b"), "ab")
        self.assertIn(("ab", "c", "b"), merged)

    def test_merge_pair_multiple_occurrences(self):
        wf = {("a", "b", "a", "b"): 1}
        merged = _merge_pair(wf, ("a", "b"), "ab")
        self.assertIn(("ab", "ab"), merged)


class TestByteLevelBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ByteLevelBPETokenizer()
        self.result = self.tok.train(TINY_CORPUS, vocab_size=300)

    def test_algorithm_identifier(self):
        self.assertEqual(self.tok.algorithm, "bpe-byte-v1")

    def test_vocab_size_at_least_target(self):
        # vocab_size is a ceiling; actual may be less if corpus exhausted
        self.assertGreaterEqual(self.tok.vocab_size, 260)
        self.assertLessEqual(self.tok.vocab_size, 300)

    def test_encode_returns_ints(self):
        ids = self.tok.encode("the cat")
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertGreater(len(ids), 0)

    def test_decode_roundtrip(self):
        text = "the cat sat"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        # Pre-tokenization strips spaces, so decoded is the concatenation of
        # the pre-tokens without spaces.  Words should be present.
        self.assertIn("cat", decoded)
        self.assertIn("sat", decoded)

    def test_encode_unknown_bytes_no_crash(self):
        # Byte-level BPE should handle any Unicode without raising
        ids = self.tok.encode("日本語テスト")
        self.assertIsInstance(ids, list)

    def test_determinism(self):
        tok2 = ByteLevelBPETokenizer()
        tok2.train(TINY_CORPUS, vocab_size=300)
        self.assertEqual(self.tok._merges, tok2._merges)
        self.assertEqual(self.tok._vocab, tok2._vocab)

    def test_training_result_structure(self):
        self.assertIsInstance(self.result, TrainingResult)
        self.assertIsInstance(self.result.merge_history, list)
        self.assertIsInstance(self.result.metrics, dict)
        for key in ("base_vocab_size", "final_vocab_size", "merge_count",
                    "compression_ratio", "training_time_s"):
            self.assertIn(key, self.result.metrics)

    def test_merge_history_steps(self):
        for step in self.result.merge_history:
            self.assertIsInstance(step, MergeStep)
            self.assertIsInstance(step.pair, tuple)
            self.assertEqual(len(step.pair), 2)

    def test_compression_ratio_gte_1(self):
        self.assertGreaterEqual(self.result.metrics["compression_ratio"], 1.0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            self.tok.save(model_dir)
            self.assertTrue((model_dir / "merges.json").is_file())
            self.assertTrue((model_dir / "vocabulary.json").is_file())

            loaded = ByteLevelBPETokenizer.load(model_dir)
            self.assertEqual(loaded._merges, self.tok._merges)
            self.assertEqual(loaded._vocab, self.tok._vocab)
            self.assertEqual(loaded.encode("the cat"), self.tok.encode("the cat"))

    def test_save_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            self.tok.save(model_dir)
            merges = json.loads((model_dir / "merges.json").read_text(encoding="utf-8"))
            vocab = json.loads((model_dir / "vocabulary.json").read_text(encoding="utf-8"))
            self.assertEqual(merges["algorithm"], "bpe-byte-v1")
            self.assertIsInstance(merges["merges"], list)
            self.assertIsInstance(vocab["tokens"], list)

    def test_progress_fn_called(self):
        calls = []
        tok = ByteLevelBPETokenizer()
        tok.train(TINY_CORPUS, vocab_size=270,
                  progress_fn=lambda f, m: calls.append(f))
        self.assertGreater(len(calls), 0)
        self.assertLessEqual(max(calls), 1.0)

    def test_vocab_size_too_small_raises(self):
        tok = ByteLevelBPETokenizer()
        with self.assertRaises(ValueError):
            tok.train(TINY_CORPUS, vocab_size=10)

    def test_abstract_base_not_instantiable(self):
        with self.assertRaises(TypeError):
            Tokenizer()


class TestTokenizerAbstraction(unittest.TestCase):
    def test_tokenizer_is_abstract(self):
        import inspect
        self.assertTrue(inspect.isabstract(Tokenizer))

    def test_byte_level_bpe_is_concrete(self):
        import inspect
        self.assertFalse(inspect.isabstract(ByteLevelBPETokenizer))

    def test_algorithm_attribute(self):
        self.assertEqual(ByteLevelBPETokenizer.algorithm, "bpe-byte-v1")


if __name__ == "__main__":
    unittest.main()
