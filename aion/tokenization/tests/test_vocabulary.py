"""Tests for vocabulary construction, encoding/decoding, and persistence."""

import tempfile
import unittest
from pathlib import Path

from aion.tokenization.vocabulary import (
    BOS,
    DEFAULT_SPECIALS,
    EOS,
    PAD,
    UNK,
    Vocabulary,
)

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the the the",  # makes "the" clearly the most frequent token
]


class BuildTest(unittest.TestCase):
    def test_specials_occupy_the_first_ids_in_order(self):
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.id_to_token(0), PAD)
        self.assertEqual(vocab.id_to_token(1), UNK)
        self.assertEqual(vocab.id_to_token(2), BOS)
        self.assertEqual(vocab.id_to_token(3), EOS)

    def test_most_frequent_corpus_token_comes_first_after_specials(self):
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.id_to_token(len(DEFAULT_SPECIALS)), "the")

    def test_ordering_is_deterministic_across_builds(self):
        a = Vocabulary.build(CORPUS)
        b = Vocabulary.build(list(reversed(CORPUS)))  # same tokens, different doc order
        self.assertEqual(a.to_json(), b.to_json())

    def test_min_freq_drops_rare_tokens(self):
        vocab = Vocabulary.build(CORPUS, min_freq=2)
        self.assertIn("the", vocab)      # frequent -> kept
        self.assertIn("sat", vocab)      # appears twice -> kept
        self.assertNotIn("cat", vocab)   # appears once -> dropped
        self.assertNotIn("mat", vocab)

    def test_max_size_caps_total_including_specials(self):
        vocab = Vocabulary.build(CORPUS, max_size=6)
        self.assertEqual(len(vocab), 6)
        # The two survivors past the specials must be the most frequent tokens.
        self.assertEqual(vocab.id_to_token(4), "the")

    def test_max_size_smaller_than_specials_is_an_error(self):
        with self.assertRaises(ValueError):
            Vocabulary.build(CORPUS, max_size=2)

    def test_min_freq_must_be_positive(self):
        with self.assertRaises(ValueError):
            Vocabulary.build(CORPUS, min_freq=0)

    def test_frequencies_recorded_for_kept_tokens_only(self):
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.frequency("the"), 7)
        self.assertEqual(vocab.frequency("sat"), 2)
        self.assertEqual(vocab.frequency(PAD), 0)      # specials have no frequency
        self.assertEqual(vocab.frequency("absent"), 0)


class EncodeDecodeTest(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary.build(CORPUS)

    def test_round_trips_known_tokens(self):
        ids = self.vocab.encode("the cat sat")
        self.assertEqual(self.vocab.decode(ids), ["the", "cat", "sat"])

    def test_unknown_tokens_map_to_unk(self):
        ids = self.vocab.encode("the elephant")
        self.assertEqual(ids[1], self.vocab.token_to_id(UNK))
        # <unk> is a special, so decode skips it by default.
        self.assertEqual(self.vocab.decode(ids), ["the"])
        self.assertEqual(self.vocab.decode(ids, skip_specials=False), ["the", UNK])

    def test_bos_eos_wrapping(self):
        ids = self.vocab.encode("the cat", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], self.vocab.token_to_id(BOS))
        self.assertEqual(ids[-1], self.vocab.token_to_id(EOS))

    def test_encoding_shares_the_module_tokenizer(self):
        # Normalization + tokenization are applied, so case and accents fold.
        self.assertEqual(self.vocab.encode("THE cat"), self.vocab.encode("the cat"))

    def test_decode_rejects_out_of_range_id(self):
        with self.assertRaises(KeyError):
            self.vocab.decode([999999])

    def test_token_to_id_falls_back_to_unk(self):
        self.assertEqual(self.vocab.token_to_id("neverseen"),
                         self.vocab.token_to_id(UNK))


class PersistenceTest(unittest.TestCase):
    def test_json_round_trip_preserves_mapping(self):
        vocab = Vocabulary.build(CORPUS)
        restored = Vocabulary.from_json(vocab.to_json())
        self.assertEqual(vocab.to_json(), restored.to_json())
        self.assertEqual(vocab.encode("the dog sat"), restored.encode("the dog sat"))

    def test_save_and_load(self):
        vocab = Vocabulary.build(CORPUS)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vocab.json"
            vocab.save(path)
            restored = Vocabulary.load(path)
        self.assertEqual(vocab.to_json(), restored.to_json())

    def test_vocabulary_without_unk_is_rejected(self):
        with self.assertRaises(ValueError):
            Vocabulary(["a", "b"], {}, specials=())


if __name__ == "__main__":
    unittest.main()
