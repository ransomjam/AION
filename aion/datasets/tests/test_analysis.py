"""Tests for statistics, quality, and search over a dataset."""

import tempfile
import unittest
from pathlib import Path

from aion.datasets.quality import compute_quality
from aion.datasets.search import search_dataset
from aion.datasets.stats import compute_statistics
from aion.datasets.store import DatasetStore


def _dataset(tmp, docs):
    ds = DatasetStore(Path(tmp)).create("C")
    ds.add_documents(docs)
    return ds


class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts_and_ttr(self):
        ds = _dataset(self._tmp.name, ["the cat", "the dog", "the the the"])
        s = compute_statistics(ds)
        self.assertEqual(s["documents"], 3)
        self.assertEqual(s["tokens"], 7)            # 2 + 2 + 3
        self.assertEqual(s["unique_tokens"], 3)     # the, cat, dog
        self.assertAlmostEqual(s["type_token_ratio"], round(3 / 7, 4))
        self.assertEqual(s["top_tokens"][0], ["the", 5])

    def test_vocab_growth_is_monotonic(self):
        ds = _dataset(self._tmp.name, ["a b", "c d", "a e"])
        growth = [g["unique_tokens"] for g in compute_statistics(ds)["vocabulary_growth"]]
        self.assertEqual(growth, sorted(growth))
        self.assertEqual(growth[-1], 5)             # a b c d e

    def test_length_summary_and_histogram(self):
        ds = _dataset(self._tmp.name, ["one two three", "solo"])
        s = compute_statistics(ds)
        self.assertEqual(s["document_length"]["max_tokens"], 3)
        self.assertEqual(s["document_length"]["min_tokens"], 1)
        self.assertTrue(s["length_distribution"])


class QualityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_flags_empty_whitespace_and_duplicates(self):
        ds = _dataset(self._tmp.name, ["hello world", "hello world", "", "   "])
        q = compute_quality(ds)
        self.assertIn(3, q["issues"]["empty"])
        self.assertIn(4, q["issues"]["whitespace_only"])
        self.assertEqual(q["issues"]["duplicate_groups"][0]["count"], 2)

    def test_normalization_and_language_hint(self):
        ds = _dataset(self._tmp.name, ["the cat and the dog", "le chat et le chien"])
        q = compute_quality(ds)
        self.assertEqual(q["normalization"]["already_normalized"], 2)  # already lowercase ascii
        self.assertEqual(q["language_hint"].get("en", 0), 1)
        self.assertEqual(q["language_hint"].get("fr", 0), 1)


class SearchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_case_insensitive_substring(self):
        ds = _dataset(self._tmp.name, ["Confirm your PIN", "pin code"])
        r = search_dataset(ds, "pin")
        self.assertEqual(r["match_count"], 2)

    def test_case_sensitive(self):
        ds = _dataset(self._tmp.name, ["Confirm your PIN", "pin code"])
        r = search_dataset(ds, "PIN", case_sensitive=True)
        self.assertEqual(r["match_count"], 1)

    def test_regex(self):
        ds = _dataset(self._tmp.name, ["order 12345 shipped", "no digits here"])
        r = search_dataset(ds, r"\d+", regex=True)
        self.assertEqual(r["match_count"], 1)
        self.assertEqual(r["matches"][0]["doc_id"], 1)

    def test_empty_query(self):
        ds = _dataset(self._tmp.name, ["anything"])
        self.assertEqual(search_dataset(ds, "")["match_count"], 0)


if __name__ == "__main__":
    unittest.main()
