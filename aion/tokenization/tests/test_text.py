"""Tests for text normalization and tokenization.

Each test encodes a property claimed in the module or its README; if the property
breaks, the test fails.
"""

import unittest

from aion.tokenization.text import normalize, strip_accents, tokenize


class StripAccentsTest(unittest.TestCase):
    def test_removes_diacritics_but_keeps_base_letters(self):
        self.assertEqual(strip_accents("café"), "cafe")
        self.assertEqual(strip_accents("naïve résumé"), "naive resume")

    def test_handles_precomposed_and_decomposed_identically(self):
        precomposed = "é"                 # single codepoint U+00E9
        decomposed = "é"            # 'e' + combining acute U+0301
        self.assertEqual(strip_accents(precomposed), strip_accents(decomposed))
        self.assertEqual(strip_accents(precomposed), "e")

    def test_leaves_unaccented_text_untouched(self):
        self.assertEqual(strip_accents("hello"), "hello")


class NormalizeTest(unittest.TestCase):
    def test_lowercases_by_default(self):
        self.assertEqual(normalize("HELLO World"), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("  a\t b\n\nc  "), "a b c")

    def test_nfkc_folds_compatibility_variants(self):
        # Full-width digits and a ligature collapse to ASCII under NFKC.
        self.assertEqual(normalize("１２３"), "123")
        self.assertEqual(normalize("ﬁle"), "file")

    def test_casefold_is_more_aggressive_than_lower(self):
        # German sharp-s folds to "ss"; str.lower would not do this.
        self.assertEqual(normalize("STRASSE"), normalize("straße"))

    def test_accents_stripped_by_default_but_optional(self):
        self.assertEqual(normalize("Café"), "cafe")
        self.assertEqual(normalize("Café", accents=False), "café")

    def test_is_idempotent(self):
        once = normalize("Ｃafé  RÉSUMÉ\tﬁle")
        twice = normalize(once)
        self.assertEqual(once, twice)

    def test_rejects_unknown_form(self):
        with self.assertRaises(ValueError):
            normalize("x", form="NFXY")


class TokenizeTest(unittest.TestCase):
    def test_splits_words_on_whitespace(self):
        self.assertEqual(tokenize("pay now"), ["pay", "now"])

    def test_keeps_alphanumeric_runs_together(self):
        self.assertEqual(tokenize("momo123"), ["momo123"])

    def test_punctuation_becomes_individual_tokens(self):
        self.assertEqual(tokenize("hi!!"), ["hi", "!", "!"])
        self.assertEqual(tokenize("a, b."), ["a", ",", "b", "."])

    def test_normalizes_before_splitting_by_default(self):
        # café and cafe tokenize identically once normalized.
        self.assertEqual(tokenize("Café"), tokenize("cafe"))
        self.assertEqual(tokenize("Café"), ["cafe"])

    def test_can_skip_normalization(self):
        # With normalization off, the original case and accent are preserved.
        self.assertEqual(tokenize("Café", normalize_first=False), ["Café"])

    def test_unicode_letters_are_word_characters(self):
        # Non-Latin scripts count as alphanumeric, so they stay whole.
        self.assertEqual(tokenize("привет мир"), ["привет", "мир"])

    def test_empty_and_whitespace_input(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   \t\n"), [])


if __name__ == "__main__":
    unittest.main()
