import unittest
import numpy as np
from aion.inference.processors import (
    LogitsProcessorList,
    RepetitionPenaltyProcessor,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
)

_NEG_INF = -1e9


class TestTemperatureProcessor(unittest.TestCase):

    def test_divides_by_temperature(self):
        proc = TemperatureProcessor(2.0)
        logits = np.array([2.0, 4.0, 6.0])
        result = proc(logits, [])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_temperature_one_is_identity(self):
        proc = TemperatureProcessor(1.0)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, []), logits)

    def test_does_not_mutate_input(self):
        proc = TemperatureProcessor(0.5)
        logits = np.array([1.0, 2.0, 3.0])
        original = logits.copy()
        proc(logits, [])
        np.testing.assert_array_equal(logits, original)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            TemperatureProcessor(0.0)


class TestTopKProcessor(unittest.TestCase):

    def test_keeps_top_k(self):
        proc = TopKProcessor(2)
        logits = np.array([1.0, 3.0, 2.0, 0.5])
        result = proc(logits, [])
        # Top-2 are indices 1 (3.0) and 2 (2.0); others → -inf
        self.assertGreater(result[1], _NEG_INF)
        self.assertGreater(result[2], _NEG_INF)
        self.assertAlmostEqual(result[0], _NEG_INF)
        self.assertAlmostEqual(result[3], _NEG_INF)

    def test_k_zero_is_identity(self):
        proc = TopKProcessor(0)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, []), logits)

    def test_k_larger_than_vocab_is_identity(self):
        proc = TopKProcessor(100)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, []), logits)

    def test_does_not_mutate_input(self):
        proc = TopKProcessor(2)
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        original = logits.copy()
        proc(logits, [])
        np.testing.assert_array_equal(logits, original)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            TopKProcessor(-1)


class TestTopPProcessor(unittest.TestCase):

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def test_p_one_is_identity(self):
        proc = TopPProcessor(1.0)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, []), logits)

    def test_nucleus_contains_enough_mass(self):
        proc = TopPProcessor(0.9)
        # Peaked distribution: token 3 has most mass
        logits = np.array([0.1, 0.2, 0.3, 10.0])
        result = proc(logits, [])
        # Token 3 must be in nucleus
        self.assertGreater(result[3], _NEG_INF)

    def test_low_prob_tokens_filtered(self):
        proc = TopPProcessor(0.5)
        # Very peaked: token 0 has almost all mass
        logits = np.array([100.0, 0.0, 0.0, 0.0])
        result = proc(logits, [])
        # Tokens 1,2,3 should be filtered
        self.assertAlmostEqual(result[1], _NEG_INF)
        self.assertAlmostEqual(result[2], _NEG_INF)
        self.assertAlmostEqual(result[3], _NEG_INF)

    def test_does_not_mutate_input(self):
        proc = TopPProcessor(0.9)
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        original = logits.copy()
        proc(logits, [])
        np.testing.assert_array_equal(logits, original)

    def test_invalid_p(self):
        with self.assertRaises(ValueError):
            TopPProcessor(0.0)


class TestRepetitionPenaltyProcessor(unittest.TestCase):

    def test_no_penalty_when_one(self):
        proc = RepetitionPenaltyProcessor(1.0)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, [0, 1]), logits)

    def test_positive_logit_divided(self):
        proc = RepetitionPenaltyProcessor(2.0)
        logits = np.array([4.0, 2.0, 1.0])
        result = proc(logits, [0])
        self.assertAlmostEqual(result[0], 2.0)   # 4.0 / 2.0
        self.assertAlmostEqual(result[1], 2.0)   # unchanged
        self.assertAlmostEqual(result[2], 1.0)   # unchanged

    def test_negative_logit_multiplied(self):
        proc = RepetitionPenaltyProcessor(2.0)
        logits = np.array([-4.0, 2.0, 1.0])
        result = proc(logits, [0])
        self.assertAlmostEqual(result[0], -8.0)  # -4.0 * 2.0

    def test_empty_generated_ids_is_identity(self):
        proc = RepetitionPenaltyProcessor(2.0)
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(proc(logits, []), logits)

    def test_does_not_mutate_input(self):
        proc = RepetitionPenaltyProcessor(1.5)
        logits = np.array([1.0, 2.0, 3.0])
        original = logits.copy()
        proc(logits, [0])
        np.testing.assert_array_equal(logits, original)

    def test_invalid_penalty(self):
        with self.assertRaises(ValueError):
            RepetitionPenaltyProcessor(0.0)


class TestLogitsProcessorList(unittest.TestCase):

    def test_empty_list_is_identity(self):
        pl = LogitsProcessorList()
        logits = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pl(logits, []), logits)

    def test_sequential_application(self):
        # Temperature 2.0 then top-k 2
        pl = LogitsProcessorList([
            TemperatureProcessor(2.0),
            TopKProcessor(2),
        ])
        logits = np.array([2.0, 4.0, 6.0, 8.0])
        result = pl(logits, [])
        # After temperature: [1, 2, 3, 4]; top-2 keeps indices 2 and 3
        self.assertGreater(result[3], -1e8)
        self.assertGreater(result[2], -1e8)
        self.assertAlmostEqual(result[0], -1e9)
        self.assertAlmostEqual(result[1], -1e9)

    def test_len(self):
        pl = LogitsProcessorList([TemperatureProcessor(1.0), TopKProcessor(5)])
        self.assertEqual(len(pl), 2)

    def test_append(self):
        pl = LogitsProcessorList()
        pl.append(TemperatureProcessor(0.5))
        self.assertEqual(len(pl), 1)


if __name__ == "__main__":
    unittest.main()
