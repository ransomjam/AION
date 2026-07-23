import unittest
from aion.inference.config import GenerationConfig


class TestGenerationConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = GenerationConfig()
        self.assertEqual(cfg.strategy, "greedy")
        self.assertEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_k, 0)
        self.assertEqual(cfg.top_p, 1.0)
        self.assertEqual(cfg.repetition_penalty, 1.0)
        self.assertEqual(cfg.max_new_tokens, 128)
        self.assertEqual(cfg.top_candidates, 20)
        self.assertTrue(cfg.stop_on_eos)

    def test_to_dict_round_trip(self):
        cfg = GenerationConfig(strategy="top_p", temperature=0.8, top_p=0.9,
                               max_new_tokens=64, top_candidates=10)
        cfg2 = GenerationConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)

    def test_from_dict_ignores_unknown(self):
        d = GenerationConfig().to_dict()
        d["future_field"] = "ignored"
        cfg = GenerationConfig.from_dict(d)
        self.assertIsInstance(cfg, GenerationConfig)

    def test_validate_passes(self):
        GenerationConfig(strategy="temperature", temperature=0.7).validate()

    def test_validate_bad_strategy(self):
        with self.assertRaises(ValueError):
            GenerationConfig(strategy="beam_search").validate()

    def test_validate_bad_temperature(self):
        with self.assertRaises(ValueError):
            GenerationConfig(temperature=0.0).validate()

    def test_validate_bad_top_k(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_k=-1).validate()

    def test_validate_bad_top_p(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=0.0).validate()

    def test_validate_bad_repetition_penalty(self):
        with self.assertRaises(ValueError):
            GenerationConfig(repetition_penalty=0.0).validate()

    def test_validate_bad_max_new_tokens(self):
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=0).validate()

    def test_validate_bad_top_candidates(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_candidates=0).validate()


if __name__ == "__main__":
    unittest.main()
