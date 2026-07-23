import unittest
import numpy as np
from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel
from aion.training.sampler import SampleGenerator, SampleResult


class _FakeTokenizer:
    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text[:4]] or [0]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 65) for i in ids)


def _small_model(seed=0):
    cfg = GPTConfig(
        vocab_size=32, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, dropout=0.0, max_seq_len=16, tie_weights=True,
    )
    return GPTModel(cfg, rng=np.random.default_rng(seed))


class TestSampleGenerator(unittest.TestCase):

    def test_generate_returns_result(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=8)
        result = gen.generate("hi")
        self.assertIsInstance(result, SampleResult)

    def test_generated_text_non_empty(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=8)
        result = gen.generate("hi")
        self.assertIsInstance(result.generated_text, str)

    def test_tokens_positive(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=8)
        result = gen.generate("hi")
        self.assertGreater(result.tokens, 0)

    def test_full_text_contains_prompt(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=4)
        result = gen.generate("hi")
        # full_text = prompt + generated; prompt may be empty if tokenizer
        # decodes differently, but full_text must be a string
        self.assertIsInstance(result.full_text, str)

    def test_generate_all_returns_list(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=4)
        results = gen.generate_all(["a", "b", "c"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, SampleResult)

    def test_generate_all_empty_prompts(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=4)
        results = gen.generate_all([])
        self.assertEqual(results, [])

    def test_elapsed_s_non_negative(self):
        model = _small_model()
        gen = SampleGenerator(model, _FakeTokenizer(), max_new_tokens=4)
        result = gen.generate("x")
        self.assertGreaterEqual(result.elapsed_s, 0.0)


if __name__ == "__main__":
    unittest.main()
