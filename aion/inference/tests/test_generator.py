"""Tests for Generator.

Uses a minimal stub model and tokenizer so there are no external dependencies
and tests run fast.  The stub model returns fixed logits that make the
generated sequence predictable.
"""

import unittest
import numpy as np

from aion.inference.config import GenerationConfig
from aion.inference.generator import Generator
from aion.inference.processors import LogitsProcessorList, TemperatureProcessor
from aion.inference.result import GenerationResult
from aion.inference.sampler import GreedySampler
from aion.inference.streaming import CollectingCallback


# ── Stubs ─────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 16
EOS_ID = 0


class _StubTokenizer:
    """Minimal tokenizer: encodes each char as its ord % VOCAB_SIZE."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text] or [1]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 65) for i in ids)  # A=0, B=1, ...


class _FixedLogitsModel:
    """Model that always returns logits favouring token ``next_id``."""

    def __init__(self, next_id: int, vocab_size: int = VOCAB_SIZE) -> None:
        self.next_id = next_id
        self.vocab_size = vocab_size
        self.cfg = type("cfg", (), {"max_seq_len": 64})()

    def eval(self):
        pass

    def __call__(self, token_ids):
        batch, seq = token_ids.shape
        logits_data = np.zeros((batch, seq, self.vocab_size))
        logits_data[:, :, self.next_id] = 10.0  # strongly favour next_id

        class _FakeTensor:
            def __init__(self, data):
                self.data = data

        return _FakeTensor(logits_data), []


class _EosAfterNModel:
    """Model that returns EOS after N tokens, then a regular token."""

    def __init__(self, n: int, eos_id: int = EOS_ID, vocab_size: int = VOCAB_SIZE):
        self.n = n
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.cfg = type("cfg", (), {"max_seq_len": 64})()
        self._calls = 0

    def eval(self):
        pass

    def __call__(self, token_ids):
        self._calls += 1
        batch, seq = token_ids.shape
        logits_data = np.zeros((batch, seq, self.vocab_size))
        if self._calls > self.n:
            logits_data[:, :, self.eos_id] = 10.0
        else:
            logits_data[:, :, 1] = 10.0
        class _FakeTensor:
            def __init__(self, data):
                self.data = data
        return _FakeTensor(logits_data), []


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGeneratorBasic(unittest.TestCase):

    def _gen(self, next_id=2):
        model = _FixedLogitsModel(next_id)
        tok = _StubTokenizer()
        return Generator(model, tok, eos_token_id=EOS_ID)

    def test_returns_generation_result(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=5))
        self.assertIsInstance(result, GenerationResult)

    def test_generated_ids_length(self):
        g = self._gen(next_id=2)
        result = g.generate("hi", GenerationConfig(max_new_tokens=5))
        self.assertEqual(len(result.generated_ids), 5)

    def test_greedy_always_picks_same_token(self):
        g = self._gen(next_id=3)
        result = g.generate("hi", GenerationConfig(max_new_tokens=4, strategy="greedy"))
        self.assertTrue(all(tid == 3 for tid in result.generated_ids))

    def test_steps_count_matches_generated(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=6))
        self.assertEqual(len(result.steps), len(result.generated_ids))

    def test_step_indices_sequential(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=4))
        for i, step in enumerate(result.steps):
            self.assertEqual(step.step, i)

    def test_metrics_prompt_tokens(self):
        g = self._gen()
        result = g.generate("abc", GenerationConfig(max_new_tokens=3))
        self.assertEqual(result.metrics.prompt_tokens, 3)

    def test_metrics_generated_tokens(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=7))
        self.assertEqual(result.metrics.generated_tokens, 7)

    def test_metrics_tokens_per_sec_positive(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=5))
        self.assertGreater(result.metrics.tokens_per_sec, 0.0)

    def test_top_candidates_count(self):
        g = self._gen()
        cfg = GenerationConfig(max_new_tokens=3, top_candidates=5)
        result = g.generate("hi", cfg)
        for step in result.steps:
            self.assertLessEqual(len(step.top_candidates), 5)

    def test_top_candidates_sorted_by_prob(self):
        g = self._gen()
        result = g.generate("hi", GenerationConfig(max_new_tokens=2, top_candidates=4))
        for step in result.steps:
            probs = [c["prob"] for c in step.top_candidates]
            self.assertEqual(probs, sorted(probs, reverse=True))

    def test_prompt_string_stored(self):
        g = self._gen()
        result = g.generate("hello", GenerationConfig(max_new_tokens=2))
        self.assertEqual(result.prompt, "hello")

    def test_array_prompt_accepted(self):
        g = self._gen(next_id=2)
        ids = np.array([1, 2, 3], dtype=np.int32)
        result = g.generate(ids, GenerationConfig(max_new_tokens=3))
        self.assertEqual(len(result.generated_ids), 3)

    def test_config_stored_in_result(self):
        g = self._gen()
        cfg = GenerationConfig(max_new_tokens=3, strategy="greedy")
        result = g.generate("hi", cfg)
        self.assertEqual(result.config["strategy"], "greedy")


class TestGeneratorEosStopping(unittest.TestCase):

    def test_stops_on_eos(self):
        model = _EosAfterNModel(n=3, eos_id=EOS_ID)
        tok = _StubTokenizer()
        g = Generator(model, tok, eos_token_id=EOS_ID)
        cfg = GenerationConfig(max_new_tokens=20, stop_on_eos=True)
        result = g.generate("hi", cfg)
        self.assertEqual(result.metrics.stopped_by, "eos")
        self.assertLessEqual(result.metrics.generated_tokens, 20)

    def test_eos_disabled_continues(self):
        model = _EosAfterNModel(n=2, eos_id=EOS_ID)
        tok = _StubTokenizer()
        g = Generator(model, tok, eos_token_id=EOS_ID)
        cfg = GenerationConfig(max_new_tokens=5, stop_on_eos=False)
        result = g.generate("hi", cfg)
        self.assertEqual(result.metrics.generated_tokens, 5)
        self.assertEqual(result.metrics.stopped_by, "max_new_tokens")


class TestGeneratorMaxTokens(unittest.TestCase):

    def test_stops_at_max_new_tokens(self):
        model = _FixedLogitsModel(next_id=2)
        tok = _StubTokenizer()
        g = Generator(model, tok)
        result = g.generate("hi", GenerationConfig(max_new_tokens=10))
        self.assertEqual(result.metrics.generated_tokens, 10)
        self.assertEqual(result.metrics.stopped_by, "max_new_tokens")


class TestGeneratorStreaming(unittest.TestCase):

    def test_callback_receives_all_tokens(self):
        model = _FixedLogitsModel(next_id=2)
        tok = _StubTokenizer()
        g = Generator(model, tok)
        cb = CollectingCallback()
        result = g.generate("hi", GenerationConfig(max_new_tokens=5), callback=cb)
        self.assertEqual(len(cb.token_ids), 5)
        self.assertEqual(cb.token_ids, result.generated_ids)

    def test_on_complete_called(self):
        model = _FixedLogitsModel(next_id=2)
        tok = _StubTokenizer()
        g = Generator(model, tok)
        cb = CollectingCallback()
        result = g.generate("hi", GenerationConfig(max_new_tokens=3), callback=cb)
        self.assertIsNotNone(cb.result)
        self.assertIs(cb.result, result)

    def test_collecting_callback_text(self):
        model = _FixedLogitsModel(next_id=2)  # always token 2 → "C"
        tok = _StubTokenizer()
        g = Generator(model, tok)
        cb = CollectingCallback()
        g.generate("hi", GenerationConfig(max_new_tokens=3), callback=cb)
        self.assertEqual(cb.text, "CCC")


class TestGeneratorInjectedPipeline(unittest.TestCase):

    def test_injected_sampler_used(self):
        model = _FixedLogitsModel(next_id=5)
        tok = _StubTokenizer()
        # Inject greedy sampler explicitly
        g = Generator(model, tok, sampler=GreedySampler())
        result = g.generate("hi", GenerationConfig(max_new_tokens=3))
        self.assertTrue(all(tid == 5 for tid in result.generated_ids))

    def test_injected_processor_list_used(self):
        model = _FixedLogitsModel(next_id=5)
        tok = _StubTokenizer()
        pl = LogitsProcessorList([TemperatureProcessor(0.01)])  # very sharp
        g = Generator(model, tok, processors=pl, sampler=GreedySampler())
        result = g.generate("hi", GenerationConfig(max_new_tokens=2))
        self.assertEqual(len(result.generated_ids), 2)


class TestGeneratorContextTruncation(unittest.TestCase):

    def test_prompt_truncated_to_budget(self):
        model = _FixedLogitsModel(next_id=2)
        tok = _StubTokenizer()
        g = Generator(model, tok)
        # max_context_len=6, max_new_tokens=4 → budget=2 → prompt truncated to 2
        long_prompt = "abcdefgh"  # 8 tokens
        cfg = GenerationConfig(max_new_tokens=4, max_context_len=6)
        result = g.generate(long_prompt, cfg)
        self.assertEqual(result.metrics.prompt_tokens, 2)


if __name__ == "__main__":
    unittest.main()
