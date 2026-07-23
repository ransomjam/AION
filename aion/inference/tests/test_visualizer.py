import unittest
import numpy as np
from aion.inference.result import GenerationMetrics, GenerationResult, TokenStep
from aion.inference.visualizer import GenerationVisualizer


def _make_result(n_steps=4, vocab_size=8) -> GenerationResult:
    """Build a minimal GenerationResult for testing."""
    rng = np.random.default_rng(0)
    steps = []
    for i in range(n_steps):
        probs = rng.dirichlet(np.ones(vocab_size))
        top_idx = np.argsort(probs)[::-1][:5]
        candidates = [
            {"token_id": int(j), "token_str": chr(65 + int(j)), "prob": round(float(probs[j]), 8)}
            for j in top_idx
        ]
        steps.append(TokenStep(
            step=i,
            token_id=int(top_idx[0]),
            token_str=chr(65 + int(top_idx[0])),
            prob=round(float(probs[top_idx[0]]), 8),
            entropy=round(float(-np.sum(probs * np.log(probs + 1e-9))), 6),
            top_candidates=candidates,
            elapsed_ms=round(float(rng.uniform(1, 10)), 3),
        ))
    metrics = GenerationMetrics(
        prompt_tokens=3,
        generated_tokens=n_steps,
        total_tokens=3 + n_steps,
        generation_time_s=0.1,
        tokens_per_sec=float(n_steps) / 0.1,
        mean_entropy=float(np.mean([s.entropy for s in steps])),
        mean_top1_prob=float(np.mean([s.prob for s in steps])),
        stopped_by="max_new_tokens",
    )
    return GenerationResult(
        prompt="abc",
        generated_text="ABCD",
        full_text="abcABCD",
        generated_ids=[s.token_id for s in steps],
        steps=steps,
        metrics=metrics,
        config={},
    )


class TestGenerationVisualizer(unittest.TestCase):

    def setUp(self):
        self.viz = GenerationVisualizer()
        self.result = _make_result()

    def test_token_probabilities_length(self):
        out = self.viz.token_probabilities(self.result)
        self.assertEqual(len(out), len(self.result.steps))

    def test_token_probabilities_fields(self):
        out = self.viz.token_probabilities(self.result)
        for item in out:
            self.assertIn("step", item)
            self.assertIn("token_id", item)
            self.assertIn("token_str", item)
            self.assertIn("prob", item)
            self.assertIn("entropy", item)

    def test_token_probabilities_no_numpy(self):
        out = self.viz.token_probabilities(self.result)
        for item in out:
            for v in item.values():
                self.assertNotIsInstance(v, np.generic)

    def test_entropy_curve_length(self):
        curve = self.viz.entropy_curve(self.result)
        self.assertEqual(len(curve), len(self.result.steps))

    def test_entropy_curve_positive(self):
        curve = self.viz.entropy_curve(self.result)
        self.assertTrue(all(e >= 0 for e in curve))

    def test_top_k_distribution_length(self):
        out = self.viz.top_k_distribution(self.result, step=0, k=3)
        self.assertLessEqual(len(out), 3)

    def test_top_k_distribution_sorted(self):
        out = self.viz.top_k_distribution(self.result, step=0)
        probs = [c["prob"] for c in out]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top_k_distribution_bad_step(self):
        with self.assertRaises(IndexError):
            self.viz.top_k_distribution(self.result, step=999)

    def test_top_p_distribution_cumulative(self):
        out = self.viz.top_p_distribution(self.result, step=0, p=0.9)
        self.assertIn("cumulative", out[-1])
        self.assertGreaterEqual(out[-1]["cumulative"], 0.0)

    def test_top_p_distribution_bad_step(self):
        with self.assertRaises(IndexError):
            self.viz.top_p_distribution(self.result, step=999)

    def test_generation_timeline_length(self):
        out = self.viz.generation_timeline(self.result)
        self.assertEqual(len(out), len(self.result.steps))

    def test_generation_timeline_fields(self):
        out = self.viz.generation_timeline(self.result)
        for item in out:
            self.assertIn("step", item)
            self.assertIn("elapsed_ms", item)
            self.assertIn("token_str", item)

    def test_temperature_effect_default_temps(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = self.viz.temperature_effect(logits)
        self.assertEqual(len(out), 5)  # default 5 temperatures

    def test_temperature_effect_fields(self):
        logits = np.array([1.0, 2.0, 3.0])
        out = self.viz.temperature_effect(logits, temperatures=[0.5, 1.0, 2.0])
        for item in out:
            self.assertIn("temperature", item)
            self.assertIn("entropy", item)
            self.assertIn("top1_prob", item)
            self.assertIn("top1_token_id", item)

    def test_temperature_effect_low_temp_high_top1(self):
        logits = np.array([0.0, 0.0, 10.0])
        out = self.viz.temperature_effect(logits, temperatures=[0.1, 2.0])
        # Low temperature → higher top1 probability
        self.assertGreater(out[0]["top1_prob"], out[1]["top1_prob"])

    def test_temperature_effect_no_numpy(self):
        logits = np.array([1.0, 2.0, 3.0])
        out = self.viz.temperature_effect(logits, temperatures=[1.0])
        for item in out:
            for v in item.values():
                self.assertNotIsInstance(v, np.generic)

    def test_summary_fields(self):
        out = self.viz.summary(self.result)
        for key in ("prompt_tokens", "generated_tokens", "tokens_per_sec",
                    "mean_entropy", "mean_top1_prob", "stopped_by", "generated_text"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()
