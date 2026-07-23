import unittest
import tempfile
from pathlib import Path
from aion.training.metrics import MetricsCollector
from aion.training.dashboard import TrainingDashboard


class TestTrainingDashboard(unittest.TestCase):

    def _collector(self, tmp):
        return MetricsCollector(Path(tmp) / "metrics.jsonl")

    def test_snapshot_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertIsInstance(snap, dict)

    def test_snapshot_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            for key in ("status", "epoch", "epochs", "global_step",
                        "train_loss", "val_loss", "perplexity",
                        "lr", "grad_norm", "tokens_per_sec",
                        "elapsed_s", "eta_s", "checkpoint_path",
                        "loss_curve", "lr_curve"):
                self.assertIn(key, snap)

    def test_epoch_zero_before_any_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertEqual(snap["epoch"], 0)

    def test_loss_curve_populated_after_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            c.record_epoch(0, {"mean_loss": 3.0, "val_loss": 3.2})
            c.record_epoch(1, {"mean_loss": 2.5, "val_loss": 2.8})
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertEqual(len(snap["loss_curve"]), 2)
            self.assertEqual(snap["loss_curve"][0]["train_loss"], 3.0)
            self.assertEqual(snap["loss_curve"][1]["val_loss"], 2.8)

    def test_lr_curve_populated_after_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            for i in range(10):
                c.record_step(i, 0, {"lr": 0.001 * (i + 1)})
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertGreater(len(snap["lr_curve"]), 0)

    def test_lr_curve_sampled_when_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            for i in range(1000):
                c.record_step(i, 0, {"lr": 0.001})
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertLessEqual(len(snap["lr_curve"]), 500)

    def test_eta_none_before_any_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot()
            self.assertIsNone(snap["eta_s"])

    def test_status_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=5)
            snap = dash.snapshot(status="complete")
            self.assertEqual(snap["status"], "complete")

    def test_epochs_field_matches_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._collector(tmp)
            dash = TrainingDashboard(c, epochs=10)
            snap = dash.snapshot()
            self.assertEqual(snap["epochs"], 10)


if __name__ == "__main__":
    unittest.main()
