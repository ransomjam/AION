"""Tests for TransformerStore."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from aion.nn.trainer import TrainingResult
from aion.transformer.config import DecoderConfig, EncoderConfig, TransformerConfig
from aion.transformer.encoder import TransformerEncoder
from aion.transformer.stack import TransformerStack
from aion.transformer.model import Transformer
from aion.transformer.store import TransformerNotFound, TransformerStore


def _result(**kw):
    metrics = {"final_loss": 0.5, "epochs": 1, "param_count": 0,
               "training_time_s": 0.1, "seed": 0, "loss_history": [0.5]}
    metrics.update(kw)
    return TrainingResult(metrics=metrics)


def _rng():
    return np.random.default_rng(0)


class TestTransformerStoreEncoder(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TransformerStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self):
        cfg = EncoderConfig(d_model=16, n_heads=4, n_layers=2, d_ff=64)
        return TransformerEncoder.from_config(cfg, rng=_rng()), cfg

    def test_save_returns_manifest(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="test-enc",
                            architecture="encoder-v1")
        self.assertEqual(m["architecture"], "encoder-v1")
        self.assertIn("id", m)
        self.assertIn("model_fingerprint", m)

    def test_manifest_fields(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="my encoder",
                            architecture="encoder-v1",
                            description="test", tokenizer_id="tok-123")
        self.assertEqual(m["name"], "my encoder")
        self.assertEqual(m["description"], "test")
        self.assertEqual(m["tokenizer_id"], "tok-123")
        self.assertEqual(m["status"], "experimental")
        self.assertIn("created_at", m)
        self.assertIn("produced_by", m)

    def test_exists(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="enc",
                            architecture="encoder-v1")
        self.assertTrue(self.store.exists(m["id"]))
        self.assertFalse(self.store.exists("nonexistent"))

    def test_open_manifest(self):
        model, _ = self._model()
        saved = self.store.save(model, _result(), name="enc",
                                architecture="encoder-v1")
        loaded = self.store.open_manifest(saved["id"])
        self.assertEqual(loaded["id"], saved["id"])

    def test_open_manifest_not_found(self):
        with self.assertRaises(TransformerNotFound):
            self.store.open_manifest("does-not-exist")

    def test_load_weight_fidelity(self):
        model, _ = self._model()
        original_weights = [p.data.copy() for p in model.parameters()]
        m = self.store.save(model, _result(), name="enc",
                            architecture="encoder-v1")
        loaded = self.store.load(m["id"])
        for orig, p in zip(original_weights, loaded.parameters()):
            np.testing.assert_array_equal(orig, p.data)

    def test_load_is_transformer_encoder(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="enc",
                            architecture="encoder-v1")
        loaded = self.store.load(m["id"])
        self.assertIsInstance(loaded, TransformerEncoder)

    def test_statistics(self):
        model, _ = self._model()
        m = self.store.save(model, _result(final_loss=0.42),
                            name="enc", architecture="encoder-v1")
        stats = self.store.statistics(m["id"])
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["final_loss"], 0.42)

    def test_list(self):
        model, _ = self._model()
        self.store.save(model, _result(), name="enc-a", architecture="encoder-v1")
        self.store.save(model, _result(), name="enc-b", architecture="encoder-v1")
        manifests = self.store.list()
        self.assertEqual(len(manifests), 2)

    def test_list_empty(self):
        self.assertEqual(self.store.list(), [])

    def test_delete(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="enc",
                            architecture="encoder-v1")
        self.store.delete(m["id"])
        self.assertFalse(self.store.exists(m["id"]))
        self.assertEqual(self.store.list(), [])

    def test_set_status(self):
        model, _ = self._model()
        m = self.store.save(model, _result(), name="enc",
                            architecture="encoder-v1")
        updated = self.store.set_status(m["id"], "default")
        self.assertEqual(updated["status"], "default")
        self.assertEqual(self.store.open_manifest(m["id"])["status"], "default")

    def test_unknown_architecture_raises(self):
        model, _ = self._model()
        with self.assertRaises(ValueError):
            self.store.save(model, _result(), name="enc",
                            architecture="unknown-v99")


class TestTransformerStoreDecoder(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TransformerStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self):
        cfg = DecoderConfig(d_model=16, n_heads=4, n_layers=2, d_ff=64)
        return TransformerStack(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            d_ff=cfg.d_ff, dropout=cfg.dropout, activation=cfg.activation,
            pre_norm=cfg.pre_norm, rng=_rng(),
        ), cfg

    def test_save_load_decoder(self):
        model, _ = self._model()
        original_weights = [p.data.copy() for p in model.parameters()]
        m = self.store.save(model, _result(), name="dec",
                            architecture="decoder-v1")
        loaded = self.store.load(m["id"])
        self.assertIsInstance(loaded, TransformerStack)
        for orig, p in zip(original_weights, loaded.parameters()):
            np.testing.assert_array_equal(orig, p.data)


class TestTransformerStoreFullModel(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TransformerStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self):
        cfg = TransformerConfig(d_model=16, n_heads=4, n_encoder_layers=2,
                                n_decoder_layers=2, d_ff=64)
        return Transformer(cfg, rng=_rng()), cfg

    def test_save_load_transformer(self):
        model, _ = self._model()
        original_weights = [p.data.copy() for p in model.parameters()]
        m = self.store.save(model, _result(), name="full",
                            architecture="transformer-v1")
        loaded = self.store.load(m["id"])
        self.assertIsInstance(loaded, Transformer)
        for orig, p in zip(original_weights, loaded.parameters()):
            np.testing.assert_array_equal(orig, p.data)

    def test_weights_tied_recorded_in_manifest(self):
        from aion.nn.parameter import Parameter
        model, _ = self._model()
        emb = Parameter(np.zeros((100, 16)), name="table")
        model.tie_weights(emb, emb)
        m = self.store.save(model, _result(), name="tied",
                            architecture="transformer-v1")
        self.assertTrue(m["weights_tied"])


if __name__ == "__main__":
    unittest.main()
