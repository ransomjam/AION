"""Tests for transformer configuration objects."""

import unittest
from aion.transformer.config import DecoderConfig, EncoderConfig, TransformerConfig


class TestTransformerConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = TransformerConfig()
        self.assertEqual(cfg.d_model, 128)
        self.assertEqual(cfg.n_heads, 4)
        self.assertEqual(cfg.activation, "relu")
        self.assertTrue(cfg.pre_norm)

    def test_validate_passes(self):
        TransformerConfig(d_model=64, n_heads=4).validate()

    def test_validate_bad_divisibility(self):
        with self.assertRaises(ValueError):
            TransformerConfig(d_model=10, n_heads=3).validate()

    def test_validate_bad_activation(self):
        with self.assertRaises(ValueError):
            TransformerConfig(activation="swish").validate()

    def test_validate_bad_dropout(self):
        with self.assertRaises(ValueError):
            TransformerConfig(dropout=1.0).validate()

    def test_round_trip(self):
        cfg = TransformerConfig(d_model=64, n_heads=4, n_encoder_layers=3,
                                activation="gelu", dropout=0.1)
        cfg2 = TransformerConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)

    def test_from_dict_ignores_unknown_keys(self):
        d = TransformerConfig().to_dict()
        d["future_field"] = 42
        cfg = TransformerConfig.from_dict(d)
        self.assertIsInstance(cfg, TransformerConfig)


class TestEncoderConfig(unittest.TestCase):

    def test_to_transformer_config(self):
        cfg = EncoderConfig(d_model=64, n_heads=4, n_layers=3)
        tc = cfg.to_transformer_config()
        self.assertEqual(tc.n_encoder_layers, 3)
        self.assertEqual(tc.n_decoder_layers, 0)

    def test_validate_bad_divisibility(self):
        with self.assertRaises(ValueError):
            EncoderConfig(d_model=10, n_heads=3).validate()

    def test_round_trip(self):
        cfg = EncoderConfig(d_model=32, n_heads=2, n_layers=4)
        cfg2 = EncoderConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)


class TestDecoderConfig(unittest.TestCase):

    def test_default_activation_gelu(self):
        cfg = DecoderConfig()
        self.assertEqual(cfg.activation, "gelu")

    def test_to_transformer_config(self):
        cfg = DecoderConfig(d_model=64, n_heads=4, n_layers=2)
        tc = cfg.to_transformer_config()
        self.assertEqual(tc.n_encoder_layers, 0)
        self.assertEqual(tc.n_decoder_layers, 2)

    def test_round_trip(self):
        cfg = DecoderConfig(d_model=32, n_heads=2, n_layers=2)
        cfg2 = DecoderConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)


if __name__ == "__main__":
    unittest.main()
