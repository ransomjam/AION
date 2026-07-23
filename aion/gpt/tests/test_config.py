import unittest
from aion.gpt.config import GPTConfig
from aion.transformer.config import DecoderConfig


class TestGPTConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = GPTConfig()
        self.assertEqual(cfg.vocab_size, 256)
        self.assertEqual(cfg.activation, "gelu")
        self.assertTrue(cfg.tie_weights)

    def test_to_dict_round_trip(self):
        cfg = GPTConfig(vocab_size=512, d_model=64, n_heads=4, n_layers=2, d_ff=256)
        cfg2 = GPTConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg, cfg2)

    def test_from_dict_ignores_unknown_keys(self):
        d = GPTConfig().to_dict()
        d["unknown_field"] = "ignored"
        cfg = GPTConfig.from_dict(d)
        self.assertIsInstance(cfg, GPTConfig)

    def test_validate_passes(self):
        GPTConfig(vocab_size=100, d_model=64, n_heads=4).validate()

    def test_validate_bad_d_model(self):
        with self.assertRaises(ValueError):
            GPTConfig(d_model=0).validate()

    def test_validate_bad_heads(self):
        with self.assertRaises(ValueError):
            GPTConfig(d_model=64, n_heads=3).validate()

    def test_validate_bad_activation(self):
        with self.assertRaises(ValueError):
            GPTConfig(activation="swish").validate()

    def test_validate_bad_dropout(self):
        with self.assertRaises(ValueError):
            GPTConfig(dropout=1.0).validate()

    def test_validate_bad_vocab(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=0).validate()

    def test_to_decoder_config(self):
        cfg = GPTConfig(d_model=64, n_heads=4, n_layers=3, d_ff=256,
                        dropout=0.1, activation="gelu", pre_norm=True, max_seq_len=128)
        dc = cfg.to_decoder_config()
        self.assertIsInstance(dc, DecoderConfig)
        self.assertEqual(dc.d_model, 64)
        self.assertEqual(dc.n_layers, 3)
        self.assertEqual(dc.activation, "gelu")
        self.assertEqual(dc.max_seq_len, 128)

    def test_to_decoder_config_does_not_include_vocab(self):
        cfg = GPTConfig(vocab_size=1000)
        dc = cfg.to_decoder_config()
        self.assertFalse(hasattr(dc, "vocab_size"))


if __name__ == "__main__":
    unittest.main()
