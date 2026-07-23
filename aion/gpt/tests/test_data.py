import unittest
import numpy as np
from aion.gpt.data import BatchSampler, TokenizedDataset


class TestTokenizedDataset(unittest.TestCase):

    def _make_tokens(self, n=100):
        return np.arange(n, dtype=np.int32)

    def test_len(self):
        ds = TokenizedDataset(self._make_tokens(100), block_size=9)
        # 100 // 10 = 10 blocks
        self.assertEqual(len(ds), 10)

    def test_getitem_shapes(self):
        ds = TokenizedDataset(self._make_tokens(100), block_size=9)
        x, y = ds[0]
        self.assertEqual(x.shape, (9,))
        self.assertEqual(y.shape, (9,))

    def test_getitem_offset(self):
        tokens = np.arange(20, dtype=np.int32)
        ds = TokenizedDataset(tokens, block_size=9)
        x, y = ds[0]
        np.testing.assert_array_equal(x, tokens[:9])
        np.testing.assert_array_equal(y, tokens[1:10])

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            TokenizedDataset(np.arange(5, dtype=np.int32), block_size=9)

    def test_pack(self):
        docs = [[1, 2, 3], [4, 5, 6]]
        ds = TokenizedDataset.pack(docs, block_size=3, eos_id=0)
        # flat = [1,2,3,0,4,5,6,0] → 8 tokens → 8//4 = 2 blocks
        self.assertEqual(len(ds), 2)

    def test_pack_eos_present(self):
        docs = [[1, 2, 3], [4, 5, 6]]
        ds = TokenizedDataset.pack(docs, block_size=3, eos_id=0)
        x, y = ds[0]
        # block 0 = [1,2,3,0]; x=[1,2,3], y=[2,3,0]
        self.assertEqual(int(y[-1]), 0)  # EOS appears as target


class TestBatchSampler(unittest.TestCase):

    def _make_ds(self, n=100, block_size=9):
        return TokenizedDataset(np.arange(n, dtype=np.int32), block_size)

    def test_batch_shapes(self):
        ds = self._make_ds()
        sampler = BatchSampler(ds, batch_size=4, shuffle=False)
        x, y = next(iter(sampler))
        self.assertEqual(x.shape, (4, 9))
        self.assertEqual(y.shape, (4, 9))

    def test_all_blocks_covered(self):
        ds = self._make_ds(100, 9)  # 10 blocks
        sampler = BatchSampler(ds, batch_size=3, shuffle=False)
        total = sum(x.shape[0] for x, _ in sampler)
        self.assertEqual(total, 10)

    def test_shuffle_changes_order(self):
        ds = self._make_ds(100, 9)
        rng = np.random.default_rng(42)
        s1 = BatchSampler(ds, batch_size=10, shuffle=True, rng=rng)
        xs1 = [x for x, _ in s1]

        rng2 = np.random.default_rng(99)
        s2 = BatchSampler(ds, batch_size=10, shuffle=True, rng=rng2)
        xs2 = [x for x, _ in s2]

        # Different seeds → different order (with overwhelming probability)
        self.assertFalse(np.array_equal(xs1[0], xs2[0]))

    def test_no_shuffle_deterministic(self):
        ds = self._make_ds(100, 9)
        s1 = BatchSampler(ds, batch_size=5, shuffle=False)
        s2 = BatchSampler(ds, batch_size=5, shuffle=False)
        for (x1, _), (x2, _) in zip(s1, s2):
            np.testing.assert_array_equal(x1, x2)

    def test_len(self):
        ds = self._make_ds(100, 9)  # 10 blocks
        sampler = BatchSampler(ds, batch_size=3, shuffle=False)
        self.assertEqual(len(sampler), 4)  # ceil(10/3)


if __name__ == "__main__":
    unittest.main()
