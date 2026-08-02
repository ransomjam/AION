"""Contract tests for the compute-device backend.

These run on whichever device ``AION_DEVICE`` selects.  On the default CPU
they assert the properties that make the CUDA path possible; on a GPU pod the
same file asserts they still hold there.  Run them on the pod before starting
a long job — that is what turns "CuPy imported fine" into "the engine works".
"""

from __future__ import annotations

import io
import unittest

import numpy as np

from aion import backend
from aion.backend import xp
from aion.nn.loss import CrossEntropyLoss
from aion.nn.ops import embedding_lookup, dropout_mask, softmax, t_sum
from aion.nn.optim import Adam
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor


class TestDevicePlacement(unittest.TestCase):
    """Everything entering the graph lands on the compute device."""

    def test_tensor_construction_places_data_on_device(self):
        t = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3))
        self.assertIs(type(t.data), type(xp.zeros(1)))

    def test_tensor_accepts_host_and_device_arrays_alike(self):
        host = Tensor(np.ones((2, 2)))
        device = Tensor(host.data)
        np.testing.assert_allclose(backend.to_host(device.data), np.ones((2, 2)))

    def test_to_host_always_returns_numpy(self):
        self.assertIsInstance(backend.to_host(Tensor(np.ones(3)).data), np.ndarray)

    def test_item_crosses_back_to_a_python_float(self):
        self.assertEqual(Tensor(np.array(2.5)).item(), 2.5)


class TestScatterAdd(unittest.TestCase):
    """Embedding gradients depend on duplicate indices accumulating."""

    def test_duplicate_indices_accumulate(self):
        target = xp.zeros((3, 2))
        backend.scatter_add(target, backend.as_index([0, 0, 2]), xp.ones((3, 2)))
        np.testing.assert_allclose(
            backend.to_host(target), [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])

    def test_embedding_backward_sums_repeated_tokens(self):
        table = Parameter(np.zeros((4, 3)), name="table")
        # Token 1 appears twice, so its row must receive twice the gradient.
        out = t_sum(embedding_lookup(table, np.array([1, 1, 3])))
        out.backward()
        grad = backend.to_host(table.grad)
        np.testing.assert_allclose(grad[1], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(grad[3], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(grad[0], [0.0, 0.0, 0.0])


class TestHostArraysCrossIn(unittest.TestCase):
    """Host-built indices and masks are accepted without manual conversion."""

    def test_cross_entropy_accepts_host_targets(self):
        logits = Parameter(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]), name="w")
        loss = CrossEntropyLoss()(logits, np.array([0, 1]))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        # Gradient over a probability simplex sums to zero on each row.
        np.testing.assert_allclose(
            backend.to_host(logits.grad).sum(axis=1), [0.0, 0.0], atol=1e-6)

    def test_dropout_accepts_a_host_sampled_mask(self):
        x = Parameter(np.ones((2, 4)), name="x")
        mask = np.array([[True, False, True, False], [False, True, False, True]])
        out = dropout_mask(x, mask, keep_prob=0.5)
        np.testing.assert_allclose(
            backend.to_host(out.data),
            [[2.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 2.0]])

    def test_attention_style_mask_tensor(self):
        # Masks are built with NumPy and wrapped in a Tensor; the additive
        # -1e9 bias must drive the masked position's softmax weight to zero.
        scores = Tensor(np.zeros((1, 3)), requires_grad=True)
        bias = Tensor(np.array([[0.0, 0.0, -1e9]]), requires_grad=False)
        from aion.nn.ops import add
        w = backend.to_host(softmax(add(scores, bias), axis=-1).data)
        self.assertAlmostEqual(float(w[0, 2]), 0.0, places=6)


class TestCheckpointsAreDeviceNeutral(unittest.TestCase):
    """What is written to disk is host data, on every device."""

    def test_optimizer_moment_arrays_are_host_arrays(self):
        p = Parameter(np.ones((2, 2)), name="p")
        p.grad = xp.ones((2, 2))
        opt = Adam([p], lr=0.1)
        opt.step()
        arrays = opt.moment_arrays()
        self.assertTrue(arrays)
        for key, value in arrays.items():
            self.assertIsInstance(value, np.ndarray, f"{key} must be a host array")
        # And they must survive a real npz round-trip.
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        buf.seek(0)
        with np.load(buf) as loaded:
            self.assertEqual(sorted(loaded.files), sorted(arrays))

    def test_optimizer_state_reloads_onto_the_device(self):
        p = Parameter(np.ones((2, 2)), name="p")
        p.grad = xp.ones((2, 2))
        opt = Adam([p], lr=0.1)
        opt.step()
        restored = Adam([p], lr=0.1)
        restored.load_state_dict(opt.state_dict(), opt.moment_arrays())
        for buffer in list(restored._m.values()) + list(restored._v.values()):
            self.assertIs(type(buffer), type(xp.zeros(1)))


class TestDeviceReporting(unittest.TestCase):

    def test_device_info_is_json_serializable_and_names_the_device(self):
        import json
        info = backend.device_info()
        self.assertIn(info["device"], ("cpu", "cuda"))
        self.assertEqual(info["device"] == "cuda", backend.is_gpu())
        json.dumps(info)  # must not raise

    def test_synchronize_and_free_pool_are_safe_on_any_device(self):
        backend.synchronize()
        backend.free_pool()


if __name__ == "__main__":
    unittest.main()
