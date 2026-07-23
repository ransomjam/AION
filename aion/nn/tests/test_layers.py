"""Tests for Module, Parameter, layers, and Sequential."""

import unittest
import numpy as np
from aion.nn.tensor import Tensor
from aion.nn.parameter import Parameter
from aion.nn.module import Module
from aion.nn.layers import EmbeddingLayer, Linear, ReLU, Sigmoid, Tanh
from aion.nn.sequential import Sequential
from aion.nn.ops import t_sum as _sum


# ── Parameter ─────────────────────────────────────────────────────────────────

class TestParameter(unittest.TestCase):
    def test_is_tensor(self):
        p = Parameter(np.ones((3, 4)), name="W")
        self.assertIsInstance(p, Tensor)

    def test_requires_grad_true_by_default(self):
        p = Parameter(np.ones(5))
        self.assertTrue(p.requires_grad)

    def test_requires_grad_false(self):
        p = Parameter(np.ones(5), requires_grad=False)
        self.assertFalse(p.requires_grad)

    def test_zero_grad_sets_none(self):
        p = Parameter(np.ones(3), requires_grad=True)
        p.grad = np.ones(3)
        p.zero_grad()
        self.assertIsNone(p.grad)

    def test_name_stored(self):
        p = Parameter(np.zeros(2), name="bias")
        self.assertEqual(p.name, "bias")


# ── Module registration ───────────────────────────────────────────────────────

class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.W = Parameter(np.ones((2, 3)), name="W")
        self.b = Parameter(np.zeros(3), name="b")

    def forward(self, x):
        from aion.nn.ops import add, matmul
        return add(matmul(x, self.W), self.b)


class TestModuleRegistration(unittest.TestCase):
    def test_parameters_registered(self):
        m = SimpleModel()
        params = m.parameters()
        self.assertEqual(len(params), 2)

    def test_parameter_names(self):
        m = SimpleModel()
        names = {p.name for p in m.parameters()}
        self.assertEqual(names, {"W", "b"})

    def test_zero_grad(self):
        m = SimpleModel()
        for p in m.parameters():
            p.grad = np.ones_like(p.data)
        m.zero_grad()
        for p in m.parameters():
            self.assertIsNone(p.grad)

    def test_param_count(self):
        m = SimpleModel()
        self.assertEqual(m.param_count(), 2 * 3 + 3)

    def test_submodule_parameters_collected(self):
        class Outer(Module):
            def __init__(self):
                super().__init__()
                self.inner = SimpleModel()
                self.extra = Parameter(np.ones(5), name="extra")
            def forward(self, x): return x

        outer = Outer()
        params = outer.parameters()
        self.assertEqual(len(params), 3)  # W, b, extra

    def test_frozen_parameter_excluded_from_grad(self):
        class FrozenModel(Module):
            def __init__(self):
                super().__init__()
                self.frozen = Parameter(np.ones((2, 2)), name="frozen",
                                        requires_grad=False)
                self.active = Parameter(np.ones((2, 2)), name="active",
                                        requires_grad=True)
            def forward(self, x):
                from aion.nn.ops import add, matmul
                return add(matmul(x, self.frozen), matmul(x, self.active))

        m = FrozenModel()
        x = Tensor(np.ones((1, 2)))
        out = _sum(m(x))
        out.backward()
        self.assertIsNone(m.frozen.grad)
        self.assertIsNotNone(m.active.grad)


# ── Linear ────────────────────────────────────────────────────────────────────

class TestLinear(unittest.TestCase):
    def test_output_shape(self):
        layer = Linear(4, 8)
        x = Tensor(np.ones((3, 4)))
        out = layer(x)
        self.assertEqual(out.shape, (3, 8))

    def test_no_bias(self):
        layer = Linear(4, 8, bias=False)
        self.assertIsNone(layer.b)
        self.assertEqual(len(layer.parameters()), 1)

    def test_grad_flows_to_weights(self):
        layer = Linear(3, 2)
        x = Tensor(np.ones((1, 3)))
        out = _sum(layer(x))
        out.backward()
        self.assertIsNotNone(layer.W.grad)
        self.assertIsNotNone(layer.b.grad)

    def test_deterministic_with_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        l1 = Linear(4, 4, rng=rng1)
        l2 = Linear(4, 4, rng=rng2)
        np.testing.assert_array_equal(l1.W.data, l2.W.data)


# ── EmbeddingLayer ────────────────────────────────────────────────────────────

class TestEmbeddingLayer(unittest.TestCase):
    def test_output_shape(self):
        emb = EmbeddingLayer(10, 4)
        ids = np.array([0, 3, 7])
        out = emb(ids)
        self.assertEqual(out.shape, (3, 4))

    def test_grad_scatter_add(self):
        emb = EmbeddingLayer(5, 3)
        ids = np.array([1, 1, 3])
        out = _sum(emb(ids))
        out.backward()
        # Row 1 selected twice, row 3 once.
        self.assertIsNotNone(emb.table.grad)
        np.testing.assert_allclose(emb.table.grad[0], np.zeros(3))
        np.testing.assert_allclose(emb.table.grad[1], np.full(3, 2.0))
        np.testing.assert_allclose(emb.table.grad[3], np.full(3, 1.0))


# ── Activations ───────────────────────────────────────────────────────────────

class TestActivations(unittest.TestCase):
    def test_relu_no_params(self):
        self.assertEqual(len(ReLU().parameters()), 0)

    def test_tanh_no_params(self):
        self.assertEqual(len(Tanh().parameters()), 0)

    def test_sigmoid_no_params(self):
        self.assertEqual(len(Sigmoid().parameters()), 0)

    def test_relu_forward(self):
        x = Tensor(np.array([-1.0, 0.0, 2.0]))
        out = ReLU()(x)
        np.testing.assert_allclose(out.data, [0.0, 0.0, 2.0])


# ── Sequential ────────────────────────────────────────────────────────────────

class TestSequential(unittest.TestCase):
    def test_forward_pipeline(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = Tensor(np.ones((3, 4)))
        out = model(x)
        self.assertEqual(out.shape, (3, 2))

    def test_parameters_collected(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        # Linear(4,8): W(4,8) + b(8); Linear(8,2): W(8,2) + b(2)
        self.assertEqual(len(model.parameters()), 4)

    def test_grad_flows_through(self):
        model = Sequential(Linear(3, 4), ReLU(), Linear(4, 1))
        x = Tensor(np.ones((2, 3)))
        out = _sum(model(x))
        out.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
