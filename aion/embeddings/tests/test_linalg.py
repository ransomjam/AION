"""Tests for linalg primitives: dot, cosine_similarity, mean_vector, pca2."""

import math
import unittest

from aion.embeddings.linalg import (
    add_, cosine_similarity, dot, mean_vector, norm, pca2, scale_,
)


class TestDotAndNorm(unittest.TestCase):
    def test_dot_basic(self):
        self.assertAlmostEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)

    def test_dot_orthogonal(self):
        self.assertAlmostEqual(dot([1, 0], [0, 1]), 0.0)

    def test_norm(self):
        self.assertAlmostEqual(norm([3, 4]), 5.0)

    def test_add_inplace(self):
        v = [1.0, 2.0]
        add_(v, [1.0, 1.0], 2.0)
        self.assertAlmostEqual(v[0], 3.0)
        self.assertAlmostEqual(v[1], 4.0)

    def test_scale_inplace(self):
        v = [2.0, 4.0]
        scale_(v, 0.5)
        self.assertAlmostEqual(v[0], 1.0)
        self.assertAlmostEqual(v[1], 2.0)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)


class TestMeanVector(unittest.TestCase):
    def test_mean_of_two(self):
        result = mean_vector([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 3.0)

    def test_mean_of_one(self):
        result = mean_vector([[5.0, 6.0]])
        self.assertAlmostEqual(result[0], 5.0)


class TestPCA2(unittest.TestCase):
    def test_returns_one_point_per_row(self):
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = pca2(matrix)
        self.assertEqual(len(result), 3)
        for x, y in result:
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_empty_matrix(self):
        self.assertEqual(pca2([]), [])

    def test_2d_data_preserves_structure(self):
        # Points on a line in 2D — first PC should capture all variance.
        # Eigenvectors have arbitrary sign; check the projection is monotone
        # in either ascending or descending order.
        matrix = [[float(i), float(i)] for i in range(10)]
        result = pca2(matrix)
        self.assertEqual(len(result), 10)
        xs = [p[0] for p in result]
        self.assertTrue(xs == sorted(xs) or xs == sorted(xs, reverse=True))


if __name__ == "__main__":
    unittest.main()
