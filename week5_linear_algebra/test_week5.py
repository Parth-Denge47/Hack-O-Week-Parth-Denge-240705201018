"""Tests for Week 5. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_algebra import cosine_similarity, dot, gram_schmidt, pca, power_iteration, project, rotation  # noqa: E402


class TestWeek5(unittest.TestCase):
    def test_dot_and_cosine(self):
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([2, 2], [1, 1]), 1.0)

    def test_projection_residual_is_orthogonal(self):
        u, d = np.array([3.0, 2.0]), np.array([4.0, 0.5])
        self.assertAlmostEqual(dot(u - project(u, d), d), 0.0)

    def test_gram_schmidt_orthonormal(self):
        Q = gram_schmidt([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-9)
        with self.assertRaises(ValueError):
            gram_schmidt([[1, 2], [2, 4]])

    def test_rotation_preserves_length(self):
        v = np.array([3.0, 4.0])
        self.assertAlmostEqual(np.linalg.norm(rotation(0.7) @ v), 5.0)

    def test_power_iteration_matches_numpy(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        lam, _ = power_iteration(A)
        self.assertAlmostEqual(lam, np.linalg.eigvalsh(A).max(), places=6)

    def test_pca_variance_ratio(self):
        X = np.random.default_rng(0).normal(size=(200, 3)) @ np.diag([5.0, 1.0, 0.1])
        Z, ratio, _ = pca(X, 2)
        self.assertEqual(Z.shape, (200, 2))
        self.assertGreater(ratio[0], 0.9)


if __name__ == "__main__":
    unittest.main()
