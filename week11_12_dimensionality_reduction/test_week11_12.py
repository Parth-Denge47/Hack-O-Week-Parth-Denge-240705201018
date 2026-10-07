"""Tests for Weeks 11 & 12. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dimensionality_reduction import components_for_variance, pca_from_scratch, reconstruct  # noqa: E402


class TestWeek11And12(unittest.TestCase):
    def setUp(self):
        self.X = load_digits().data

    def test_scratch_pca_matches_sklearn_variance(self):
        _, ratio = pca_from_scratch(self.X, 5)
        np.testing.assert_allclose(ratio, PCA(5).fit(self.X).explained_variance_ratio_, atol=1e-8)

    def test_projection_shape_and_centering(self):
        Z, _ = pca_from_scratch(self.X, 3)
        self.assertEqual(Z.shape, (len(self.X), 3))
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-8)

    def test_reconstruction_error_shrinks_with_more_components(self):
        errs = [((self.X - reconstruct(self.X, k)) ** 2).mean() for k in (2, 10, 40)]
        self.assertGreater(errs[0], errs[1])
        self.assertGreater(errs[1], errs[2])
        self.assertAlmostEqual(((self.X - reconstruct(self.X, 64)) ** 2).mean(), 0.0, places=10)

    def test_components_for_variance(self):
        k90, k99 = components_for_variance(self.X, 0.90), components_for_variance(self.X, 0.99)
        self.assertLess(k90, k99)
        self.assertLess(k90, 64)


if __name__ == "__main__":
    unittest.main()
