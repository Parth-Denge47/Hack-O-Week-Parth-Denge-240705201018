"""Tests for Week 7. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regression import (GDLinearRegression, Lasso, LinearRegression, Ridge, mse,  # noqa: E402
                        polynomial_features, r2_score)


def line_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    w = np.array([2.0, -1.0, 0.0])
    return X, X @ w + 4.0 + rng.normal(0, 0.05, 120)


class TestWeek7(unittest.TestCase):
    def test_normal_equation_recovers_weights(self):
        X, y = line_data()
        m = LinearRegression().fit(X, y)
        np.testing.assert_allclose(m.w_, [4.0, 2.0, -1.0, 0.0], atol=0.05)
        self.assertGreater(r2_score(y, m.predict(X)), 0.99)

    def test_gradient_descent_matches_normal_equation(self):
        X, y = line_data()
        np.testing.assert_allclose(GDLinearRegression(lr=0.1, epochs=1500).fit(X, y).w_,
                                   LinearRegression().fit(X, y).w_, atol=1e-2)

    def test_polynomial_features_shape(self):
        self.assertEqual(polynomial_features(np.arange(5), 3).shape, (5, 3))

    def test_ridge_shrinks_weights(self):
        X, y = line_data()
        w0 = np.abs(Ridge(0.0).fit(X, y).w_[1:]).sum()
        w1 = np.abs(Ridge(500.0).fit(X, y).w_[1:]).sum()
        self.assertLess(w1, w0)

    def test_lasso_zeroes_irrelevant_feature(self):
        X, y = line_data()
        m = Lasso(alpha=0.1).fit(X, y)
        self.assertEqual(m.w_[2], 0.0)
        self.assertLess(mse(y, m.predict(X)), 0.5)


if __name__ == "__main__":
    unittest.main()
