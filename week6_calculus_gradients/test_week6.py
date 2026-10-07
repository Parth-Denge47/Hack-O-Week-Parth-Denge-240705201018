"""Tests for Week 6. Author: Parth Denge | PRN: 240705201018"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculus_gradients import (Value, adam, bowl, bowl_grad, gradient_descent, momentum_descent,  # noqa: E402
                                numerical_derivative, numerical_gradient)


class TestWeek6(unittest.TestCase):
    def test_numerical_derivative(self):
        self.assertAlmostEqual(numerical_derivative(lambda x: x ** 2, 3.0), 6.0, places=4)

    def test_numerical_gradient(self):
        g = numerical_gradient(lambda p: p[0] ** 2 + 3 * p[1], np.array([2.0, 5.0]))
        np.testing.assert_allclose(g, [4.0, 3.0], atol=1e-4)

    def test_autograd_matches_numeric(self):
        def f(a, b):
            return (a * b + a ** 2 - b / 2).tanh()
        x, y = Value(0.7), Value(-0.4)
        f(x, y).backward()
        h = 1e-6
        dx = (f(Value(0.7 + h), Value(-0.4)).data - f(Value(0.7 - h), Value(-0.4)).data) / (2 * h)
        dy = (f(Value(0.7), Value(-0.4 + h)).data - f(Value(0.7), Value(-0.4 - h)).data) / (2 * h)
        self.assertAlmostEqual(x.grad, dx, places=5)
        self.assertAlmostEqual(y.grad, dy, places=5)

    def test_autograd_reused_node(self):
        a = Value(3.0)
        (a * a + a).backward()  # d/da (a^2 + a) = 2a + 1 = 7
        self.assertAlmostEqual(a.grad, 7.0)

    def test_exp_gradient(self):
        a = Value(1.0)
        a.exp().backward()
        self.assertAlmostEqual(a.grad, math.e)

    def test_optimisers_converge(self):
        for path in (gradient_descent(bowl_grad, [3.5, 1.2], lr=0.09, steps=200),
                     momentum_descent(bowl_grad, [3.5, 1.2], steps=300),
                     adam(bowl_grad, [3.5, 1.2], steps=300)):
            self.assertLess(bowl(path[-1]), 1e-2)


if __name__ == "__main__":
    unittest.main()
