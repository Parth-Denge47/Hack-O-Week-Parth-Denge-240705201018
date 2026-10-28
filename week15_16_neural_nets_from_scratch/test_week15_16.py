"""Tests for Weeks 15 & 16. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_persistence import load_pickle, roundtrip, save_pickle, train_model  # noqa: E402
from neural_net import (ACTIVATIONS, GATE_X, GATES, LOSSES, Adam, NeuralNet, Perceptron, SGD,  # noqa: E402
                        gradient_check)


class TestPersistence(unittest.TestCase):
    def test_pickle_and_joblib_roundtrip_give_identical_predictions(self):
        model, Xte, _ = train_model()
        with tempfile.TemporaryDirectory() as d:
            rep = roundtrip(model, Xte, d)
        self.assertEqual(set(rep), {"pickle", "joblib", "joblib (compress=3)"})
        self.assertTrue(all(r["identical"] for r in rep.values()))

    def test_saved_pipeline_includes_preprocessing(self):
        model, Xte, yte = train_model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pkl")
            save_pickle(model, path)
            self.assertEqual(load_pickle(path).score(Xte, yte), model.score(Xte, yte))


class TestBuildingBlocks(unittest.TestCase):
    def test_activation_derivatives_match_numerical(self):
        z = np.linspace(-3, 3, 41) + 0.013  # avoid the ReLU kink at 0
        for name, (f, df) in ACTIVATIONS.items():
            num = (f(z + 1e-6) - f(z - 1e-6)) / 2e-6
            np.testing.assert_allclose(df(z), num, atol=1e-5, err_msg=name)

    def test_loss_gradients_match_numerical(self):
        rng = np.random.default_rng(0)
        p, y = rng.uniform(0.1, 0.9, (5, 1)), rng.integers(0, 2, (5, 1)).astype(float)
        for name, (f, df) in LOSSES.items():
            num = np.zeros_like(p)
            for i in range(len(p)):
                d = np.zeros_like(p); d[i] = 1e-6
                num[i] = (f(p + d, y) - f(p - d, y)) / 2e-6
            np.testing.assert_allclose(df(p, y), num, atol=1e-5, err_msg=name)

    def test_perceptron_learns_and_or_but_not_xor(self):
        for g in ("AND", "OR"):
            t = np.array(GATES[g])
            self.assertTrue((Perceptron().fit(GATE_X, t).predict(GATE_X) == t).all(), g)
        t = np.array(GATES["XOR"])
        self.assertFalse((Perceptron(epochs=200).fit(GATE_X, t).predict(GATE_X) == t).all())

    def test_adam_first_step_is_about_lr(self):
        p = [np.array([1.0])]
        Adam(lr=0.01).step(p, [np.array([5.0])])
        self.assertAlmostEqual(float(p[0][0]), 0.99, places=6)

    def test_sgd_momentum_accumulates(self):
        p, opt = [np.array([0.0])], SGD(lr=1.0, momentum=0.5)
        opt.step(p, [np.array([1.0])]); opt.step(p, [np.array([1.0])])
        self.assertAlmostEqual(float(p[0][0]), -2.5)  # -1, then -1 + (0.5 * -1) = -1.5 more


class TestNeuralNet(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(6, 3)), rng.integers(0, 2, (6, 1)).astype(float)
        for hidden in ("tanh", "sigmoid", "relu"):
            self.assertLess(gradient_check(NeuralNet([3, 5, 4, 1], hidden=hidden, seed=2), X, y), 1e-4, hidden)

    def test_solves_xor_which_perceptron_cannot(self):
        y = np.array(GATES["XOR"], dtype=float).reshape(-1, 1)
        net = NeuralNet([2, 4, 1], seed=1)
        hist = net.fit(GATE_X, y, Adam(0.05), epochs=400)
        self.assertEqual(net.predict(GATE_X).ravel().tolist(), GATES["XOR"])
        self.assertLess(hist[-1], hist[0])

    def test_adam_beats_plain_sgd_on_moons(self):
        from sklearn.datasets import make_moons
        X, y = make_moons(300, noise=0.2, random_state=0)
        y = y.reshape(-1, 1).astype(float)
        final = {}
        for name, opt in [("sgd", SGD(0.1)), ("adam", Adam(0.01))]:
            final[name] = NeuralNet([2, 16, 1], hidden="relu", seed=0).fit(X, y, opt, epochs=100, batch=64)[-1]
        self.assertLess(final["adam"], final["sgd"])


if __name__ == "__main__":
    unittest.main()
