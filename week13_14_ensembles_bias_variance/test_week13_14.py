"""Tests for Weeks 13 & 14. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensembles import bias_variance_decomposition, compare_models, depth_sweep, get_data, regularisation_paths  # noqa: E402


class TestWeek13And14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Xtr, cls.Xte, cls.ytr, cls.yte = get_data()

    def test_ensembles_beat_single_tree(self):
        s = compare_models(self.Xtr, self.Xte, self.ytr, self.yte)
        for name in ("Bagging", "Random Forest", "XGBoost", "LightGBM"):
            self.assertGreater(s[name][1], s["Single tree"][1], name)

    def test_single_tree_overfits(self):
        train, test = compare_models(self.Xtr, self.Xte, self.ytr, self.yte)["Single tree"]
        self.assertEqual(train, 1.0)
        self.assertGreater(train - test, 0.1)

    def test_depth_sweep_shows_generalisation_gap(self):
        _, tr, te = depth_sweep(self.Xtr, self.Xte, self.ytr, self.yte)
        self.assertLess(tr[0], tr[-1])
        self.assertGreater(tr[-1] - te[-1], tr[0] - te[0])

    def test_bias_down_variance_up_with_complexity(self):
        bv = bias_variance_decomposition(n_rounds=40, depths=(1, 12))
        self.assertGreater(bv[1][0], bv[12][0])   # shallow tree: higher bias
        self.assertLess(bv[1][1], bv[12][1])      # shallow tree: lower variance

    def test_l1_gives_sparser_models_than_l2(self):
        _, reg = regularisation_paths(self.Xtr, self.ytr, self.Xte, self.yte, cs=[0.01, 0.05])
        self.assertLess(reg["l1"][1][0], reg["l2"][1][0])


if __name__ == "__main__":
    unittest.main()
