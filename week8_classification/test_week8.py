"""Tests for Week 8. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classification import (KNN, LogisticRegression, OneVsRestLogistic, accuracy, auc,  # noqa: E402
                            confusion_matrix, make_blobs, precision_recall_f1, roc_curve, sigmoid,
                            train_test_split)


class TestWeek8(unittest.TestCase):
    def setUp(self):
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(*make_blobs())

    def test_sigmoid(self):
        self.assertAlmostEqual(float(sigmoid(0)), 0.5)
        self.assertGreater(float(sigmoid(50)), 0.999)

    def test_logistic_binary(self):
        m = LogisticRegression().fit(self.Xtr, (self.ytr == 1).astype(int))
        self.assertGreater(accuracy((self.yte == 1).astype(int), m.predict(self.Xte)), 0.9)

    def test_multiclass_and_knn(self):
        self.assertGreater(accuracy(self.yte, OneVsRestLogistic().fit(self.Xtr, self.ytr).predict(self.Xte)), 0.85)
        self.assertGreater(accuracy(self.yte, KNN(5).fit(self.Xtr, self.ytr).predict(self.Xte)), 0.85)

    def test_confusion_matrix_and_prf(self):
        y, p = np.array([1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1])
        np.testing.assert_array_equal(confusion_matrix(y, p), [[1, 1], [1, 2]])
        pr, rc, f1 = precision_recall_f1(y, p)
        self.assertAlmostEqual(pr, 2 / 3)
        self.assertAlmostEqual(rc, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_auc_perfect_and_random(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(auc(*roc_curve(y, np.array([0.1, 0.2, 0.8, 0.9]))), 1.0)
        self.assertAlmostEqual(auc(*roc_curve(y, np.array([0.9, 0.8, 0.2, 0.1]))), 0.0)


if __name__ == "__main__":
    unittest.main()
