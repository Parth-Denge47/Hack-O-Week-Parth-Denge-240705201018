"""Tests for Week 10. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clustering import KMeans, agglomerative, elbow  # noqa: E402


class TestWeek10(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(300, centers=3, cluster_std=0.7, random_state=1)

    def test_kmeans_recovers_blobs(self):
        self.assertGreater(adjusted_rand_score(self.y, KMeans(3).fit(self.X).labels_), 0.95)

    def test_inertia_decreases_with_k(self):
        e = elbow(self.X, range(1, 6))
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(e, e[1:])))

    def test_agglomerative_labels(self):
        _, labels = agglomerative(self.X, 3)
        self.assertEqual(len(np.unique(labels)), 3)
        self.assertGreater(adjusted_rand_score(self.y, labels), 0.95)

    def test_dbscan_beats_kmeans_on_moons(self):
        Xm, ym = make_moons(300, noise=0.06, random_state=0)
        db = adjusted_rand_score(ym, DBSCAN(eps=0.2).fit_predict(Xm))
        km = adjusted_rand_score(ym, KMeans(2).fit(Xm).labels_)
        self.assertGreater(db, km)


if __name__ == "__main__":
    unittest.main()
