"""Tests for Week 4. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numpy_pandas_dataviz import (clean_and_enrich, make_sales_data, monthly_revenue,  # noqa: E402
                                  pairwise_distances, revenue_summary, zscore)


class TestWeek4(unittest.TestCase):
    def test_zscore(self):
        z = zscore(np.random.default_rng(1).normal(5, 3, size=(100, 2)))
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-9)

    def test_pairwise_distances(self):
        d = pairwise_distances(np.array([[0.0, 0.0], [3.0, 4.0]]))
        self.assertAlmostEqual(d[0, 1], 5.0)
        self.assertEqual(d[0, 0], 0.0)

    def test_cleaning_removes_nans(self):
        df = clean_and_enrich(make_sales_data())
        self.assertEqual(df["unit_price"].isna().sum(), 0)
        self.assertTrue((df["revenue"] > 0).all())

    def test_aggregations(self):
        df = clean_and_enrich(make_sales_data())
        self.assertAlmostEqual(revenue_summary(df).to_numpy().sum(), df["revenue"].sum(), places=1)
        self.assertAlmostEqual(monthly_revenue(df).sum(), df["revenue"].sum(), places=1)


if __name__ == "__main__":
    unittest.main()
