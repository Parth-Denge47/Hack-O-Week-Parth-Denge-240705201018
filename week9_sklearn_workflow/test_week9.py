"""Tests for Week 9. Author: Parth Denge | PRN: 240705201018"""
import os
import sys
import unittest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn_workflow import build_pipeline, make_churn_data  # noqa: E402


class TestWeek9(unittest.TestCase):
    def setUp(self):
        df = make_churn_data(800)
        X, y = df.drop(columns="churn"), df["churn"]
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def test_data_has_missing_values_and_both_classes(self):
        df = make_churn_data(800)
        self.assertGreater(df.isna().sum().sum(), 0)
        self.assertEqual(set(df["churn"].unique()), {0, 1})

    def test_pipeline_handles_missing_and_unseen_categories(self):
        pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(self.Xtr, self.ytr)
        Xnew = self.Xte.copy()
        Xnew.iloc[0, Xnew.columns.get_loc("contract")] = "never-seen-before"
        self.assertEqual(len(pipe.predict(Xnew)), len(Xnew))

    def test_pipeline_beats_chance(self):
        pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(self.Xtr, self.ytr)
        self.assertGreater(roc_auc_score(self.yte, pipe.predict_proba(self.Xte)[:, 1]), 0.65)


if __name__ == "__main__":
    unittest.main()
