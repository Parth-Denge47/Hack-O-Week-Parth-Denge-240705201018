"""Weeks 15 & 16 (a): Model deployment basics - saving and loading trained models with pickle and joblib.

Author: Parth Denge | PRN: 240705201018
"""
import os
import pickle
import sys
import tempfile
import time

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner  # noqa: E402


def train_model(seed: int = 0):
    """A full Pipeline (scaler + model) is saved as ONE object, so preprocessing travels with the model."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(200, random_state=seed)).fit(Xtr, ytr)
    return model, Xte, yte


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # only ever unpickle files you trust: unpickling can execute code


def roundtrip(model, X, directory):
    """Save with pickle and joblib, reload, and report size, load time and whether predictions are identical."""
    expected = model.predict_proba(X)
    report = {}
    for name, save, load, ext in [("pickle", save_pickle, load_pickle, "pkl"),
                                  ("joblib", joblib.dump, joblib.load, "joblib"),
                                  ("joblib (compress=3)", lambda m, p: joblib.dump(m, p, compress=3),
                                   joblib.load, "gz.joblib")]:
        path = os.path.join(directory, f"model.{ext}")
        save(model, path)
        t = time.perf_counter()
        restored = load(path)
        report[name] = {"kb": os.path.getsize(path) / 1024, "load_ms": (time.perf_counter() - t) * 1000,
                        "identical": bool(np.array_equal(restored.predict_proba(X), expected))}
    return report


def main():
    banner("Weeks 15-16a - Model persistence with pickle / joblib")
    model, Xte, yte = train_model()
    print(f"Test accuracy of the trained pipeline: {model.score(Xte, yte):.4f}\n")
    with tempfile.TemporaryDirectory() as d:
        rep = roundtrip(model, Xte, d)
    print(f"{'format':20s} {'size (KB)':>10s} {'load (ms)':>10s} {'same predictions':>17s}")
    for n, r in rep.items():
        print(f"{n:20s} {r['kb']:10.0f} {r['load_ms']:10.1f} {str(r['identical']):>17s}")
    sample = Xte[:1]
    print("\nServing one request with the reloaded model: class", int(model.predict(sample)[0]))


if __name__ == "__main__":
    main()
