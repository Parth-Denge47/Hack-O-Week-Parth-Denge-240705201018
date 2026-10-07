"""Run every weekly demo, save the console output to output_screenshots/console_output.txt and the
plots as PNGs. Author: Parth Denge | PRN: 240705201018"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP = {"test_", "__init__"}
DEMOS = [
    "week3_python_essentials/python_essentials.py",
    "week4_numpy_pandas_dataviz/numpy_pandas_dataviz.py",
    "week5_linear_algebra/linear_algebra.py",
    "week6_calculus_gradients/calculus_gradients.py",
    "week7_regression/regression.py",
    "week8_classification/classification.py",
    "week9_sklearn_workflow/sklearn_workflow.py",
    "week10_clustering/clustering.py",
    "week11_12_dimensionality_reduction/dimensionality_reduction.py",
    "week13_14_ensembles_bias_variance/ensembles.py",
]


def main() -> None:
    os.makedirs(os.path.join(ROOT, "output_screenshots"), exist_ok=True)
    log = []
    for demo in DEMOS:
        r = subprocess.run([sys.executable, os.path.join(ROOT, demo)], capture_output=True, text=True,
                           encoding="utf-8", cwd=ROOT)
        log.append(r.stdout)
        print(r.stdout)
        if r.returncode:
            print(r.stderr)
            raise SystemExit(f"{demo} failed")
    with open(os.path.join(ROOT, "output_screenshots", "console_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))


if __name__ == "__main__":
    main()
