"""Run every weekly test suite. Author: Parth Denge | PRN: 240705201018"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    failed = 0
    for test in sorted(glob.glob(os.path.join(ROOT, "week*", "test_*.py"))):
        rel = os.path.relpath(test, ROOT)
        r = subprocess.run([sys.executable, test], capture_output=True, text=True)
        status = "PASS" if r.returncode == 0 else "FAIL"
        summary = [ln for ln in r.stderr.strip().splitlines() if ln.startswith(("Ran", "OK", "FAILED"))]
        print(f"[{status}] {rel}  {' | '.join(summary)}")
        if r.returncode:
            failed += 1
            print(r.stderr)
    print(f"\n{'ALL TESTS PASSED' if not failed else str(failed) + ' suite(s) failed'}  - Parth Denge | PRN 240705201018")
    return failed


if __name__ == "__main__":
    sys.exit(main())
