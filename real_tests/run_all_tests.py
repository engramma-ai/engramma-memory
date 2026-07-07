"""
Run all Engramma Memory real-world tests and produce a summary report.

Usage:
    python run_all_tests.py          # Run all tests
    python run_all_tests.py -v       # Verbose output
    python run_all_tests.py -k comp  # Run only composition tests
"""

import subprocess
import sys
from pathlib import Path


def main():
    test_dir = Path(__file__).parent
    args = sys.argv[1:]

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "--tb=short",
        "-q",
        "--no-header",
    ] + args

    print("=" * 70)
    print("  ENGRAMMA MEMORY - Real-World Test Suite")
    print("=" * 70)
    print(f"\n  Running from: {test_dir}")
    print(f"  Command: {' '.join(cmd)}\n")
    print("-" * 70)

    result = subprocess.run(cmd, cwd=str(test_dir.parent))

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  SOME TESTS FAILED (exit code: {result.returncode})")
    print("=" * 70)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
