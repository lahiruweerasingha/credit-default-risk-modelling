"""Write seeded synthetic raw files (official layout) for the pipeline system test.

    python scripts/make_synthetic_data.py [--loans-per-vintage 4000]

Output: data/synthetic/raw/sample_{orig,svcg}_YYYY.txt. Nothing is derived from real data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tests.fixtures.synthetic import write_raw_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loans-per-vintage", type=int, default=4000)
    parser.add_argument("--out", default="data/synthetic/raw")
    args = parser.parse_args()
    out = write_raw_files(
        Path(args.out), list(range(2010, 2017)), args.loans_per_vintage, seed=2024
    )
    print(f"wrote synthetic raw files to {out}")


if __name__ == "__main__":
    main()
