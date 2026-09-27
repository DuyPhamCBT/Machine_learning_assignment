"""Chạy preprocess → features → kmeans → knn → evaluate (đã có combined.jsonl)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_features import run_features  # noqa: E402
from scripts.evaluate import run_evaluate  # noqa: E402
from scripts.preprocess import run_preprocess  # noqa: E402
from scripts.train_kmeans import run_kmeans  # noqa: E402
from scripts.train_knn import run_knn  # noqa: E402


def main() -> None:
    run_preprocess()
    run_features()
    run_kmeans()
    run_knn()
    run_evaluate()


if __name__ == "__main__":
    main()
