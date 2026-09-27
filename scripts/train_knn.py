"""Fit cosine KNN trên vector đã reduce."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.paths import (  # noqa: E402
    JOBS_CLUSTERED,
    JOBS_PKL,
    JOB_VECTORS_REDUCED,
    KNN_PATH,
    META_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    load_config,
)


def run_knn(vectors: Path | None = None, jobs: Path | None = None, meta: Path | None = None):
    vectors = Path(vectors) if vectors else JOB_VECTORS_REDUCED
    jobs = Path(jobs) if jobs else JOBS_CLUSTERED
    meta = Path(meta) if meta else META_PATH
    if not vectors.exists() or not jobs.exists():
        raise SystemExit("Thiếu job_vectors_reduced.npy hoặc jobs_clustered.pkl.\nChạy: python scripts/train_kmeans.py")
    ensure_dirs()
    x = np.load(vectors)
    df = pd.read_pickle(jobs)
    if len(df) != len(x):
        raise SystemExit(f"Lệch kích thước: jobs={len(df)} vectors={len(x)}")
    cfg = load_config()
    n_neighbors = min(int(cfg["knn"]["n_neighbors"]), max(1, len(df) - 1))
    knn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors, algorithm="brute")
    knn.fit(x)
    joblib.dump(knn, KNN_PATH)
    df.to_pickle(JOBS_PKL)
    export = df.copy()
    for col in ("skills_raw", "skills_norm", "skills_must", "skills_nice"):
        if col in export.columns:
            export[col] = export[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    export.to_csv(PROCESSED_DIR / "jobs.csv", index=False, encoding="utf-8-sig")
    if meta.exists() and meta.resolve() != (MODELS_DIR / "meta.json").resolve():
        shutil.copy2(meta, MODELS_DIR / "meta.json")
    elif not META_PATH.exists():
        META_PATH.write_text(
            json.dumps({"n_jobs": int(len(df)), "n_features": int(x.shape[1])}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[knn] jobs={len(df)} n_neighbors={n_neighbors} metric=cosine")
    print(f"[knn] wrote {KNN_PATH}")
    print(f"[knn] wrote {JOBS_PKL}")
    return knn, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit KNN cosine trên vector JD")
    parser.add_argument("--vectors", type=Path, default=JOB_VECTORS_REDUCED)
    parser.add_argument("--jobs", type=Path, default=JOBS_CLUSTERED)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    args = parser.parse_args()
    run_knn(args.vectors, args.jobs, args.meta)


if __name__ == "__main__":
    main()
