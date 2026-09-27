"""Đánh giá cụm + Precision@K + 3 persona."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import PERSONAS, internal_metrics, retrieval_by_role_family  # noqa: E402
from src.paths import (  # noqa: E402
    JOBS_PKL,
    JOB_VECTORS_REDUCED,
    KMEANS_PATH,
    KNN_PATH,
    META_PATH,
    PIPELINE_PATH,
    REPORTS_DIR,
    SPACE_PATH,
)
from src.recommend.knn import recommend  # noqa: E402


def run_evaluate() -> dict:
    missing = [p for p in (PIPELINE_PATH, JOB_VECTORS_REDUCED, SPACE_PATH, KMEANS_PATH, KNN_PATH, JOBS_PKL) if not p.exists()]
    if missing:
        raise SystemExit("Thiếu artifact:\n" + "\n".join(str(p) for p in missing))
    pipe = joblib.load(PIPELINE_PATH)
    space = joblib.load(SPACE_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    knn = joblib.load(KNN_PATH)
    x = np.load(JOB_VECTORS_REDUCED)
    df = pd.read_pickle(JOBS_PKL)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    labels = df["cluster_id"].to_numpy()
    internal = internal_metrics(x, labels)
    retrieval = retrieval_by_role_family(df, x, pipe, kmeans, recommend, k=10, n_queries=40, space=space)
    personas = []
    for p in PERSONAS:
        vec, canon = pipe.transform_user(p["skills"], p["years"], p["level_ordinal"])
        vec = np.asarray(space.transform(vec))
        recs = recommend(vec[0], canon, p["years"], p["level_ordinal"], df, knn, kmeans, top_n=5)
        personas.append(
            {
                "name": p["name"],
                "note": p["note"],
                "skills": canon,
                "top_titles": recs["title"].tolist() if not recs.empty else [],
                "top_families": recs["role_family"].tolist() if not recs.empty else [],
                "missing_example": recs["missing_skills"].iloc[0] if not recs.empty else [],
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "n_jobs": meta.get("n_jobs", int(len(df))),
        "best_k": meta.get("best_k"),
        "internal": internal,
        "retrieval": retrieval,
        "personas": personas,
    }
    (REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


def main() -> None:
    run_evaluate()


if __name__ == "__main__":
    main()
