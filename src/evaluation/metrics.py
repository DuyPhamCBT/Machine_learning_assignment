"""Unsupervised + retrieval evaluation helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.paths import REPORTS_DIR


def internal_metrics(x: np.ndarray, labels: np.ndarray) -> dict:
    if len(set(labels)) < 2:
        return {"silhouette": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(x, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "cluster_sizes": {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }


def retrieval_by_role_family(
    df: pd.DataFrame,
    x: np.ndarray,
    pipeline,
    kmeans,
    recommend_fn,
    k: int = 10,
    n_queries: int = 80,
    seed: int = 0,
    space=None,
) -> dict:
    """Hold-out jobs: query with their own skills, measure Precision@K on role_family."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 20:
        return {"precision_at_k": None, "n_queries": 0}
    idx = rng.choice(n, size=min(n_queries, n // 4), replace=False)
    hits = []
    for i in idx:
        row = df.iloc[int(i)]
        skills = list(row.get("skills_norm") or [])
        if len(skills) < 2:
            continue
        mask = np.ones(n, dtype=bool)
        mask[int(i)] = False
        knn = NearestNeighbors(metric="cosine", n_neighbors=min(k, mask.sum()), algorithm="brute")
        knn.fit(x[mask])
        sub = df.loc[mask].reset_index(drop=True)
        vec, canon = pipeline.transform_user(
            skills=skills,
            years=float(row.get("years_mid") or 2),
            level_ordinal=int(row["level_ordinal"]) if pd.notna(row.get("level_ordinal")) else 2,
        )
        if space is not None:
            vec = np.asarray(space.transform(vec))
        recs = recommend_fn(
            user_vec=vec[0],
            user_skills=canon,
            user_years=float(row.get("years_mid") or 2),
            user_level=int(row["level_ordinal"]) if pd.notna(row.get("level_ordinal")) else 2,
            df=sub,
            knn=knn,
            kmeans=kmeans,
            top_n=k,
        )
        if recs.empty:
            continue
        precision = float((recs["role_family"] == row["role_family"]).mean())
        hits.append(precision)
    report = {
        "precision_at_k": float(np.mean(hits)) if hits else None,
        "k": k,
        "n_queries": len(hits),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrieval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


PERSONAS = [
    {
        "name": "Fresher Python",
        "skills": ["python", "django", "postgresql", "sql", "git"],
        "years": 0.5,
        "level": "fresher",
        "level_ordinal": 1,
        "note": "Mới ra trường, backend Python cơ bản.",
    },
    {
        "name": "Mid React",
        "skills": ["javascript", "typescript", "react", "nextjs", "redux", "html", "css", "git"],
        "years": 3.0,
        "level": "middle",
        "level_ordinal": 3,
        "note": "Frontend 3 năm, stack React/Next.",
    },
    {
        "name": "Senior DevOps",
        "skills": ["linux", "docker", "kubernetes", "aws", "terraform", "ci/cd", "python", "prometheus"],
        "years": 6.0,
        "level": "senior",
        "level_ordinal": 4,
        "note": "DevOps/SRE, IaC và Kubernetes.",
    },
]
