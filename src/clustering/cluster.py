"""K-Means training, K selection, cluster naming, 2D projection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.paths import load_config


def choose_k(x: np.ndarray, k_min: int | None = None, k_max: int | None = None) -> dict:
    cfg = load_config()["kmeans"]
    k_min = k_min or int(cfg["k_min"])
    k_max = k_max or int(cfg["k_max"])
    k_max = min(k_max, max(2, x.shape[0] // 15))
    k_min = min(k_min, k_max)
    rows = []
    best_k, best_sil = k_min, -1.0
    rng = int(cfg["random_state"])
    n_init = int(cfg["n_init"])
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=rng, n_init=n_init, max_iter=int(cfg["max_iter"]))
        labels = model.fit_predict(x)
        sil = float(silhouette_score(x, labels, metric="euclidean")) if len(set(labels)) > 1 else -1.0
        db = float(davies_bouldin_score(x, labels)) if len(set(labels)) > 1 else 99.0
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": sil,
                "davies_bouldin": db,
            }
        )
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return {"best_k": int(best_k), "search": rows}


def fit_kmeans(x: np.ndarray, k: int | None = None) -> KMeans:
    cfg = load_config()["kmeans"]
    k = k or int(cfg.get("best_k") or cfg["k_min"])
    model = KMeans(
        n_clusters=int(k),
        random_state=int(cfg["random_state"]),
        n_init=int(cfg["n_init"]),
        max_iter=int(cfg["max_iter"]),
    )
    model.fit(x)
    return model


def name_clusters(df: pd.DataFrame, labels: np.ndarray, top_n: int = 4) -> dict[int, dict]:
    profiles: dict[int, dict] = {}
    df = df.copy()
    df["_cluster"] = labels
    for cid, part in df.groupby("_cluster"):
        skill_counter: Counter[str] = Counter()
        for skills in part.get("skills_norm", []):
            skill_counter.update(list(skills or [])[:20])
        top_skills = [s for s, _ in skill_counter.most_common(top_n)]
        family = "Other"
        if "role_family" in part.columns and not part["role_family"].empty:
            mode = part["role_family"].mode()
            if not mode.empty:
                family = str(mode.iloc[0])
        titles = part["title"].str.replace(
            r"(?i)\b(intern|fresher|junior|middle|mid|senior|lead|sr|jr)\b",
            "",
            regex=True,
        ).str.strip()
        common_title = titles.mode().iloc[0] if not titles.empty else family
        median_salary = part["salary_mid_vnd"].median() if "salary_mid_vnd" in part.columns else None
        name = f"{family}: {', '.join(top_skills[:3])}" if top_skills else str(family)
        profiles[int(cid)] = {
            "cluster_id": int(cid),
            "name": name,
            "role_family": family,
            "size": int(len(part)),
            "top_skills": top_skills,
            "example_title": common_title,
            "median_salary_vnd": None if median_salary is None or pd.isna(median_salary) else float(median_salary),
            "mean_years": float(part["years_mid"].mean()) if "years_mid" in part.columns else None,
            "level_mode": part["level_norm"].mode().iloc[0] if "level_norm" in part.columns and not part["level_norm"].empty else "",
        }
    return profiles


def pca_2d(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    n = min(2, x.shape[0], x.shape[1])
    pca = PCA(n_components=n, random_state=random_state)
    coords = pca.fit_transform(x)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return coords
