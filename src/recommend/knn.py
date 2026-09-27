"""KNN cosine recommendation with level/experience factors and skill-gap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.paths import load_config


def _ordinal(value, default: int = 2) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, float) and np.isnan(value):
            return default
    except (TypeError, ValueError):
        pass
    return int(value)


def embed_user(pipeline, space, skills, years, level_ordinal, extra_text: str = ""):
    """CV/form → cùng không gian với JD lúc train (Features + optional PCA space)."""
    vec, canon = pipeline.transform_user(skills, years, level_ordinal, extra_text)
    if space is not None:
        vec = np.asarray(space.transform(vec))
    return vec, canon


def _level_factor(user_level: int, job_level: int) -> float:
    gap = abs(int(user_level) - int(job_level))
    if gap <= 1:
        return 1.0
    if gap == 2:
        return 0.55
    return 0.22


def _exp_factor(user_years: float, job_min: float, job_max: float) -> float:
    if job_min <= user_years <= job_max:
        return 1.0
    if user_years < job_min:
        gap = job_min - user_years
    else:
        gap = user_years - job_max
    return float(np.exp(-0.35 * gap))


def _coverage(user_skills: list[str], job_must: list[str]) -> float:
    must = [s for s in (job_must or []) if s]
    if not must:
        return 0.5
    inter = len(set(user_skills) & set(must))
    return inter / len(set(must))


def suitable_max_level_gap(user_level: int) -> int:
    """Intern/fresher chỉ lệch tối đa 1 cấp; các cấp khác giữ 2."""
    return 1 if int(user_level) <= 1 else 2


def recommend(
    user_vec: np.ndarray,
    user_skills: list[str],
    user_years: float,
    user_level: int,
    df: pd.DataFrame,
    knn: NearestNeighbors,
    kmeans,
    top_n: int | None = None,
    location: str | None = None,
    remote_only: bool = False,
    salary_min_vnd: float | None = None,
    max_level_gap: int | None = 2,
    apply_filters: bool = True,
    rank_by: str = "score",
    min_skill_overlap: int = 0,
) -> pd.DataFrame:
    cfg = load_config()
    top_n = top_n or int(cfg["knn"]["default_top_n"])
    # Top liên quan: lân cận cosine. Danh sách siết cấp: quét rộng vì nearest thường lệch senior/lead.
    if max_level_gap is None and not apply_filters:
        n_fetch = min(len(df), max(top_n * 20, knn.n_neighbors, 200))
    elif max_level_gap is not None and int(max_level_gap) <= 1:
        n_fetch = len(df)
    else:
        n_fetch = min(len(df), max(top_n * 20, knn.n_neighbors, 200))
    distances, indices = knn.kneighbors(user_vec.reshape(1, -1), n_neighbors=n_fetch)
    distances, indices = distances[0], indices[0]
    cosine_sim = 1.0 - distances

    cluster_id = int(kmeans.predict(user_vec.reshape(1, -1))[0])
    rows = []
    for dist_i, idx in enumerate(indices):
        job = df.iloc[int(idx)]
        if apply_filters:
            if location and location != "Tất cả" and location != "All":
                if job.get("location_norm") not in {location, "Remote"} and not job.get("is_remote"):
                    continue
            if remote_only and not bool(job.get("is_remote")):
                continue
            if salary_min_vnd is not None and not job.get("salary_unknown"):
                mid = job.get("salary_mid_vnd")
                if pd.notna(mid) and float(mid) < salary_min_vnd:
                    continue
        job_level = _ordinal(job.get("level_ordinal"), 2)
        gap = abs(int(user_level) - job_level)
        if max_level_gap is not None and gap > int(max_level_gap):
            continue
        must = list(job.get("skills_must") or job.get("skills_norm") or [])
        all_sk = list(job.get("skills_norm") or [])
        overlap = sorted(set(user_skills) & set(all_sk))
        if min_skill_overlap and len(overlap) < int(min_skill_overlap):
            continue
        missing = sorted(set(must) - set(user_skills))
        cov = _coverage(user_skills, must)
        sim = float(cosine_sim[dist_i])
        lf = _level_factor(user_level, job_level)
        ef = _exp_factor(user_years, float(job.get("years_min") or 0), float(job.get("years_max") or 10))
        score = sim * (0.75 + 0.25 * lf) * (0.75 + 0.25 * ef) * (0.8 + 0.2 * cov)
        rows.append(
            {
                "rank": 0,
                "score": score,
                "cosine": sim,
                "level_factor": lf,
                "exp_factor": ef,
                "skill_coverage": cov,
                "level_gap": gap,
                "title": job.get("title"),
                "company": job.get("company"),
                "location_norm": job.get("location_norm"),
                "salary_raw": job.get("salary_raw"),
                "salary_mid_vnd": job.get("salary_mid_vnd"),
                "level_norm": job.get("level_norm"),
                "level_ordinal": job_level,
                "years_mid": job.get("years_mid"),
                "role_family": job.get("role_family"),
                "cluster_id": int(job.get("cluster_id") or 0),
                "cluster_name": job.get("cluster_name"),
                "user_cluster_id": cluster_id,
                "overlap_skills": overlap,
                "missing_skills": missing[:12],
                "is_remote": bool(job.get("is_remote")),
                "source": job.get("source"),
                "source_url": job.get("source_url"),
                "job_id": job.get("job_id"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = "cosine" if rank_by == "cosine" else "score"
    if sort_col not in out.columns:
        sort_col = "score"
    out = out.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["match_pct"] = (out["score"] / max(out["score"].max(), 1e-6) * 100).round(1)
    return out
