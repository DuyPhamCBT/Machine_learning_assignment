"""Lọc JD không-IT / ít skill trước khi vector hóa và K-Means."""

from __future__ import annotations

import re

import pandas as pd

from src.features.skills import COMMON_FAMILIES, DESIGN_SKILLS, SKIP_FAMILIES
from src.paths import load_config

IT_ROLE_FAMILIES = {
    "Backend",
    "Frontend",
    "Fullstack",
    "Mobile",
    "AI/ML",
    "QA",
    "DevOps",
    "Data",
    "Security",
}

_NON_IT_TITLE = re.compile(
    r"kinh\s*doanh|telesale|\bsales?\b|chăm sóc khách hàng|tư vấn bán|"
    r"thiết kế đồ họa|graphic design|nhân viên thiết kế|"
    r"\bcnc\b|mastercam|gia công phay|gia công cơ khí|lập trình cnc|"
    r"kế toán(?!.*phần mềm)|lái xe|hành chính nhân sự|"
    r"content creator|video editor|motion graphic|vfx artist",
    re.I,
)


def count_it_skills(skills: list | None, taxonomy) -> int:
    n = 0
    for skill in skills or []:
        family = taxonomy.canonical_to_family.get(skill, "")
        if family in SKIP_FAMILIES or skill in DESIGN_SKILLS:
            continue
        if family in COMMON_FAMILIES:
            n += 1
    return n


def is_non_it_title(title: str | None) -> bool:
    return bool(_NON_IT_TITLE.search(title or ""))


def keep_it_job(
    row,
    taxonomy,
    min_it_skills: int,
    min_it_skills_if_it_family: int,
    drop_non_it_title: bool = True,
) -> bool:
    if drop_non_it_title and is_non_it_title(row.get("title")):
        return False
    n = count_it_skills(row.get("skills_norm"), taxonomy)
    family = row.get("role_family") or "Other"
    if family in IT_ROLE_FAMILIES:
        return n >= min_it_skills_if_it_family
    return n >= min_it_skills


def filter_it_corpus(df: pd.DataFrame, taxonomy) -> tuple[pd.DataFrame, dict]:
    cfg = (load_config().get("corpus") or {})
    enabled = bool(cfg.get("enabled", True))
    drop_title = bool(cfg.get("drop_non_it_title", True))
    min_it = int(cfg.get("min_it_skills", 3))
    min_fam = int(cfg.get("min_it_skills_if_it_family", 2))
    before = int(len(df))
    families = (
        df["role_family"].value_counts().to_dict()
        if "role_family" in df.columns and not df.empty
        else {}
    )
    if not enabled:
        stats = {
            "n_before": before,
            "n_after": before,
            "n_dropped": 0,
            "enabled": False,
            "drop_non_it_title": False,
            "min_it_skills": min_it,
            "min_it_skills_if_it_family": min_fam,
            "drop_reasons": {"non_it_title": 0, "too_few_it_skills": 0},
            "dropped_role_family": {},
            "kept_role_family": families,
        }
        return df.copy(), stats

    keep = df.apply(
        lambda row: keep_it_job(row, taxonomy, min_it, min_fam, drop_title),
        axis=1,
    )
    dropped = df.loc[~keep]
    out = df.loc[keep].copy()
    title_hit = df["title"].fillna("").map(is_non_it_title)
    reasons = {
        "non_it_title": int(title_hit.sum()) if drop_title else 0,
        "too_few_it_skills": int((~keep & ~title_hit).sum()) if drop_title else int((~keep).sum()),
    }
    stats = {
        "n_before": before,
        "n_after": int(len(out)),
        "n_dropped": before - int(len(out)),
        "enabled": True,
        "drop_non_it_title": drop_title,
        "min_it_skills": min_it,
        "min_it_skills_if_it_family": min_fam,
        "drop_reasons": reasons,
        "dropped_role_family": dropped["role_family"].value_counts().to_dict() if not dropped.empty else {},
        "kept_role_family": out["role_family"].value_counts().to_dict() if not out.empty else {},
    }
    return out, stats
