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


def keep_it_job(row, taxonomy, min_it_skills: int, min_it_skills_if_it_family: int) -> bool:
    if is_non_it_title(row.get("title")):
        return False
    n = count_it_skills(row.get("skills_norm"), taxonomy)
    family = row.get("role_family") or "Other"
    if family in IT_ROLE_FAMILIES:
        return n >= min_it_skills_if_it_family
    return n >= min_it_skills


def filter_it_corpus(df: pd.DataFrame, taxonomy) -> tuple[pd.DataFrame, dict]:
    cfg = (load_config().get("corpus") or {})
    min_it = int(cfg.get("min_it_skills", 3))
    min_fam = int(cfg.get("min_it_skills_if_it_family", 2))
    before = int(len(df))
    keep = df.apply(lambda row: keep_it_job(row, taxonomy, min_it, min_fam), axis=1)
    dropped = df.loc[~keep]
    out = df.loc[keep].copy()
    reasons = {
        "non_it_title": int(df["title"].fillna("").map(is_non_it_title).sum()),
        "too_few_it_skills": int((~keep & ~df["title"].fillna("").map(is_non_it_title)).sum()),
    }
    stats = {
        "n_before": before,
        "n_after": int(len(out)),
        "n_dropped": before - int(len(out)),
        "min_it_skills": min_it,
        "min_it_skills_if_it_family": min_fam,
        "drop_reasons": reasons,
        "dropped_role_family": dropped["role_family"].value_counts().to_dict() if not dropped.empty else {},
        "kept_role_family": out["role_family"].value_counts().to_dict() if not out.empty else {},
    }
    return out, stats
