"""End-to-end JD cleaning: salary, experience, level, location, sections."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.paths import INTERIM_DIR, ensure_dirs, load_config
from src.preprocess.experience import parse_experience
from src.preprocess.level import parse_level
from src.preprocess.location import parse_location
from src.preprocess.role_family import infer_role_family
from src.preprocess.salary import parse_salary
from src.preprocess.sections import split_sections
from src.preprocess.text import normalize_text

_HOT_PREFIX = re.compile(r"^(?:HOT|NEW|URGENT)\s*", re.I)


def _clean_title(raw: str | None) -> str:
    title = normalize_text(raw or "")
    return _HOT_PREFIX.sub("", title).strip()


def _experience_blob(row: dict) -> str:
    raw = str(row.get("experience_raw") or "").strip()
    if raw:
        return raw
    req = str(row.get("requirements") or row.get("requirements_text") or "")
    jd = str(row.get("jd_text") or "")
    return " ".join(part for part in (row.get("title") or "", req[:2000], jd[:1500]) if part)


def preprocess_jobs(rows: list[dict], output_dir: Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    min_chars = int(cfg["features"]["min_jd_chars"])
    cleaned: list[dict] = []
    for row in rows:
        raw_jd = row.get("jd_text") or row.get("jd_html") or ""
        display = normalize_text(raw_jd, collapse_newlines=True)
        if len(display) < min_chars:
            continue
        sections = split_sections(raw_jd)
        salary = parse_salary(row.get("salary_raw"))
        exp_src = _experience_blob(row)
        title_raw = row.get("title")
        exp = parse_experience(exp_src, title_raw)
        level = parse_level(row.get("level_raw"), title_raw)
        loc_blob = " ".join(
            str(part) for part in (row.get("location_raw"), row.get("employment_type")) if part
        )
        loc = parse_location(loc_blob)
        skills_raw = row.get("skills_raw") or []
        if isinstance(skills_raw, str):
            skills_raw = [s.strip() for s in skills_raw.split(",") if s.strip()]
        title = _clean_title(title_raw)
        company = normalize_text(row.get("company") or "")
        record = {
            **row,
            "title": title,
            "company": company,
            "display_text": display,
            "requirements_text": sections["requirements"],
            "nice_to_have_text": sections["nice_to_have"],
            "benefits_text": sections["benefits"],
            "description_text": sections["description"],
            "skills_raw": skills_raw,
            **salary,
            **exp,
            **level,
            **loc,
        }
        record["role_family"] = infer_role_family(title, list(skills_raw))
        cleaned.append(record)

    df = pd.DataFrame(cleaned)
    if df.empty:
        return df
    if "source_url" in df.columns:
        df = df.drop_duplicates(subset=["source_url"], keep="first")
    key = df["title"].str.lower().str.strip() + "|" + df["company"].str.lower().str.strip()
    df = df.loc[~key.duplicated()].reset_index(drop=True)

    out = Path(output_dir) if output_dir else INTERIM_DIR
    out.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    df.to_pickle(out / "jobs_clean.pkl")
    slim = df.copy()
    if "jd_html" in slim.columns:
        slim = slim.drop(columns=["jd_html"])
    slim.to_csv(out / "jobs_clean.csv", index=False, encoding="utf-8-sig")
    return df
