"""Chuẩn hoá job thô (ITviec / TopCV) sang schema dùng cho pipeline gợi ý việc."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id(source: str, url: str) -> str:
    path = urlparse(url or "").path.rstrip("/")
    slug = path.split("/")[-1].replace(".html", "") if path else "unknown"
    return f"{source}:{slug}"


def _skills(tags: str | None) -> list[str]:
    if not tags:
        return []
    return [s.strip() for s in tags.split(",") if s.strip()]


def to_unified(job: dict[str, Any], source: str) -> dict[str, Any]:
    desc = job.get("descriptions") or ""
    req = job.get("requirements") or ""
    jd_text = "\n\n".join(part for part in (desc, req) if part).strip()
    url = job.get("url") or ""
    return {
        "source": source,
        "source_url": url,
        "job_id": _job_id(source, url),
        "crawled_at": job.get("crawled_at") or utc_now(),
        "title": job.get("title") or "",
        "company": job.get("company") or "",
        "location_raw": job.get("location") or "",
        "salary_raw": job.get("salary") or "",
        "experience_raw": job.get("experience") or "",
        "level_raw": job.get("level") or "",
        "skills_raw": _skills(job.get("tags")),
        "jd_html": "",
        "jd_text": jd_text,
        "employment_type": job.get("mode") or job.get("type_of_work") or "",
        "logo": job.get("logo") or "",
        "job_category": job.get("job_cat") or "",
        "descriptions": desc,
        "requirements": req,
        "education": job.get("education") or "",
        "listing_key": job.get("listing_key") or "",
    }


def deduplicate(jobs: list[dict[str, Any]], key: str = "url") -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        value = (job.get(key) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(job)
    return out
