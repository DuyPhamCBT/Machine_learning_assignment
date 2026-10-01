"""CV skill extract: ưu tiên khối Technical Skills, bỏ intro/education."""

from __future__ import annotations

import re

_HEADER_KIND = [
    (re.compile(r"technical\s+skills", re.I), "skills"),
    (re.compile(r"kỹ\s*năng", re.I), "skills"),
    (re.compile(r"work\s+experience|employment history|kinh\s*nghiệm", re.I), "project"),
    (re.compile(r"projects?", re.I), "project"),
    (re.compile(r"dự\s*án", re.I), "project"),
    (re.compile(r"introduction|summary|objective|about me|giới\s*thiệu", re.I), "intro"),
    (re.compile(r"education|học\s*vấn|academic", re.I), "education"),
    (re.compile(r"^skills$", re.I), "skills"),
]


def split_cv_sections(text: str) -> dict[str, str]:
    """Tách CV theo heading dòng. Preamble (tên, liên hệ) không lấy skill."""
    chunks: dict[str, list[str]] = {
        "preamble": [],
        "skills": [],
        "project": [],
        "intro": [],
        "education": [],
        "other": [],
    }
    current = "preamble"
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        kind = _header_kind(line)
        if kind:
            current = kind
            continue
        chunks[current].append(line)
    return {key: "\n".join(parts) for key, parts in chunks.items()}


def _header_kind(line: str) -> str | None:
    compact = re.sub(r"[\s:]+$", "", line).strip()
    if len(compact) > 48:
        return None
    for pattern, kind in _HEADER_KIND:
        if pattern.search(compact) and (compact.lower() in {"skills", "skill"} or len(compact.split()) <= 6):
            if pattern.pattern == r"^skills$" and compact.lower() not in {"skills", "skill"}:
                continue
            return kind
    return None


def _collapse(skills: list[str], rules: dict) -> list[str]:
    have = set(skills)
    drop: set[str] = set()
    for child, parents in (rules or {}).items():
        if child in have and any(p in have for p in (parents or [])):
            drop.add(child)
    return [s for s in skills if s not in drop]


def extract_cv_skills(extractor, text: str, cfg: dict | None = None, level_ordinal: int | None = None) -> list[str]:
    cfg = cfg or {}
    sections = split_cv_sections(text)
    skills_blob = sections.get("skills") or ""
    project_blob = sections.get("project") or ""
    if not skills_blob.strip() and not project_blob.strip():
        return extractor.extract(text, order_by_position=True)

    core = extractor.extract(skills_blob, order_by_position=True) if skills_blob.strip() else []
    extras = extractor.extract(project_blob, order_by_position=True) if project_blob.strip() else []
    extras = [s for s in extras if s not in set(core)]

    if not cfg.get("ignore_intro", True):
        intro = extractor.extract(sections.get("intro") or "", order_by_position=True)
        extras.extend(s for s in intro if s not in set(core) and s not in extras)

    cap_intern = int(cfg.get("max_project_extras_intern", 3))
    cap_other = int(cfg.get("max_project_extras", 6))
    cap = cap_intern if level_ordinal is not None and int(level_ordinal) <= 1 else cap_other
    extras = extras[: max(0, cap)]

    merged = core + extras
    merged = _collapse(merged, cfg.get("collapse") or {})
    return merged
