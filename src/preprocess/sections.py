"""Split JD into requirements / nice-to-have / benefits."""

from __future__ import annotations

import re

from src.preprocess.text import normalize_text

_HEADING = re.compile(
    r"(?:^|[\n\r])\s*(?P<h>yêu cầu|yeu cau|requirements?|what you.?ll need|"
    r"job requirements?|nice to have|a plus|ưu tiên|uu tien|"
    r"plus points?|quyền lợi|quyen loi|benefits?|why you.?ll love|"
    r"mô tả công việc|mo ta cong viec|mô tả|mo ta|job description|responsibilities|"
    r"what you.?ll do|trách nhiệm|trach nhiem)\s*[:\-–]?\s*",
    re.I,
)


def split_sections(jd_text: str) -> dict[str, str]:
    text = normalize_text(jd_text, collapse_newlines=False)
    if not text:
        return {"full": "", "description": "", "requirements": "", "nice_to_have": "", "benefits": ""}

    parts: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(text))
    if not matches:
        return {
            "full": text,
            "description": text,
            "requirements": text,
            "nice_to_have": "",
            "benefits": "",
        }

    if matches[0].start() > 0:
        parts.append(("description", text[: matches[0].start()].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        key = _bucket(m.group("h"))
        parts.append((key, body))

    buckets = {"description": [], "requirements": [], "nice_to_have": [], "benefits": []}
    for key, body in parts:
        if body:
            buckets[key].append(body)
    joined = {k: "\n".join(v) for k, v in buckets.items()}
    if not joined["requirements"]:
        joined["requirements"] = joined["description"] or text
    joined["full"] = text
    return joined


def _bucket(heading: str) -> str:
    h = heading.lower()
    if any(x in h for x in ("nice", "plus", "ưu tiên", "uu tien")):
        return "nice_to_have"
    if any(x in h for x in ("quyền", "quyen", "benefit", "love")):
        return "benefits"
    if any(x in h for x in ("yêu cầu", "yeu cau", "requirement", "need")):
        return "requirements"
    return "description"
