"""Seniority parser shared by JD preprocess and CV parse.

JD: đọc title trước (Trưởng nhóm / Team lead → lead).
CV: ưu tiên intern/fresher vì sinh viên hay viết 'lead a team' trong đồ án.
Không dùng `alias in blob` (substring) — 'lead' không còn khớp 'leadership'.
"""

from __future__ import annotations

import re
import unicodedata

from src.preprocess.experience import parse_experience

LEVEL_MAP = {
    "intern": 0,
    "fresher": 1,
    "junior": 2,
    "middle": 3,
    "mid": 3,
    "senior": 4,
    "lead": 5,
    "manager": 5,
    "principal": 5,
}

# (canonical, phrases) — phrase khớp Unicode, không cần \b cho tiếng Việt có dấu.
_LEAD = (
    "lead",
    (
        r"trưởng\s*nhóm",
        r"truong\s*nhom",
        r"trưởng\s*phòng",
        r"truong\s*phong",
        r"trưởng\s*dự\s*án",
        r"team\s*lead",
        r"tech\s*lead",
        r"technical\s*lead",
        r"group\s*lead",
        r"squad\s*lead",
        r"engineering\s*manager",
        r"head\s*of",
        r"\bprincipal\b",
        r"\bdirector\b",
        r"\bmanager\b",
        r"\bstaff\s+engineer\b",
        r"\blead\b",
        r"quản\s*lý",
        r"quan\s*ly",
    ),
)
_SENIOR = (
    "senior",
    (r"\bsenior\b", r"\bsr\.?\b", r"chuyên\s*gia", r"chuyen\s*gia"),
)
_MIDDLE = (
    "middle",
    (r"\bmiddle\b", r"\bmid-level\b", r"\bmid level\b", r"\bmid\b"),
)
_JUNIOR = (
    "junior",
    (r"\bjunior\b", r"\bjr\.?\b"),
)
_FRESHER = (
    "fresher",
    (r"\bfresher\b", r"\bfresh\b", r"mới\s*ra\s*trường", r"moi\s*ra\s*truong"),
)
_INTERN = (
    "intern",
    (r"\bintern(ship)?\b", r"thực\s*tập", r"thuc\s*tap"),
)

# Title JD: chức danh cao trước. CV: intern/fresher trước.
JD_ORDER = (_LEAD, _SENIOR, _MIDDLE, _JUNIOR, _FRESHER, _INTERN)
CV_ORDER = (_INTERN, _FRESHER, _JUNIOR, _MIDDLE, _SENIOR, _LEAD)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _match(blob: str, order: tuple) -> str | None:
    if not blob.strip():
        return None
    folded = _fold(blob)
    for canon, patterns in order:
        for pat in patterns:
            if re.search(pat, blob, re.I) or re.search(pat, folded, re.I):
                return canon
    return None


def parse_level(raw: str | None, title: str | None = None) -> dict:
    title = title or ""
    raw = raw or ""
    found = None
    if title.strip():
        found = _match(title, JD_ORDER)
        if found is None:
            found = _match(raw, JD_ORDER)
    else:
        found = _match(raw, CV_ORDER)

    if found is None:
        mid = parse_experience(raw, title)["years_mid"]
        if mid < 0.8:
            found = "intern"
        elif mid < 1.5:
            found = "fresher"
        elif mid < 2.5:
            found = "junior"
        elif mid < 4.5:
            found = "middle"
        elif mid < 7:
            found = "senior"
        else:
            found = "lead"

    try:
        from src.paths import load_config

        mapping = load_config().get("level_map") or LEVEL_MAP
    except Exception:
        mapping = LEVEL_MAP
    ordinal = int(mapping.get(found, LEVEL_MAP.get(found, 3)))
    return {"level_norm": found, "level_ordinal": ordinal}
