"""Parse years of experience from JD/title strings."""

from __future__ import annotations

import re

_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to|đến)\s*(\d+(?:\.\d+)?)")
_PLUS = re.compile(r"(\d+(?:\.\d+)?)\s*\+|\bat least\s*(\d+)|từ\s*(\d+)", re.I)
_SINGLE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:năm|year|years|yrs|yr)\b", re.I)


def parse_experience(raw: str | None, title: str | None = None) -> dict:
    text = f"{raw or ''} {title or ''}".strip()
    low = text.lower()
    years_min: float | None = None
    years_max: float | None = None

    if re.search(r"intern|thực tập", low):
        years_min, years_max = 0.0, 0.5
    elif re.search(r"không yêu cầu|no experience|fresher", low):
        years_min, years_max = 0.0, 1.0
    else:
        m = _RANGE.search(low)
        if m:
            years_min, years_max = float(m.group(1)), float(m.group(2))
        else:
            m = _PLUS.search(low)
            if m:
                val = float(next(g for g in m.groups() if g))
                years_min, years_max = val, val + 3
            else:
                m = _SINGLE.search(low)
                if m:
                    val = float(m.group(1))
                    years_min, years_max = max(0.0, val - 1), val + 1

    if years_min is None or years_max is None:
        years_min, years_max = 1.0, 3.0
    if years_min > years_max:
        years_min, years_max = years_max, years_min

    return {
        "years_min": float(years_min),
        "years_max": float(years_max),
        "years_mid": float((years_min + years_max) / 2),
    }
