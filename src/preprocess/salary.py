"""Parse VN/EN salary strings into VND bounds."""

from __future__ import annotations

import re

from src.paths import load_config

_RANGE = re.compile(
    r"(?P<a>\d[\d\.,]*)\s*(?:-|–|to|đến|~)\s*(?P<b>\d[\d\.,]*)",
    re.I,
)
_SINGLE = re.compile(r"(?P<a>\d[\d\.,]*)")
_NEGOTIABLE = re.compile(
    r"thương lượng|thoả thuận|thoa thuan|negotiable|competitive|you'll love",
    re.I,
)


def _to_float(num: str) -> float | None:
    cleaned = num.replace(" ", "").replace(",", "")
    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_salary(raw: str | None, usd_to_vnd: float | None = None) -> dict:
    text = (raw or "").strip()
    cfg_rate = float(load_config()["usd_to_vnd"])
    rate = cfg_rate if usd_to_vnd is None else usd_to_vnd
    empty = {
        "salary_min_vnd": None,
        "salary_max_vnd": None,
        "salary_mid_vnd": None,
        "salary_unknown": True,
        "salary_currency": None,
    }
    if not text or _NEGOTIABLE.search(text):
        return empty

    low = text.lower()
    is_usd = bool(re.search(r"usd|\$|đô", low))
    is_vnd = bool(re.search(r"vnd|vnđ|đồng|triệu|trieu|tr\b|million", low))
    m = _RANGE.search(text.replace(" ", "")) or _RANGE.search(text)
    if m:
        a, b = _to_float(m.group("a")), _to_float(m.group("b"))
    else:
        s = _SINGLE.search(text)
        a = b = _to_float(s.group("a")) if s else None
    if a is None or b is None:
        return empty
    if a > b:
        a, b = b, a

    unit_million = bool(re.search(r"triệu|trieu|\btr\b|million", low))
    if is_usd and not unit_million:
        a_vnd, b_vnd = a * rate, b * rate
        currency = "USD"
    elif unit_million or (not is_usd and a < 1000):
        a_vnd, b_vnd = a * 1_000_000, b * 1_000_000
        currency = "VND"
    elif is_vnd or a >= 1_000_000:
        a_vnd, b_vnd = a, b
        currency = "VND"
    else:
        a_vnd, b_vnd = a * 1_000_000, b * 1_000_000
        currency = "VND"

    return {
        "salary_min_vnd": float(a_vnd),
        "salary_max_vnd": float(b_vnd),
        "salary_mid_vnd": float((a_vnd + b_vnd) / 2),
        "salary_unknown": False,
        "salary_currency": currency,
    }
