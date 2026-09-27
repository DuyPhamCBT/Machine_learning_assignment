"""Normalize workplace location and remote/hybrid flags."""

from __future__ import annotations

import re

from src.preprocess.text import fold_ascii

_NOISE = {"dang", "hot", "new", "tuyen", "tuyen dung"}


def parse_location(raw: str | None) -> dict:
    text = fold_ascii(raw or "")
    is_remote = bool(re.search(r"remote|lam tu xa|work from home|wfh", text))
    is_hybrid = bool(re.search(r"hybrid|linh hoat", text))
    tokens = {t for t in re.split(r"\W+", text) if t}
    if tokens and tokens <= _NOISE:
        city = "Remote" if is_remote else "Other"
        return {
            "location_norm": city,
            "is_remote": is_remote or city == "Remote",
            "is_hybrid": is_hybrid,
        }
    if re.search(r"da nang|danang|dn\b", text):
        city = "Da Nang"
    elif re.search(r"ha noi|hanoi|hn\b", text):
        city = "Ha Noi"
    elif re.search(
        r"ho chi minh|hcm|sai gon|saigon|tp hcm|tphcm|hcmc|thu duc",
        text,
    ):
        city = "Ho Chi Minh"
    elif is_remote:
        city = "Remote"
    else:
        city = "Other"
    return {
        "location_norm": city,
        "is_remote": is_remote or city == "Remote",
        "is_hybrid": is_hybrid,
    }
