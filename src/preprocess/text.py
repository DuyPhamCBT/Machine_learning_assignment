"""HTML strip, Unicode normalize, bilingual whitespace cleanup."""

from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")
_HWS = re.compile(r"[^\S\n]+")
_MULTI_NL = re.compile(r"\n{3,}")


def strip_html(raw: str | None, separator: str = " ") -> str:
    if not raw:
        return ""
    text = BeautifulSoup(str(raw), "lxml").get_text(separator, strip=True)
    return html.unescape(text)


def normalize_text(raw: str | None, collapse_newlines: bool = True) -> str:
    text = strip_html(raw, separator="\n" if not collapse_newlines else " ")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    if collapse_newlines:
        text = _WS.sub(" ", text).strip()
    else:
        text = _HWS.sub(" ", text)
        text = _MULTI_NL.sub("\n\n", text).strip()
    return text


def fold_ascii(raw: str) -> str:
    """Lowercase + strip accents for coarse matching (keeps original separately)."""
    text = unicodedata.normalize("NFD", raw.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text
