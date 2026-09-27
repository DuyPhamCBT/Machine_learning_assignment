"""Safe BeautifulSoup helpers (cùng pattern với JobPulse extracting_info)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_text(element: Any) -> Optional[str]:
    try:
        text = element.get_text(strip=True)
        return text or None
    except Exception:
        return None


def safe_attr(element: Any, attr: str) -> Optional[str]:
    try:
        value = element.get(attr) if hasattr(element, "get") else element[attr]
        if value is None:
            return None
        return str(value).strip() or None
    except Exception:
        return None


def safe_find(parent: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return parent.find(*args, **kwargs)
    except Exception:
        logger.debug("safe_find missed: args=%s kwargs=%s", args, kwargs)
        return None
