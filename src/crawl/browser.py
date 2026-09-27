"""Chrome WebDriver factory — cùng cách JobPulse khởi tạo Selenium."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

_DRIVER_PATH: str | None = None
_DRIVER_LOCK = threading.Lock()


def _resolve_chromedriver(path: str) -> str:
    """webdriver-manager trên Windows đôi khi trả file THIRD_PARTY_NOTICES."""
    p = Path(path)
    if p.name.startswith("THIRD_PARTY_NOTICES"):
        for candidate in p.parent.iterdir():
            name = candidate.name.lower()
            if candidate.is_file() and name.startswith("chromedriver") and "notices" not in name:
                return str(candidate)
    return path


def chromedriver_path() -> str:
    global _DRIVER_PATH
    with _DRIVER_LOCK:
        if _DRIVER_PATH is None:
            logger.info("Downloading / locating ChromeDriver...")
            _DRIVER_PATH = _resolve_chromedriver(ChromeDriverManager().install())
            logger.info("ChromeDriver: %s", _DRIVER_PATH)
        return _DRIVER_PATH


def chrome_options(headless: bool = True) -> Options:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return options


def init_driver(headless: bool = True, page_load_timeout: int = 30) -> webdriver.Chrome:
    driver = webdriver.Chrome(
        service=Service(chromedriver_path()),
        options=chrome_options(headless=headless),
    )
    driver.set_page_load_timeout(page_load_timeout)
    return driver


_CHALLENGE_TITLES = (
    "just a moment",
    "attention required",
    "access denied",
    "verify you are human",
    "um...sorry",
)


def looks_blocked(html: str, title: str | None = None) -> bool:
    """Chỉ dừng khi đúng trang challenge.

    ITviec/TopCV HTML luôn chứa chữ cloudflare/captcha trong script CDN,
    nên không được scan raw HTML — lần test trước bị false positive.
    """
    page_title = (title or "").lower()
    if any(t in page_title for t in _CHALLENGE_TITLES):
        return True
    if not html:
        return True
    lower = html.lower()
    if "<html" not in lower:
        return True
    # Trang challenge Cloudflare thường rất ngắn và không có job card.
    if "cf-browser-verification" in lower or "cf-challenge-running" in lower:
        if "ipy-2" not in lower and "job-item-search-result" not in lower:
            return True
    return False


def clear_browser_state(driver: webdriver.Chrome) -> None:
    try:
        driver.delete_all_cookies()
    except Exception:
        pass
    try:
        driver.execute_cdp_cmd("Network.clearBrowserCookies", {})
        driver.execute_cdp_cmd("Network.clearBrowserCache", {})
    except Exception:
        pass
