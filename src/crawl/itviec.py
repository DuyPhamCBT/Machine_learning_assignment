"""Cào JD IT từ ITviec — listing rồi vào trang chi tiết (Selenium + BeautifulSoup).

Giữ selector và luồng xử lý của JobPulse (`ITViecScraper`), bổ sung phân trang
và giới hạn số job.
"""

from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from .browser import clear_browser_state, init_driver, looks_blocked
from .helpers import safe_attr, safe_find, safe_text
from .normalize import utc_now

logger = logging.getLogger(__name__)

BASE = "https://itviec.com"
# Không dùng a[href*='/it-jobs/'] — nav luôn match trước khi card hydrate.
LISTING_WAIT_CSS = "div.ipy-2, h3.imt-3, a[data-url*='/it-jobs/']"
JOB_DETAIL_RE = re.compile(r"/it-jobs/[^/?#]+-\d+(?:/)?(?:$|[?#])", re.I)


def with_page(url: str, page: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if page <= 1:
        query.pop("page", None)
    else:
        query["page"] = str(page)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class ITViecScraper:
    def __init__(
        self,
        headless: bool = True,
        delay_seconds: float = 1.2,
        page_load_timeout: int = 30,
    ) -> None:
        self.headless = headless
        self.delay_seconds = delay_seconds
        self.page_load_timeout = page_load_timeout

    def _sleep(self) -> None:
        time.sleep(self.delay_seconds + random.uniform(0.4, 1.4))

    def _extract_text(self, section) -> Optional[str]:
        try:
            items = section.find_all(["p", "li"], recursive=True)
            texts = [i.get_text(" ", strip=True) for i in items if i.get_text(strip=True)]
            return " ".join(texts) if texts else None
        except Exception:
            return None

    def _parse_listing_card(self, job) -> dict[str, Optional[str]] | None:
        data: dict[str, Optional[str]] = {
            "title": None,
            "company": None,
            "logo": None,
            "url": None,
            "job_cat": None,
            "location": None,
            "mode": None,
            "tags": None,
            "descriptions": None,
            "requirements": None,
            "salary": None,
            "experience": None,
            "source": "itviec",
            "crawled_at": utc_now(),
        }

        url_el = safe_find(job, "h3", class_="imt-3 text-break")
        raw_url = safe_attr(url_el, "data-url")
        if raw_url:
            data["url"] = raw_url.split("?lab_feature=")[0]
        if not data["url"]:
            link = job.find("a", href=True)
            href = (link.get("href") or "") if link else ""
            if "/it-jobs/" in href and href.count("/") >= 2:
                data["url"] = urljoin(BASE, href.split("?")[0])
        if not data["url"]:
            return None

        data["title"] = safe_text(safe_find(job, "h3"))
        company_el = safe_find(job, "div", class_="imy-3 d-flex align-items-center")
        data["company"] = safe_text(safe_find(company_el, "span"))
        data["logo"] = safe_attr(safe_find(company_el, "img"), "data-src") or safe_attr(
            safe_find(company_el, "img"), "src"
        )
        data["mode"] = safe_text(safe_find(job, "div", class_="text-rich-grey flex-shrink-0"))
        location_el = safe_find(
            job,
            "div",
            class_="text-rich-grey text-truncate text-nowrap stretched-link position-relative",
        )
        data["location"] = safe_attr(location_el, "title") or safe_text(location_el)

        tag_container = safe_find(job, "div", class_="imt-4 imb-3 d-flex igap-1")
        if tag_container:
            tags = [safe_text(a) for a in tag_container.find_all("a") if safe_text(a)]
            data["tags"] = ", ".join(tags) if tags else None
        return data

    def _job_links(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """Fallback khi class card JobPulse (`div.ipy-2`) không còn trên listing."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for node in soup.select("a[href], [data-url]"):
            href = node.get("href") or node.get("data-url") or ""
            if not JOB_DETAIL_RE.search(href):
                continue
            full = urljoin(BASE, href.split("?")[0].split("#")[0])
            if full in seen:
                continue
            seen.add(full)
            h3 = node if node.name == "h3" else (node.find("h3") or (node.parent.find("h3") if node.parent else None))
            title = safe_text(h3) or safe_text(node)
            if title and len(title) > 160:
                title = title[:160]
            out.append(
                {
                    "title": title,
                    "company": None,
                    "logo": None,
                    "url": full,
                    "job_cat": None,
                    "location": None,
                    "mode": None,
                    "tags": None,
                    "descriptions": None,
                    "requirements": None,
                    "salary": None,
                    "experience": None,
                    "source": "itviec",
                    "crawled_at": utc_now(),
                }
            )
        return out

    def _fill_detail(self, soup: BeautifulSoup, data: dict[str, Any]) -> dict[str, Any]:
        job_cat_div = soup.find("div", string="Job Expertise:")
        if job_cat_div:
            nxt = job_cat_div.find_next("div")
            if nxt:
                cats = [a.get_text(strip=True) for a in nxt.find_all("a") if a.get_text(strip=True)]
                data["job_cat"] = ", ".join(cats) if cats else None

        sections = soup.find_all("div", class_="imy-5 paragraph")
        if len(sections) > 0:
            data["descriptions"] = self._extract_text(sections[0])
        if len(sections) > 1:
            data["requirements"] = self._extract_text(sections[1])

        if not data.get("descriptions") or not data.get("requirements"):
            for block in soup.select("div.imy-5, section, div.paragraph"):
                heading = safe_text(safe_find(block, "h2")) or safe_text(safe_find(block, "h3")) or ""
                body = self._extract_text(block) or safe_text(block)
                low = heading.lower()
                if not data.get("descriptions") and any(k in low for k in ("description", "mô tả")):
                    data["descriptions"] = body
                if not data.get("requirements") and any(k in low for k in ("requirement", "yêu cầu")):
                    data["requirements"] = body
        return data

    def _collect_listing(
        self,
        url: str,
        max_pages: int,
        max_jobs: int | None,
        skip_urls: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        skip_urls = skip_urls or set()
        driver = init_driver(headless=self.headless, page_load_timeout=self.page_load_timeout)
        cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for page in range(1, max_pages + 1):
                page_url = with_page(url, page)
                logger.info("[itviec] listing page %s: %s", page, page_url)
                try:
                    driver.get(page_url)
                except WebDriverException as exc:
                    logger.error("[itviec] cannot open listing: %s", exc)
                    break
                try:
                    WebDriverWait(driver, self.page_load_timeout).until(
                        lambda d: d.find_elements(By.CSS_SELECTOR, LISTING_WAIT_CSS)
                    )
                except TimeoutException:
                    logger.warning("[itviec] timeout waiting listing cards, parse anyway: %s", page_url)
                time.sleep(3)
                html = driver.page_source
                title = driver.title or ""
                if looks_blocked(html, title=title):
                    logger.warning("[itviec] listing looks blocked (title=%r), stop pagination", title)
                    break
                soup = BeautifulSoup(html, "html.parser")
                jobs = soup.find_all("div", class_="ipy-2")
                fallback_links = self._job_links(soup)
                logger.info(
                    "[itviec] title=%r html=%s ipy-2=%s job_links=%s",
                    title,
                    len(html),
                    len(jobs),
                    len(fallback_links),
                )
                added = 0
                page_had = 0
                for job in jobs:
                    parsed = self._parse_listing_card(job)
                    if not parsed or not parsed.get("url"):
                        continue
                    page_had += 1
                    job_url = parsed["url"]
                    if job_url in seen or job_url in skip_urls:
                        continue
                    seen.add(job_url)
                    cards.append(parsed)
                    added += 1
                    if max_jobs is not None and len(cards) >= max_jobs:
                        return cards
                if added == 0:
                    for item in fallback_links:
                        job_url = item["url"]
                        if not job_url:
                            continue
                        page_had += 1
                        if job_url in seen or job_url in skip_urls:
                            continue
                        seen.add(job_url)
                        cards.append(item)
                        added += 1
                        if max_jobs is not None and len(cards) >= max_jobs:
                            return cards
                if added == 0 and page_had == 0:
                    debug_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "_debug_itviec_listing.html"
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(html, encoding="utf-8")
                    logger.info("[itviec] dumped listing HTML -> %s", debug_path)
                    logger.info("[itviec] no cards on page %s", page)
                    break
                logger.info(
                    "[itviec] page %s: DOM=%s skipped_already=%s +%s new (queue=%s)",
                    page,
                    max(page_had, len(jobs)),
                    max(0, page_had - added),
                    added,
                    len(cards),
                )
                self._sleep()
        finally:
            driver.quit()
        return cards

    def scrape_jobs(
        self,
        url: str,
        max_pages: int = 1,
        max_jobs: int | None = None,
        listing_key: str | None = None,
        skip_urls: set[str] | None = None,
        on_job=None,
    ) -> list[dict[str, Optional[str]]]:
        skip_urls = skip_urls or set()
        cards = self._collect_listing(
            url, max_pages=max_pages, max_jobs=max_jobs, skip_urls=skip_urls
        )
        logger.info("[itviec] found %s listing cards", len(cards))
        if not cards:
            return []

        job_data: list[dict[str, Optional[str]]] = []
        detail_driver = init_driver(headless=self.headless, page_load_timeout=self.page_load_timeout)
        try:
            for idx, data in enumerate(cards, 1):
                if listing_key:
                    data["listing_key"] = listing_key
                job_url = data.get("url") or ""
                if job_url in skip_urls:
                    continue
                logger.info("[itviec] detail %s/%s: %s", idx, len(cards), job_url)
                try:
                    detail_driver.get(job_url)
                    try:
                        WebDriverWait(detail_driver, self.page_load_timeout).until(
                            lambda d: d.execute_script("return document.body.innerText.length") > 250
                        )
                    except TimeoutException:
                        logger.warning("[itviec] timeout detail: %s", job_url)
                        self._sleep()
                        continue
                    if looks_blocked(detail_driver.page_source, title=detail_driver.title):
                        logger.warning("[itviec] blocked on detail, skip: %s", job_url)
                        self._sleep()
                        continue
                    soup = BeautifulSoup(detail_driver.page_source, "html.parser")
                    self._fill_detail(soup, data)
                except Exception as exc:
                    logger.error("[itviec] skip job: %s", exc)
                finally:
                    clear_browser_state(detail_driver)
                    self._sleep()

                jd_len = len(data.get("descriptions") or "") + len(data.get("requirements") or "")
                if data.get("url") and jd_len >= 40:
                    if not data.get("descriptions"):
                        data["descriptions"] = data.get("requirements")
                    if not data.get("requirements"):
                        data["requirements"] = data.get("descriptions")
                    if on_job is not None and on_job(data) is False:
                        continue
                    job_data.append(data)
                if max_jobs is not None and len(job_data) >= max_jobs:
                    break
        finally:
            detail_driver.quit()

        logger.info("[itviec] done, scraped %s complete jobs", len(job_data))
        return job_data
