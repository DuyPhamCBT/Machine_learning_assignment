"""Cào JD IT từ TopCV — listing rồi vào trang chi tiết (Selenium + BeautifulSoup).

Giữ selector / parse brand vs việc-làm của JobPulse (`TopCVScraper`), bổ sung
phân trang, xử lý logo None, và giới hạn số job.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from .browser import clear_browser_state, init_driver, looks_blocked
from .helpers import safe_attr, safe_find, safe_text
from .normalize import utc_now

logger = logging.getLogger(__name__)

LISTING_WAIT_CSS = "div.job-item-search-result"
DETAIL_WAIT_CSS = (
    "div.box-job-information-detail-item__text, "
    "div.job-description__item, "
    "div.premium-job-description__box--content"
)


def with_page(url: str, page: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if page <= 1:
        query.pop("page", None)
    else:
        query["page"] = str(page)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class TopCVScraper:
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

    def _job_href(self, job) -> Optional[str]:
        """Ưu tiên link JD (/viec-lam/ hoặc /brand/), không lấy a đầu tiên (logo/company)."""
        if job is None:
            return None
        for a in job.select("h3 a[href], a[href*='/viec-lam/'], a[href*='/brand/']"):
            href = a.get("href") or ""
            if "/viec-lam/" in href or "/brand/" in href:
                return href
        return safe_attr(safe_find(job, "a"), "href")

    def _extract_job_info(self, job) -> tuple:
        title = safe_text(safe_find(job, "h3"))
        company = safe_text(safe_find(job, "a", class_="company"))
        img_tag = job.find("img") if job is not None else None
        logo = None
        if img_tag is not None:
            logo = img_tag.get("src") or img_tag.get("data-src") or None
        href = self._job_href(job)
        job_url = href.split("?ta_source")[0] if href else None
        address = job.find("label", class_="address") if job is not None else None
        salary_box = None
        if job is not None:
            salary_box = job.find("label", class_="title-salary") or job.find("label", class_="salary")
        exp_box = job.find("label", class_="exp") if job is not None else None
        location = safe_text(safe_find(address, "span"))
        salary = safe_text(safe_find(salary_box, "span"))
        exp = safe_text(safe_find(exp_box, "span"))
        return title, company, logo, job_url, location, salary, exp

    def _parse_brand_job(self, soup) -> tuple:
        def extract_general_info(div):
            label = div.select_one(".general-information-data__label")
            value = div.select_one(".general-information-data__value")
            if label and value:
                return label, value
            label = div.find("strong")
            value = div.find("span")
            if label and value:
                return label, value
            return None, None

        def extract_description_requirement(div):
            h2 = div.select_one("h2.premium-job-description__box--title")
            content_div = div.select_one("div.premium-job-description__box--content")
            if h2 and content_div:
                return h2, content_div
            h2 = div.select_one("h2.title")
            content_div = div.select_one("div.content-tab")
            if h2 and content_div:
                return h2, content_div
            return None, None

        descriptions = requirements = edu = type_of_work = None
        for div in soup.select("div.premium-job-description__box, div.box-info"):
            title, content = extract_description_requirement(div)
            if not title:
                continue
            heading = title.get_text(strip=True)
            if heading == "Mô tả công việc":
                descriptions = safe_text(content)
            elif heading == "Yêu cầu ứng viên":
                requirements = safe_text(content)
            if descriptions and requirements:
                break

        for div in soup.select("div.general-information-data, div.box-item"):
            label, value = extract_general_info(div)
            if not label:
                continue
            heading = label.get_text(strip=True)
            if heading == "Hình thức làm việc":
                type_of_work = safe_text(value)
            elif heading == "Học vấn":
                edu = safe_text(value)
            if type_of_work and edu:
                break
        return descriptions, requirements, edu, type_of_work

    def _parse_job_detail_v2(self, soup) -> tuple:
        """Layout TopCV 2026: box-job-information-detail-item + __text."""
        descriptions = requirements = edu = type_of_work = None
        for div in soup.select("div.box-job-information-detail-item"):
            heading_el = div.select_one("h2.box-job-information-detail-item__title--title") or div.find("h2")
            content = div.select_one("div.box-job-information-detail-item__text")
            if not heading_el:
                continue
            heading = heading_el.get_text(strip=True)
            if heading == "Mô tả công việc":
                descriptions = safe_text(content)
            elif heading == "Yêu cầu ứng viên":
                requirements = safe_text(content)

        for item in soup.select("div.box-job-information-general-info-list__item"):
            label = safe_text(item.select_one(".box-job-information-general-info-list__item--content-title"))
            value = safe_text(item.select_one(".box-job-information-general-info-list__item--content-desc"))
            if label == "Hình thức làm việc":
                type_of_work = value
            elif label == "Học vấn":
                edu = value
        return descriptions, requirements, edu, type_of_work

    def _parse_job_detail_legacy(self, soup) -> tuple:
        descriptions = requirements = edu = type_of_work = None
        for div in soup.select("div.job-description__item"):
            h3 = safe_find(div, "h3")
            content = safe_find(div, "div", "job-description__item--content")
            if not h3:
                continue
            title = h3.get_text(strip=True)
            if title == "Mô tả công việc":
                descriptions = safe_text(content)
            elif title == "Yêu cầu ứng viên":
                requirements = safe_text(content)
            if descriptions and requirements:
                break

        for div in soup.find_all("div", class_="box-general-group-info"):
            title_div = safe_find(div, "div", "box-general-group-info-title")
            value_div = safe_find(div, "div", "box-general-group-info-value")
            if not title_div:
                continue
            label = title_div.get_text(strip=True)
            value = safe_text(value_div)
            if label == "Hình thức làm việc":
                type_of_work = value
            elif label == "Học vấn":
                edu = value
            if type_of_work and edu:
                break
        return descriptions, requirements, edu, type_of_work

    def _parse_job_detail(self, soup) -> tuple:
        descriptions, requirements, edu, type_of_work = self._parse_job_detail_v2(soup)
        if descriptions and requirements:
            return descriptions, requirements, edu, type_of_work
        d2, r2, e2, t2 = self._parse_job_detail_legacy(soup)
        return (
            descriptions or d2,
            requirements or r2,
            edu or e2,
            type_of_work or t2,
        )

    def _experience_from_detail(self, soup) -> Optional[str]:
        for item in soup.select("div.box-header-job-list-info__item"):
            label = (safe_text(item.select_one(".list-info__content__title")) or "").strip()
            value = safe_text(item.select_one(".list-info__content__desc"))
            if label == "Kinh nghiệm" and value:
                return value
        return None

    def _fill_detail(self, soup: BeautifulSoup, job_url: str, data: dict[str, Any]) -> dict[str, Any]:
        job_cat_div = soup.find("div", string=lambda x: x and "Chuyên môn:" in x)
        if job_cat_div:
            nxt = job_cat_div.find_next("div")
            if nxt:
                cats = [a.get_text(strip=True) for a in nxt.find_all("a") if a.get_text(strip=True)]
                data["job_cat"] = ", ".join(cats) if cats else None

        if "topcv.vn/brand/" in (job_url or ""):
            descriptions, requirements, edu, type_of_work = self._parse_brand_job(soup)
            if not (descriptions and requirements):
                d2, r2, e2, t2 = self._parse_job_detail(soup)
                descriptions = descriptions or d2
                requirements = requirements or r2
                edu = edu or e2
                type_of_work = type_of_work or t2
        else:
            descriptions, requirements, edu, type_of_work = self._parse_job_detail(soup)
            if not (descriptions and requirements):
                d2, r2, e2, t2 = self._parse_brand_job(soup)
                descriptions = descriptions or d2
                requirements = requirements or r2
                edu = edu or e2
                type_of_work = type_of_work or t2

        data["descriptions"] = descriptions
        data["requirements"] = requirements
        data["education"] = edu
        data["type_of_work"] = type_of_work
        if not data.get("experience"):
            data["experience"] = self._experience_from_detail(soup)
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
                logger.info("[topcv] listing page %s: %s", page, page_url)
                try:
                    driver.get(page_url)
                except WebDriverException as exc:
                    logger.error("[topcv] cannot open listing: %s", exc)
                    break
                try:
                    WebDriverWait(driver, self.page_load_timeout).until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR, LISTING_WAIT_CSS)) >= 10
                    )
                except TimeoutException:
                    logger.warning("[topcv] timeout waiting listing cards: %s", page_url)
                    if not driver.find_elements(By.CSS_SELECTOR, LISTING_WAIT_CSS):
                        break
                time.sleep(1.5)
                html = driver.page_source
                title = driver.title or ""
                if looks_blocked(html, title=title):
                    logger.warning("[topcv] listing looks blocked (title=%r), stop pagination", title)
                    break
                soup = BeautifulSoup(html, "html.parser")
                jobs = soup.find_all("div", class_="job-item-search-result")
                if not jobs:
                    logger.info("[topcv] no cards on page %s", page)
                    break
                added = 0
                skipped = 0
                for job in jobs:
                    try:
                        job_title, company, logo, job_url, location, salary, exp = self._extract_job_info(job)
                    except Exception as exc:
                        logger.debug("[topcv] skip card: %s", exc)
                        continue
                    if not job_url:
                        continue
                    if job_url in seen or job_url in skip_urls:
                        skipped += 1
                        continue
                    seen.add(job_url)
                    cards.append(
                        {
                            "title": job_title,
                            "company": company,
                            "logo": logo,
                            "url": job_url,
                            "location": location,
                            "salary": salary,
                            "descriptions": None,
                            "requirements": None,
                            "experience": exp,
                            "education": None,
                            "type_of_work": None,
                            "job_cat": None,
                            "source": "topcv",
                            "crawled_at": utc_now(),
                        }
                    )
                    added += 1
                    if max_jobs is not None and len(cards) >= max_jobs:
                        return cards
                logger.info(
                    "[topcv] page %s: DOM=%s skipped_already=%s +%s new (queue=%s)",
                    page,
                    len(jobs),
                    skipped,
                    added,
                    len(cards),
                )
                if added == 0 and not jobs:
                    break
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
        logger.info("[topcv] found %s listing cards", len(cards))
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
                logger.info("[topcv] detail %s/%s: %s", idx, len(cards), job_url)
                try:
                    detail_driver.get(job_url)
                    try:
                        WebDriverWait(detail_driver, self.page_load_timeout).until(
                            lambda d: d.find_elements(By.CSS_SELECTOR, DETAIL_WAIT_CSS)
                            or d.execute_script("return document.body.innerText.length") > 250
                        )
                    except TimeoutException:
                        logger.warning("[topcv] timeout detail: %s", job_url)
                        self._sleep()
                        continue
                    time.sleep(1.5)
                    if looks_blocked(detail_driver.page_source, title=detail_driver.title):
                        logger.warning("[topcv] blocked on detail, skip: %s", job_url)
                        self._sleep()
                        continue
                    soup = BeautifulSoup(detail_driver.page_source, "html.parser")
                    self._fill_detail(soup, job_url, data)
                    logger.info(
                        "[topcv] parsed desc=%s req=%s exp=%s title=%r",
                        bool(data.get("descriptions")),
                        bool(data.get("requirements")),
                        data.get("experience"),
                        detail_driver.title,
                    )
                except Exception as exc:
                    logger.error("[topcv] skip job: %s", exc)
                finally:
                    clear_browser_state(detail_driver)
                    self._sleep()

                if data.get("url") and data.get("requirements") and data.get("descriptions"):
                    if on_job is not None and on_job(data) is False:
                        continue
                    job_data.append(data)
                else:
                    logger.info(
                        "[topcv] incomplete skip url=%s desc=%s req=%s",
                        job_url,
                        bool(data.get("descriptions")),
                        bool(data.get("requirements")),
                    )
                if max_jobs is not None and len(job_data) >= max_jobs:
                    break
        finally:
            detail_driver.quit()

        logger.info("[topcv] done, scraped %s complete jobs", len(job_data))
        return job_data
