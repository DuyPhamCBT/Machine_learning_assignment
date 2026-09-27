"""Facade chọn scraper theo nguồn — cùng API với JobPulse `Crawler`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .itviec import ITViecScraper
from .topcv import TopCVScraper

OnJob = Callable[[dict[str, Any]], bool]


class Crawler:
    def __init__(
        self,
        source: str,
        headless: bool = True,
        delay_seconds: float = 1.2,
        page_load_timeout: int = 30,
    ) -> None:
        self.source = source.lower().strip()
        self.headless = headless
        self.delay_seconds = delay_seconds
        self.page_load_timeout = page_load_timeout

    def crawler(
        self,
        url: str,
        max_pages: int = 1,
        max_jobs: int | None = None,
        listing_key: str | None = None,
        skip_urls: set[str] | None = None,
        on_job: OnJob | None = None,
    ) -> list[dict[str, Any]]:
        kwargs = dict(
            headless=self.headless,
            delay_seconds=self.delay_seconds,
            page_load_timeout=self.page_load_timeout,
        )
        extra = dict(
            max_pages=max_pages,
            max_jobs=max_jobs,
            listing_key=listing_key,
            skip_urls=skip_urls,
            on_job=on_job,
        )
        if self.source == "itviec":
            return ITViecScraper(**kwargs).scrape_jobs(url, **extra)
        if self.source == "topcv":
            return TopCVScraper(**kwargs).scrape_jobs(url, **extra)
        raise ValueError(f"Unsupported source: {self.source}. Use 'itviec' or 'topcv'.")
