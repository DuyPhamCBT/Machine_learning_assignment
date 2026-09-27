"""Chạy crawl nhiều URL nguồn: 2 luồng (ITviec + TopCV), ghi JSONL ngay mỗi job."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .crawler import Crawler
from .io import read_jsonl
from .normalize import deduplicate
from .store import IncrementalStore

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sources(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Sources file must be an object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def scrape_source(
    source: str,
    urls: dict[str, str],
    *,
    headless: bool,
    delay_seconds: float,
    page_load_timeout: int,
    max_pages: int,
    max_jobs: int | None,
    store: IncrementalStore | None = None,
) -> list[dict[str, Any]]:
    crawler = Crawler(
        source,
        headless=headless,
        delay_seconds=delay_seconds,
        page_load_timeout=page_load_timeout,
    )
    collected: list[dict[str, Any]] = []
    already = store.count(source) if store else 0
    remaining = None if max_jobs is None else max(0, max_jobs - already)
    logger.info(
        "[%s] resume already=%s max_jobs=%s remaining=%s",
        source,
        already,
        max_jobs,
        remaining,
    )
    if remaining == 0:
        logger.info("[%s] already have %s jobs (>= max_jobs), skip", source, already)
        return []

    def on_job(job: dict[str, Any]) -> bool:
        if store is None:
            return True
        return store.save_job(source, job)

    for key, url in urls.items():
        if remaining is not None and remaining <= 0:
            break
        logger.info("=== %s / %s === %s", source, key, url)
        try:
            skip_urls = store.urls_for(source) if store else set()
            jobs = crawler.crawler(
                url,
                max_pages=max_pages,
                max_jobs=remaining,
                listing_key=key,
                skip_urls=skip_urls,
                on_job=on_job if store else None,
            )
            collected.extend(jobs)
            collected = deduplicate(collected)
            if store:
                remaining = None if max_jobs is None else max(0, max_jobs - store.count(source))
            else:
                remaining = None if max_jobs is None else max(0, max_jobs - len(collected))
            logger.info("[%s] after %s: %s unique jobs this run", source, key, len(collected))
        except Exception as exc:
            logger.error("[%s] failed source %s: %s", source, key, exc)
    return collected


def run_crawl(
    *,
    sources: list[str],
    root: Path | None = None,
    headless: bool | None = None,
    max_pages: int | None = None,
    max_jobs: int | None = None,
    delay_seconds: float | None = None,
    extra_url: str | None = None,
    use_categories: bool = False,
    workers: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    root = root or ROOT
    cfg = load_yaml(root / "config.yaml")
    crawl_cfg = cfg.get("crawl") or {}
    source_files = cfg.get("sources") or {}
    if use_categories:
        source_files = {
            "itviec": cfg.get("sources", {}).get("itviec_categories") or "data/sources/itviec_categories.json",
            "topcv": cfg.get("sources", {}).get("topcv_categories") or "data/sources/topcv_categories.json",
        }

    headless = crawl_cfg.get("headless", True) if headless is None else headless
    max_pages = int(crawl_cfg.get("max_pages", 1) if max_pages is None else max_pages)
    max_jobs = int(crawl_cfg.get("max_jobs_per_source", 40) if max_jobs is None else max_jobs)
    delay_seconds = float(crawl_cfg.get("delay_seconds", 1.2) if delay_seconds is None else delay_seconds)
    page_load_timeout = int(crawl_cfg.get("page_load_timeout", 30))
    output_dir = root / str(crawl_cfg.get("output_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = int(crawl_cfg.get("workers", 2))
    workers = max(1, min(int(workers), len(sources)))

    store = IncrementalStore(output_dir)
    results: dict[str, list[dict[str, Any]]] = {}

    def _work(source: str) -> tuple[str, list[dict[str, Any]]]:
        if extra_url:
            urls = {f"{source}_cli": extra_url}
        else:
            rel = source_files.get(source)
            if not rel:
                raise ValueError(f"No sources file mapped for {source}")
            urls = load_sources(root / rel)
        jobs = scrape_source(
            source,
            urls,
            headless=headless,
            delay_seconds=delay_seconds,
            page_load_timeout=page_load_timeout,
            max_pages=max_pages,
            max_jobs=max_jobs,
            store=store,
        )
        return source, jobs

    logger.info(
        "[pipeline] sources=%s workers=%s max_pages=%s max_jobs=%s (Selenium I/O, không dùng GPU)",
        sources,
        workers,
        max_pages,
        max_jobs,
    )

    try:
        if workers <= 1 or len(sources) == 1:
            for source in sources:
                name, jobs = _work(source)
                results[name] = jobs
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl") as pool:
                futs = {pool.submit(_work, source): source for source in sources}
                for fut in as_completed(futs):
                    source = futs[fut]
                    try:
                        name, jobs = fut.result()
                        results[name] = jobs
                    except Exception as exc:
                        logger.exception("[%s] worker failed: %s", source, exc)
                        results[source] = []
    finally:
        store.finalize()

    logger.info(
        "Wrote %s (itviec=%s topcv=%s combined=%s)",
        output_dir,
        store.count("itviec"),
        store.count("topcv"),
        len(read_jsonl(output_dir / "combined.jsonl")),
    )
    return results
