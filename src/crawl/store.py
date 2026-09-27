"""Lưu từng job ngay khi cào xong (JSONL + fsync), thread-safe, resume được."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, write_csv, write_json
from .normalize import to_unified

logger = logging.getLogger(__name__)


class IncrementalStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.seen: set[str] = set()
        self.by_source: dict[str, set[str]] = {"itviec": set(), "topcv": set()}
        self._load_existing()

    def _load_existing(self) -> None:
        for source in ("itviec", "topcv"):
            for row in read_jsonl(self.output_dir / f"{source}.jsonl"):
                url = (row.get("url") or row.get("source_url") or "").strip()
                if url:
                    self.seen.add(url)
                    self.by_source.setdefault(source, set()).add(url)
        for row in read_jsonl(self.output_dir / "combined.jsonl"):
            url = (row.get("source_url") or row.get("url") or "").strip()
            if url:
                self.seen.add(url)
        logger.info(
            "[store] resume itviec=%s topcv=%s unique_urls=%s",
            len(self.by_source.get("itviec") or []),
            len(self.by_source.get("topcv") or []),
            len(self.seen),
        )

    def urls_for(self, source: str) -> set[str]:
        with self._lock:
            return set(self.seen)

    def count(self, source: str) -> int:
        with self._lock:
            return len(self.by_source.get(source) or [])

    def save_job(self, source: str, job: dict[str, Any]) -> bool:
        url = (job.get("url") or "").strip()
        if not url:
            return False
        with self._lock:
            if url in self.seen:
                return False
            self.seen.add(url)
            self.by_source.setdefault(source, set()).add(url)
            append_jsonl(self.output_dir / f"{source}.jsonl", job)
            append_jsonl(self.output_dir / "combined.jsonl", to_unified(job, source))
            n = len(self.by_source[source])
        logger.info("[store] saved %s #%s %s", source, n, url)
        return True

    def finalize(self) -> None:
        """Snapshot JSON/CSV từ JSONL (an toàn hơn ghi đè JSON mỗi job)."""
        with self._lock:
            for source in ("itviec", "topcv"):
                rows = read_jsonl(self.output_dir / f"{source}.jsonl")
                write_json(self.output_dir / f"{source}.json", rows)
            combined = read_jsonl(self.output_dir / "combined.jsonl")
            write_json(self.output_dir / "combined.json", combined)
            write_csv(self.output_dir / "combined.csv", combined)
            logger.info(
                "[store] snapshot itviec=%s topcv=%s combined=%s",
                len(self.by_source.get("itviec") or []),
                len(self.by_source.get("topcv") or []),
                len(combined),
            )
