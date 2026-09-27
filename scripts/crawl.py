"""CLI: python scripts/crawl.py --source both --max-pages 1 --max-jobs 5"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.crawl.pipeline import run_crawl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cào JD IT từ ITviec và TopCV.")
    parser.add_argument("--source", choices=("itviec", "topcv", "both"), default="both")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--url", default=None, help="Chỉ cào 1 URL (kèm --source itviec|topcv)")
    parser.add_argument("--categories", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    )
    args = build_parser().parse_args()
    if args.url and args.source == "both":
        raise SystemExit("Khi dùng --url hãy chọn --source itviec hoặc --source topcv")
    sources = ["itviec", "topcv"] if args.source == "both" else [args.source]
    run_crawl(
        sources=sources,
        root=ROOT,
        headless=not args.no_headless,
        max_pages=args.max_pages,
        max_jobs=args.max_jobs,
        delay_seconds=args.delay,
        extra_url=args.url,
        use_categories=args.categories,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
