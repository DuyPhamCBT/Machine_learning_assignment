"""Load crawler JSONL / JSON into a list of job dicts."""

from __future__ import annotations

import json
from pathlib import Path


def load_raw_jobs(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json" and not path.name.endswith(".jsonl"):
        data = json.loads(text)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows
