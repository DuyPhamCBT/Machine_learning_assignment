"""Làm sạch combined.jsonl → data/interim/jobs_clean.pkl"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io import load_raw_jobs  # noqa: E402
from src.paths import COMBINED_JSONL, INTERIM_DIR  # noqa: E402
from src.preprocess.pipeline import preprocess_jobs  # noqa: E402


def run_preprocess(input_path: Path | None = None, output_dir: Path | None = None):
    input_path = Path(input_path) if input_path else COMBINED_JSONL
    output_dir = Path(output_dir) if output_dir else INTERIM_DIR
    if not input_path.exists():
        raise SystemExit(f"Không thấy JD thô: {input_path}\nChạy crawler trước, hoặc truyền --input.")
    rows = load_raw_jobs(input_path)
    if not rows:
        raise SystemExit(f"File rỗng: {input_path}")
    df = preprocess_jobs(rows, output_dir=output_dir)
    print(f"[preprocess] input={input_path}")
    print(f"[preprocess] raw={len(rows)} clean={len(df)}")
    print(f"[preprocess] wrote={output_dir / 'jobs_clean.pkl'}")
    if df.empty:
        raise SystemExit("Không còn JD nào sau khi lọc.")
    print("\nrole_family:\n", df["role_family"].value_counts().to_string())
    print("\nlocation_norm:\n", df["location_norm"].value_counts().to_string())
    print("\nlevel_norm:\n", df["level_norm"].value_counts().to_string())
    if "source" in df.columns:
        print("\nsource:\n", df["source"].value_counts().to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Làm sạch JD đã cào")
    parser.add_argument("--input", type=Path, default=COMBINED_JSONL)
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DIR)
    args = parser.parse_args()
    run_preprocess(args.input, args.output_dir)


if __name__ == "__main__":
    main()
