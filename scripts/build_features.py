"""Vector hóa jobs_clean.pkl → job_vectors.npy + feature_pipeline.joblib"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.vectorizer import FeaturePipeline  # noqa: E402
from src.paths import FEATURES_META_PATH, JOBS_CLEAN, JOBS_FEATURED, MODELS_DIR, PROCESSED_DIR, ensure_dirs  # noqa: E402


def run_features(input_path: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, FeaturePipeline]:
    input_path = Path(input_path) if input_path else JOBS_CLEAN
    if not input_path.exists():
        raise SystemExit(f"Không thấy {input_path}\nChạy trước: python scripts/preprocess.py")
    df = pd.read_pickle(input_path)
    if df.empty:
        raise SystemExit("jobs_clean.pkl rỗng.")
    ensure_dirs()
    pipe = FeaturePipeline()
    x = pipe.fit_transform(df)
    joblib.dump(pipe, MODELS_DIR / "feature_pipeline.joblib")
    np.save(MODELS_DIR / "job_vectors.npy", x)
    df.to_pickle(JOBS_FEATURED)
    export = df.copy()
    for col in ("skills_raw", "skills_norm", "skills_must", "skills_nice"):
        if col in export.columns:
            export[col] = export[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    export.to_csv(PROCESSED_DIR / "jobs_featured.csv", index=False, encoding="utf-8-sig")
    meta = {
        "n_jobs": int(len(df)),
        "n_features": int(x.shape[1]),
        "skill_dim": int(pipe.skill_dim),
        "text_dim": int(pipe.text_dim),
        "struct_dim": int(pipe.struct_dim),
        "role_family_counts": df["role_family"].value_counts().to_dict() if "role_family" in df.columns else {},
        "corpus_filter": getattr(pipe, "corpus_filter_stats", {}) or {},
    }
    FEATURES_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[features] jobs={len(df)} dim={x.shape[1]} "
        f"skill={pipe.skill_dim} text={pipe.text_dim} struct={pipe.struct_dim}"
    )
    avg_skills = float(np.mean([len(s or []) for s in df["skills_norm"]]))
    print(f"[features] avg skills/job={avg_skills:.1f}")
    return df, x, pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering từ jobs_clean.pkl")
    parser.add_argument("--input", type=Path, default=JOBS_CLEAN)
    args = parser.parse_args()
    run_features(args.input)


if __name__ == "__main__":
    main()
