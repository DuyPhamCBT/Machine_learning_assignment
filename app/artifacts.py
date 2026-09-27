"""Load trained artifacts from models/ and data/processed/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.skills import SkillExtractor  # noqa: E402
from src.levels import parse_level  # noqa: E402
from src.paths import (  # noqa: E402
    JOBS_PKL,
    JOB_VECTORS_REDUCED,
    KMEANS_PATH,
    KNN_PATH,
    META_PATH,
    PIPELINE_PATH,
    SPACE_PATH,
    artifacts_ready,
)


def _refresh_levels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    norms, ords = [], []
    for _, row in df.iterrows():
        parsed = parse_level(str(row.get("level_raw") or ""), str(row.get("title") or ""))
        norms.append(parsed["level_norm"])
        ords.append(parsed["level_ordinal"])
    df["level_norm"] = norms
    df["level_ordinal"] = ords
    return df


def load_artifacts() -> dict:
    pipe = joblib.load(PIPELINE_PATH)
    # Từ điển/regex mới không cần train lại vectorizer.
    pipe.extractor = SkillExtractor()
    df = _refresh_levels(pd.read_pickle(JOBS_PKL))
    return {
        "df": df,
        "X": np.load(JOB_VECTORS_REDUCED),
        "pipeline": pipe,
        "space": joblib.load(SPACE_PATH),
        "kmeans": joblib.load(KMEANS_PATH),
        "knn": joblib.load(KNN_PATH),
        "meta": json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {},
    }


__all__ = ["artifacts_ready", "load_artifacts"]
