"""Đường dẫn và config tập trung cho cả pipeline.

Mặc định: ROOT = folder project (chứa config.yaml).
Optional: set env IT_JOB_EXP=/path/to/exp để ghi model/report ra folder khác;
data/raw, dictionaries, interim (jobs_clean) vẫn dùng chung project.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _exp_root() -> Path:
    raw = os.environ.get("IT_JOB_EXP", "").strip()
    return Path(raw).resolve() if raw else PROJECT_ROOT


EXP_ROOT = _exp_root()
ROOT = EXP_ROOT

CONFIG_PATH = EXP_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = EXP_ROOT / "data" / "processed"
DICT_DIR = DATA_DIR / "dictionaries"
SOURCES_DIR = DATA_DIR / "sources"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = EXP_ROOT / "models"
REPORTS_DIR = EXP_ROOT / "reports"

SKILLS_PATH = DICT_DIR / "skills.yaml"
ALIAS_POLICY_PATH = DICT_DIR / "alias_policy.yaml"
COMBINED_JSONL = RAW_DIR / "combined.jsonl"
JOBS_CLEAN = INTERIM_DIR / "jobs_clean.pkl"
JOBS_FEATURED = PROCESSED_DIR / "jobs_featured.pkl"
JOBS_CLUSTERED = PROCESSED_DIR / "jobs_clustered.pkl"
JOBS_PKL = PROCESSED_DIR / "jobs.pkl"

JOB_VECTORS = MODELS_DIR / "job_vectors.npy"
JOB_VECTORS_REDUCED = MODELS_DIR / "job_vectors_reduced.npy"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
SPACE_PATH = MODELS_DIR / "space.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
KNN_PATH = MODELS_DIR / "knn.joblib"
META_PATH = MODELS_DIR / "meta.json"
FEATURES_META_PATH = MODELS_DIR / "features_meta.json"

DEFAULT_INPUT = COMBINED_JSONL
DEFAULT_VECTORS = JOB_VECTORS
DEFAULT_JOBS = JOBS_FEATURED
DEFAULT_META = META_PATH


def load_config() -> dict:
    path = CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Không thấy config: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs() -> None:
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, DICT_DIR, SOURCES_DIR, SAMPLES_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def artifacts_ready() -> bool:
    return all(
        p.exists()
        for p in (PIPELINE_PATH, SPACE_PATH, KMEANS_PATH, KNN_PATH, JOB_VECTORS_REDUCED, JOBS_PKL)
    )
