"""K-Means trên vector JD (space: StandardScaler + PCA + L2)."""

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

from src.clustering.cluster import choose_k, fit_kmeans, name_clusters, pca_2d  # noqa: E402
from src.clustering.space import fit_transform_space, space_from_config, space_meta  # noqa: E402
from src.paths import (  # noqa: E402
    FEATURES_META_PATH,
    JOBS_CLUSTERED,
    JOBS_FEATURED,
    JOB_VECTORS,
    JOB_VECTORS_REDUCED,
    KMEANS_PATH,
    META_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPACE_PATH,
    ensure_dirs,
    load_config,
)


def run_kmeans(vectors: Path | None = None, jobs: Path | None = None) -> tuple[pd.DataFrame, dict]:
    vectors = Path(vectors) if vectors else JOB_VECTORS
    jobs = Path(jobs) if jobs else JOBS_FEATURED
    if not vectors.exists() or not jobs.exists():
        raise SystemExit("Thiếu job_vectors.npy hoặc jobs_featured.pkl.\nChạy: python scripts/build_features.py")
    ensure_dirs()
    x_raw = np.load(vectors)
    df = pd.read_pickle(jobs)
    if len(df) != len(x_raw):
        raise SystemExit(f"Lệch kích thước: jobs={len(df)} vectors={len(x_raw)}")
    cfg = load_config()
    space = space_from_config(cfg)
    x = fit_transform_space(x_raw, space)
    if space is not None:
        joblib.dump(space, SPACE_PATH)
    np.save(JOB_VECTORS_REDUCED, np.asarray(x, dtype=np.float32))
    search = choose_k(x)
    kmeans = fit_kmeans(x, k=search["best_k"])
    labels = kmeans.predict(x)
    df = df.copy()
    df["cluster_id"] = labels
    profiles = name_clusters(df, labels)
    df["cluster_name"] = df["cluster_id"].map(lambda i: profiles[int(i)]["name"])
    coords = pca_2d(x)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    joblib.dump(kmeans, KMEANS_PATH)
    df.to_pickle(JOBS_CLUSTERED)
    export = df.copy()
    for col in ("skills_raw", "skills_norm", "skills_must", "skills_nice"):
        if col in export.columns:
            export[col] = export[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)
    export.to_csv(PROCESSED_DIR / "jobs_clustered.csv", index=False, encoding="utf-8-sig")
    corpus_filter = {}
    if FEATURES_META_PATH.exists():
        corpus_filter = json.loads(FEATURES_META_PATH.read_text(encoding="utf-8")).get("corpus_filter") or {}
    meta = {
        "n_jobs": int(len(df)),
        "n_features_raw": int(x_raw.shape[1]),
        "n_features": int(x.shape[1]),
        "best_k": int(search["best_k"]),
        "space": {**(cfg.get("space") or {}), **space_meta(space, x)},
        "k_search": search["search"],
        "cluster_profiles": {str(k): v for k, v in profiles.items()},
        "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "role_family_counts": df["role_family"].value_counts().to_dict() if "role_family" in df.columns else {},
        "corpus_filter": corpus_filter,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "cluster_profiles.json").write_text(
        json.dumps(meta["cluster_profiles"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[kmeans] jobs={len(df)} k={search['best_k']} dim={x_raw.shape[1]}→{x.shape[1]}")
    sm = space_meta(space, x)
    if sm.get("pca_explained_variance") is not None:
        print(f"[kmeans] pca explained variance={sm['pca_explained_variance']:.3f} steps={sm.get('steps')}")
    for row in search["search"]:
        print(
            f"  k={row['k']} silhouette={row['silhouette']:.4f} "
            f"db={row['davies_bouldin']:.4f} inertia={row['inertia']:.1f}"
        )
    print(f"[kmeans] wrote {KMEANS_PATH}")
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means trên vector JD")
    parser.add_argument("--vectors", type=Path, default=JOB_VECTORS)
    parser.add_argument("--jobs", type=Path, default=JOBS_FEATURED)
    args = parser.parse_args()
    run_kmeans(args.vectors, args.jobs)


if __name__ == "__main__":
    main()
