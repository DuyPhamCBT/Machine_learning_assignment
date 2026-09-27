"""Vẽ PCA 2D trước/sau lọc corpus + bỏ scaler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.paths import JOBS_PKL, META_PATH, REPORTS_DIR


def _scatter(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="cluster_name",
        hover_data=["title", "role_family"],
        opacity=0.7,
        title=title,
        height=520,
    )
    fig.update_layout(legend_title="Cụm", margin=dict(l=0, r=0, t=40, b=0), legend=dict(font=dict(size=10)))
    return fig


def snapshot_before(df: pd.DataFrame | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    src = df if df is not None else pd.read_pickle(JOBS_PKL)
    slim = src[["title", "role_family", "cluster_name", "pca_x", "pca_y"]].copy()
    path = REPORTS_DIR / "pca_before_filter.csv"
    slim.to_csv(path, index=False, encoding="utf-8-sig")
    fig = _scatter(slim, f"Trước lọc · {len(slim)} JD · StandardScaler + PCA(50)")
    fig.write_html(REPORTS_DIR / "pca_before_filter.html", include_plotlyjs="cdn")
    return path


def write_after_and_compare() -> dict:
    after = pd.read_pickle(JOBS_PKL)
    before_csv = REPORTS_DIR / "pca_before_filter.csv"
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    fig_after = _scatter(after, f"Sau lọc · {len(after)} JD · PCA(50)+L2, không scaler")
    fig_after.write_html(REPORTS_DIR / "pca_after_filter.html", include_plotlyjs="cdn")

    report = {
        "n_jobs_after": int(len(after)),
        "best_k": meta.get("best_k"),
        "silhouette": (meta.get("k_search") or [{}])[-1] if False else None,
        "space": meta.get("space"),
        "corpus_filter": meta.get("corpus_filter"),
        "cluster_sizes": {str(k): int(v) for k, v in after["cluster_id"].value_counts().sort_index().items()}
        if "cluster_id" in after.columns
        else {},
        "cluster_names": after.drop_duplicates("cluster_id")
        .sort_values("cluster_id")["cluster_name"]
        .tolist()
        if "cluster_id" in after.columns
        else [],
    }
    k_search = meta.get("k_search") or []
    best = meta.get("best_k")
    for row in k_search:
        if row.get("k") == best:
            report["internal"] = {
                "silhouette": row.get("silhouette"),
                "davies_bouldin": row.get("davies_bouldin"),
                "k": best,
            }
            break

    if before_csv.exists():
        before = pd.read_csv(before_csv)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Trước · {len(before)} JD · scaler+PCA",
                f"Sau · {len(after)} JD · no scaler + lọc IT",
            ),
            horizontal_spacing=0.08,
        )
        for col, frame in ((1, before), (2, after)):
            for name, part in frame.groupby("cluster_name"):
                fig.add_trace(
                    go.Scatter(
                        x=part["pca_x"],
                        y=part["pca_y"],
                        mode="markers",
                        name=str(name)[:40],
                        marker=dict(size=5, opacity=0.7),
                        legendgroup=str(col),
                        showlegend=False,
                        hovertext=part["title"] if "title" in part.columns else None,
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(title_text="pca_x", row=1, col=1)
        fig.update_xaxes(title_text="pca_x", row=1, col=2)
        fig.update_yaxes(title_text="pca_y", row=1, col=1)
        fig.update_yaxes(title_text="pca_y", row=1, col=2)
        fig.update_layout(height=560, title="PCA 2D: trước vs sau lọc corpus + bỏ StandardScaler")
        fig.write_html(REPORTS_DIR / "pca_compare_filter.html", include_plotlyjs="cdn")
        report["n_jobs_before"] = int(len(before))

    (REPORTS_DIR / "cluster_compare_filter.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
