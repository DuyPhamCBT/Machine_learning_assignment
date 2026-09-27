"""Optional StandardScaler / PCA / L2 space used before K-Means and KNN.

Feature vectors are already row-L2 in IT_Job_Features. Extra scaling + PCA
can drop noise in the 680-d skill TF-IDF block. The fitted object is a
sklearn Pipeline so App / Evaluate can `joblib.load` without this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


def build_space(
    *,
    scaler: str | None = "standard",
    pca_components: int | float | None = 50,
    l2: bool = True,
    random_state: int = 42,
) -> Pipeline | None:
    """Return sklearn Pipeline, or None for identity (cluster on raw vectors)."""
    steps: list[tuple[str, Any]] = []
    kind = (scaler or "none").lower().strip()
    if kind in {"standard", "std", "zscore"}:
        steps.append(("scaler", StandardScaler()))
    elif kind in {"minmax", "min_max"}:
        steps.append(("scaler", MinMaxScaler()))
    elif kind not in {"", "none", "identity"}:
        raise ValueError(f"Unknown scaler: {scaler}")

    if pca_components is not None:
        pca_kw: dict[str, Any] = {"random_state": random_state}
        if isinstance(pca_components, float) and pca_components < 1:
            pca_kw["svd_solver"] = "full"
        steps.append(("pca", PCA(n_components=pca_components, **pca_kw)))
    if l2:
        steps.append(("l2", Normalizer(norm="l2")))
    if not steps:
        return None
    return Pipeline(steps)


def fit_transform_space(x: np.ndarray, space: Pipeline | None) -> np.ndarray:
    if space is None:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(space.fit_transform(x), dtype=np.float64)


def transform_space(x: np.ndarray, space: Pipeline | None) -> np.ndarray:
    if space is None:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(space.transform(x), dtype=np.float64)


def space_from_config(cfg: dict[str, Any]) -> Pipeline | None:
    space_cfg = cfg.get("space") or {}
    pca_raw = space_cfg.get("pca_components")
    pca_components: int | float | None
    if pca_raw in (None, "", False):
        pca_components = None
    else:
        pca_components = pca_raw
    return build_space(
        scaler=space_cfg.get("scaler", "none"),
        pca_components=pca_components,
        l2=bool(space_cfg.get("l2", False)),
        random_state=int((cfg.get("kmeans") or {}).get("random_state") or 42),
    )


def space_meta(space: Pipeline | None, x_reduced: np.ndarray) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "n_features_out": int(x_reduced.shape[1]),
        "identity": space is None,
        "steps": [] if space is None else [name for name, _ in space.steps],
    }
    if space is None or "pca" not in space.named_steps:
        return meta
    pca: PCA = space.named_steps["pca"]
    evr = pca.explained_variance_ratio_
    meta["pca_n_components"] = int(pca.n_components_)
    meta["pca_explained_variance"] = float(evr.sum())
    return meta
