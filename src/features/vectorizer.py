"""Weighted feature matrix: skill TF-IDF + SVD text + years/level.

Skill block ~0.80, text SVD ~0.10, experience/level ~0.10.
Vectors are L2-normalized so Euclidean K-Means ≈ cosine / spherical K-Means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

from src.features.corpus_filter import filter_it_corpus
from src.features.skills import SkillExtractor
from src.features.stopwords import VI_EN_STOPWORDS
from src.paths import load_config

QUERY_STOP_SKILLS = {"git", "html", "css"}


def _as_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


@dataclass
class FeaturePipeline:
    extractor: SkillExtractor = field(default_factory=SkillExtractor)
    skill_vectorizer: TfidfVectorizer | None = None
    text_vectorizer: TfidfVectorizer | None = None
    svd: TruncatedSVD | None = None
    struct_scaler: MinMaxScaler | None = None
    family_names: list[str] = field(default_factory=list)
    cfg: dict = field(default_factory=dict)
    n_text_dim: int = 0
    skill_dim: int = 0
    text_dim: int = 0
    struct_dim: int = 0
    corpus_filter_stats: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self.cfg = load_config()["features"]
        self.family_names = self.extractor.taxonomy.families()
        skills_col = []
        must_col = []
        nice_col = []
        for _, row in df.iterrows():
            tags = list(row.get("skills_raw") or [])
            must = self.extractor.extract(
                f"{row.get('title','')} {row.get('requirements_text','')}",
                extra_tags=tags,
            )
            nice = self.extractor.extract(str(row.get("nice_to_have_text") or ""))
            union = list(dict.fromkeys(must + nice))
            if not union:
                union = must or self.extractor.extract(str(row.get("display_text") or ""), extra_tags=tags)
            skills_col.append(union)
            must_col.append(must or union)
            nice_col.append(nice)

        df["skills_norm"] = skills_col
        df["skills_must"] = must_col
        df["skills_nice"] = nice_col
        kept, stats = filter_it_corpus(df, self.extractor.taxonomy)
        self.corpus_filter_stats = stats
        df.drop(index=df.index.difference(kept.index), inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(
            f"[features] corpus filter {stats['n_before']}→{stats['n_after']} "
            f"(drop {stats['n_dropped']}: title={stats['drop_reasons'].get('non_it_title', 0)}, "
            f"few_skills={stats['drop_reasons'].get('too_few_it_skills', 0)})"
        )
        if df.empty:
            raise RuntimeError("Không còn JD nào sau khi lọc corpus IT.")

        must_docs = [" ".join(s or []) for s in df["skills_norm"]]
        req_docs = [
            str(row.get("requirements_text") or row.get("display_text") or "")
            for _, row in df.iterrows()
        ]

        self.skill_vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s]+",
            min_df=1,
            ngram_range=(1, 1),
        )
        self.skill_vectorizer.fit(must_docs)

        max_feat = int(self.cfg["tfidf_max_features"])
        n_svd = int(self.cfg["svd_components"])
        self.text_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(VI_EN_STOPWORDS),
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_feat,
        )
        try:
            text_mat = self.text_vectorizer.fit_transform(req_docs)
            if text_mat.shape[1] < 3:
                raise ValueError("too few terms")
        except ValueError:
            self.text_vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=max_feat,
            )
            text_mat = self.text_vectorizer.fit_transform(req_docs)
        n_comp = max(1, min(n_svd, max(1, text_mat.shape[0] - 1), max(1, text_mat.shape[1] - 1)))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(text_mat)
        self.n_text_dim = n_comp

        struct = df[["years_mid", "level_ordinal"]].copy()
        struct["years_mid"] = struct["years_mid"].fillna(2.0).clip(0.0, 12.0)
        struct["level_ordinal"] = struct["level_ordinal"].fillna(2)
        struct = struct.to_numpy(dtype=float)
        self.struct_scaler = MinMaxScaler()
        self.struct_scaler.fit(struct)
        return self

    def transform(self, df: pd.DataFrame, fit_skills: bool = False) -> np.ndarray:
        if self.skill_vectorizer is None:
            raise RuntimeError("FeaturePipeline is not fitted")
        skill_docs = []
        req_docs = []
        skills_col = []
        must_col = []
        nice_col = []
        for _, row in df.iterrows():
            if row.get("skills_norm"):
                union = list(row["skills_norm"])
                must = list(row.get("skills_must") or union)
                nice = list(row.get("skills_nice") or [])
            else:
                tags = list(row.get("skills_raw") or [])
                must = self.extractor.extract(
                    f"{row.get('title','')} {row.get('requirements_text','') or row.get('display_text','')}",
                    extra_tags=tags,
                )
                nice = self.extractor.extract(str(row.get("nice_to_have_text") or ""))
                union = list(dict.fromkeys(must + nice)) or must
            skills_col.append(union)
            must_col.append(must or union)
            nice_col.append(nice)
            skill_docs.append(" ".join(union))
            req_docs.append(str(row.get("requirements_text") or row.get("display_text") or " ".join(union)))

        df["skills_norm"] = skills_col
        df["skills_must"] = must_col
        df["skills_nice"] = nice_col

        skill_tfidf = _as_dense(self.skill_vectorizer.transform(skill_docs))
        text_svd = _as_dense(self.svd.transform(self.text_vectorizer.transform(req_docs)))
        struct = df[["years_mid", "level_ordinal"]].copy()
        struct["years_mid"] = struct["years_mid"].fillna(2.0).clip(0.0, 12.0)
        struct["level_ordinal"] = struct["level_ordinal"].fillna(2)
        struct = self.struct_scaler.transform(struct.to_numpy(dtype=float))

        sw = float(self.cfg["skill_weight"])
        tw = float(self.cfg["text_weight"])
        stw = float(self.cfg["structured_weight"])
        skill_block = _l2_rows(skill_tfidf) * sw
        text_svd = _l2_rows(text_svd) * tw
        struct = _l2_rows(struct) * stw
        self.skill_dim = skill_block.shape[1]
        self.text_dim = text_svd.shape[1]
        self.struct_dim = struct.shape[1]
        x = np.hstack([skill_block, text_svd, struct]).astype(np.float32)
        return _l2_rows(x)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def transform_user(
        self,
        skills: list[str],
        years: float,
        level_ordinal: int,
        extra_text: str = "",
    ) -> tuple[np.ndarray, list[str]]:
        if self.skill_vectorizer is None:
            raise RuntimeError("FeaturePipeline is not fitted")
        canon = []
        seen = set()
        for s in skills:
            extracted = self.extractor.extract(s)
            if extracted:
                for c in extracted:
                    if c not in seen:
                        seen.add(c)
                        canon.append(c)
            else:
                key = s.strip().lower().replace(" ", "_")
                if key in self.extractor.taxonomy.canonical_to_family and key not in seen:
                    seen.add(key)
                    canon.append(key)
                else:
                    mapped = self.extractor.taxonomy.alias_to_canonical.get(s.strip().lower())
                    if mapped and mapped not in seen:
                        seen.add(mapped)
                        canon.append(mapped)
        if extra_text:
            for c in self.extractor.extract(extra_text):
                if c not in seen:
                    seen.add(c)
                    canon.append(c)
        query_skills = [s for s in canon if s not in QUERY_STOP_SKILLS] or list(canon)
        tmp = pd.DataFrame(
            [
                {
                    "skills_norm": query_skills,
                    "skills_must": query_skills,
                    "skills_nice": [],
                    "skills_raw": query_skills,
                    "requirements_text": extra_text or " ".join(query_skills),
                    "display_text": extra_text or " ".join(query_skills),
                    "nice_to_have_text": "",
                    "title": "",
                    "years_mid": float(years),
                    "level_ordinal": int(level_ordinal),
                }
            ]
        )
        vec = self.transform(tmp)
        if not (extra_text or "").strip() and self.text_dim:
            s = self.skill_dim
            vec = vec.copy()
            vec[:, s : s + self.text_dim] = 0.0
            vec = _l2_rows(vec)
        return vec, canon


def _l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return normalize(x, norm="l2", axis=1)
