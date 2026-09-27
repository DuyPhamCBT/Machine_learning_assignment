"""Streamlit demo: upload CV → gợi ý việc IT."""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx


def _ensure_streamlit_server() -> None:
    """`uv run app/streamlit_app.py` không mở browser — phải đi qua `streamlit run`."""
    if get_script_run_ctx() is not None:
        return
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(stcli.main())


_ensure_streamlit_server()

from app.artifacts import load_artifacts
from src.cv_parse.parser import parse_cv
from src.evaluation.metrics import PERSONAS
from src.paths import SAMPLES_DIR, artifacts_ready
from src.recommend.knn import embed_user, recommend, suitable_max_level_gap

st.set_page_config(
    page_title="IT Match · Gợi ý việc làm",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEVEL_OPTIONS = {
    "Intern": 0,
    "Fresher": 1,
    "Junior": 2,
    "Middle": 3,
    "Senior": 4,
    "Lead": 5,
}
LEVEL_FROM_ORDINAL = {v: k for k, v in LEVEL_OPTIONS.items()}
LEVEL_NAMES = list(LEVEL_OPTIONS.keys())
PERSONA_LEVEL = {"intern": "Intern", "fresher": "Fresher", "middle": "Middle", "senior": "Senior", "lead": "Lead"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
html, body, [class*="stApp"] { font-family: "DM Sans", "Segoe UI", sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stSidebar"] {
  background: #F0FDFA;
  border-right: 1px solid #99F6E4;
}
.hero {
  background: linear-gradient(135deg, #0F766E 0%, #0E7490 55%, #1E3A5F 100%);
  color: #fff; border-radius: 18px; padding: 1.35rem 1.55rem 1.2rem;
  margin-bottom: 1.1rem; box-shadow: 0 12px 32px rgba(15, 118, 110, .22);
}
.hero h1 { color: #fff !important; font-size: 1.85rem; margin: 0 0 .35rem; letter-spacing: -.02em; }
.hero p { color: rgba(255,255,255,.86); margin: 0; font-size: .98rem; }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; opacity: .8; margin-bottom: .35rem; }
.step-row { display: flex; gap: .6rem; margin: 0 0 1rem; flex-wrap: wrap; }
.step { background: #fff; border: 1px solid #E2E8F0; border-radius: 999px; padding: .28rem .8rem; font-size: .82rem; color: #475569; }
.step b { color: #0F766E; }
.chip-row { display: flex; flex-wrap: wrap; gap: .35rem; margin: .25rem 0 .4rem; }
.chip {
  display: inline-block; padding: .15rem .55rem; border-radius: 999px;
  background: #ECFDF5; color: #115E59; font-size: .78rem; border: 1px solid #99F6E4;
}
.chip.miss { background: #FFF7ED; color: #9A3412; border-color: #FED7AA; }
.chip.level { background: #F1F5F9; color: #334155; border-color: #E2E8F0; font-weight: 600; }
.chip.level-intern, .chip.level-fresher { background: #CCFBF1; color: #0F766E; border-color: #5EEAD4; }
.chip.level-junior { background: #DBEAFE; color: #1D4ED8; border-color: #93C5FD; }
.chip.level-middle { background: #E0E7FF; color: #4338CA; border-color: #C7D2FE; }
.chip.level-senior { background: #EDE9FE; color: #6D28D9; border-color: #DDD6FE; }
.chip.level-lead { background: #FFE4E6; color: #BE123C; border-color: #FECDD3; }
.section-banner {
  border-radius: 14px; padding: .85rem 1rem; margin: .4rem 0 .85rem;
}
.section-banner.wide { background: #EFF6FF; border: 1px solid #BFDBFE; }
.section-banner.tight { background: #ECFDF5; border: 1px solid #99F6E4; }
.section-banner h3 { margin: 0 0 .2rem; font-size: 1.05rem; }
.section-banner p { margin: 0; color: #475569; font-size: .88rem; }
.job-meta { color: #64748B; font-size: .86rem; margin-top: .1rem; }
.score-bar { height: 7px; background: #E2E8F0; border-radius: 99px; overflow: hidden; margin: .45rem 0 .2rem; }
.score-bar > span { display: block; height: 100%; background: linear-gradient(90deg, #14B8A6, #0F766E); }
.muted { color: #64748B; font-size: .86rem; }
.empty-box {
  text-align: center; padding: 1.4rem 1rem; border: 1px dashed #CBD5E1;
  border-radius: 14px; background: #F8FAFC; color: #64748B;
}
div[data-testid="stMetric"] { background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: .4rem .6rem; }
</style>
"""


@st.cache_resource
def _load():
    if not artifacts_ready():
        return None
    return load_artifacts()


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _fmt_salary(row) -> str:
    raw = row.get("salary_raw") or ""
    if raw:
        return str(raw)
    mid = row.get("salary_mid_vnd")
    if pd.isna(mid) or not mid:
        return "Thương lượng"
    return f"{mid / 1_000_000:.0f} triệu VND"


def _allowed_levels(user_ord: int, gap: int) -> str:
    return " · ".join(name for i, name in enumerate(LEVEL_NAMES) if abs(i - int(user_ord)) <= int(gap))


def _canon_persona_skills(raw: list[str], all_skills: list[str]) -> list[str]:
    allowed = set(all_skills)
    out: list[str] = []
    for skill in raw:
        for cand in (skill, skill.replace("/", "_"), skill.replace("-", "_")):
            if cand in allowed and cand not in out:
                out.append(cand)
                break
    return out


def _chips(items: list, kind: str = "") -> str:
    parts = []
    for item in items[:14]:
        label = html.escape(str(item).replace("_", " "))
        cls = f"chip {kind}".strip()
        parts.append(f'<span class="{cls}">{label}</span>')
    return f'<div class="chip-row">{"".join(parts) or "<span class=muted>—</span>"}</div>'


def _level_chip(level: str) -> str:
    key = html.escape((level or "junior").lower())
    label = html.escape((level or "—").title())
    return f'<span class="chip level level-{key}">{label}</span>'


def _apply_parsed_to_form(parsed: dict, all_skills: list[str]) -> None:
    skills = [s for s in (parsed.get("skills") or []) if s in all_skills][:40]
    st.session_state["skills_sel"] = skills
    years_mid = parsed.get("years_mid")
    st.session_state["years"] = float(years_mid) if years_mid is not None else 0.0
    ordinal = parsed.get("level_ordinal")
    ordinal = 2 if ordinal is None else int(ordinal)
    st.session_state["level_name"] = LEVEL_FROM_ORDINAL.get(ordinal, "Junior")
    st.session_state["parsed"] = parsed
    st.session_state.pop("results", None)


def _drop_reason(row, user_level: int, gap_max: int, location: str, remote_only: bool) -> str:
    if row.get("level_gap") is not None:
        gap = int(row["level_gap"])
    else:
        job_ord = row.get("level_ordinal")
        job_ord = 2 if job_ord is None else int(job_ord)
        gap = abs(int(user_level) - job_ord)
    if gap > gap_max:
        return f"Lệch {gap} cấp (max {gap_max})"
    loc = row.get("location_norm")
    is_remote = bool(row.get("is_remote")) or loc == "Remote"
    if location not in {None, "Tất cả", "All"} and loc not in {location, "Remote"} and not is_remote:
        return "Khác địa điểm đã chọn"
    if remote_only and not is_remote:
        return "Không phải remote"
    return "Không lọt top sau khi siết cấp"


def _job_card(row, accent: str = "wide") -> None:
    title = html.escape(str(row.get("title") or "—"))
    company = html.escape(str(row.get("company") or "—"))
    loc = html.escape(str(row.get("location_norm") or "—"))
    family = html.escape(str(row.get("role_family") or "—"))
    cosine = float(row.get("cosine") or 0)
    score = float(row.get("score") or 0)
    gap = int(row.get("level_gap") or 0)
    width = max(4.0, min(100.0, cosine * 100))
    with st.container(border=True):
        left, right = st.columns([4.2, 1.15])
        with left:
            st.markdown(
                f"<b>{int(row['rank'])}. {title}</b>"
                f'<div class="job-meta">{company} · {loc} · {family}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _level_chip(str(row.get("level_norm") or ""))
                + f'<span class="chip">{"đúng cấp" if gap == 0 else f"lệch {gap} cấp"}</span>'
                + f'<span class="chip">{html.escape(_fmt_salary(row))}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="score-bar"><span style="width:{width:.1f}%"></span></div>'
                f'<span class="muted">Cosine {cosine:.3f} · điểm khớp {score:.3f} · coverage {float(row.get("skill_coverage") or 0):.0%}</span>',
                unsafe_allow_html=True,
            )
        with right:
            st.metric("Liên quan", f"{cosine:.0%}")
            url = row.get("source_url")
            if url and str(url).lower() not in {"nan", "none", ""}:
                st.link_button("Mở JD", str(url), use_container_width=True)
        ov = list(row.get("overlap_skills") or [])
        miss = list(row.get("missing_skills") or [])
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Kỹ năng trùng")
            st.markdown(_chips(ov), unsafe_allow_html=True)
        with c2:
            st.caption("Bạn còn thiếu")
            st.markdown(_chips(miss, "miss"), unsafe_allow_html=True)


def _render_job_list(recs: pd.DataFrame, empty_msg: str, accent: str) -> None:
    if recs is None or recs.empty:
        st.markdown(f'<div class="empty-box">{html.escape(empty_msg)}</div>', unsafe_allow_html=True)
        return
    for _, row in recs.iterrows():
        _job_card(row, accent=accent)


def page_recommend(art: dict) -> None:
    df = art["df"]
    pipe = art["pipeline"]
    knn = art["knn"]
    kmeans = art["kmeans"]
    meta = art["meta"]
    space = art["space"]
    taxonomy = pipe.extractor.taxonomy
    ok_fams = set(taxonomy.families())
    all_skills = [s for s in taxonomy.all_canonical() if taxonomy.canonical_to_family.get(s) in ok_fams]
    family_of = taxonomy.canonical_to_family
    n_jobs = meta.get("n_jobs", len(df))
    best_k = meta.get("best_k", "—")

    pending = st.session_state.pop("pending_persona", None)
    if pending:
        st.session_state["skills_sel"] = _canon_persona_skills(pending["skills"], all_skills)
        st.session_state["years"] = float(pending["years"])
        st.session_state["level_name"] = PERSONA_LEVEL.get(pending["level"], "Junior")
        st.session_state["parsed"] = None
        st.session_state["auto_run"] = True
        st.session_state.pop("results", None)

    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">IT Match · demo</div>
          <h1>Gợi ý việc làm IT từ CV</h1>
          <p>Upload PDF / DOCX / TXT → rút skill · KNN Cosine trên {n_jobs} JD · {best_k} cụm K-Means.</p>
        </div>
        <div class="step-row">
          <span class="step"><b>1</b> Tải CV</span>
          <span class="step"><b>2</b> Chỉnh hồ sơ</span>
          <span class="step"><b>3</b> Top 10 liên quan</span>
          <span class="step"><b>4</b> Việc thích hợp theo cấp</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Tải CV")
    up_col, sample_col = st.columns([2.6, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Chọn file CV",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            help="PDF, DOCX hoặc TXT. Hệ thống chỉ đọc text, không gửi file đi đâu khác.",
        )
    with sample_col:
        st.write("")
        use_sample = st.button("Dùng CV mẫu · Intern Python", use_container_width=True)

    if use_sample:
        sample_path = SAMPLES_DIR / "cv_fresher_python.txt"
        parsed = parse_cv(sample_path.name, sample_path.read_bytes(), extractor=pipe.extractor)
        _apply_parsed_to_form(parsed, all_skills)
        st.session_state["upload_sig"] = ("sample", 0)
        st.session_state["flash"] = (
            f"Đã đọc CV mẫu · {len(parsed['skills'])} kỹ năng · "
            f"cấp {LEVEL_FROM_ORDINAL.get(int(parsed['level_ordinal']), '?')}"
        )
        st.rerun()
    elif uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_sig") != sig:
            try:
                parsed = parse_cv(uploaded.name, uploaded.getvalue(), extractor=pipe.extractor)
                _apply_parsed_to_form(parsed, all_skills)
                st.session_state["upload_sig"] = sig
            except Exception as exc:
                st.error(f"Không đọc được CV: {exc}")
        parsed = st.session_state.get("parsed")
        if parsed:
            ord_ = parsed.get("level_ordinal")
            ord_ = 2 if ord_ is None else int(ord_)
            st.success(
                f"Đã đọc **{uploaded.name}** · {len(parsed['skills'])} kỹ năng · "
                f"{parsed.get('years_mid', 0):.1f} năm · {LEVEL_FROM_ORDINAL.get(ord_, 'Junior')}"
            )
            with st.expander("Xem text đã trích (2.500 ký tự đầu)"):
                st.text((parsed.get("text") or "")[:2500])

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    parsed = st.session_state.get("parsed")

    st.subheader("Hồ sơ dùng để match")
    st.caption("CV chỉ điền sẵn. Bạn có thể thêm/bớt skill trước khi gợi ý — form là nguồn sự thật.")
    if "skills_sel" not in st.session_state:
        st.session_state["skills_sel"] = []
    if "years" not in st.session_state:
        st.session_state["years"] = 2.0
    if "level_name" not in st.session_state:
        st.session_state["level_name"] = "Junior"

    skills = st.multiselect(
        "Kỹ năng",
        options=all_skills,
        key="skills_sel",
        format_func=lambda s: f"{s.replace('_', ' ')} ({family_of.get(s, '')})",
    )
    if skills:
        st.markdown(_chips(skills), unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        years = st.slider("Số năm kinh nghiệm", min_value=0.0, max_value=12.0, step=0.25, key="years")
    with f2:
        level_name = st.selectbox("Cấp bậc", list(LEVEL_OPTIONS.keys()), key="level_name")

    st.markdown("**Persona có sẵn**")
    cols = st.columns(3)
    for col, persona in zip(cols, PERSONAS):
        with col:
            with st.container(border=True):
                st.markdown(f"**{persona['name']}**")
                st.caption(persona["note"])
                if st.button("Chạy persona", key=f"p_{persona['name']}", use_container_width=True):
                    st.session_state["pending_persona"] = persona
                    st.rerun()

    go = st.button("Tìm việc phù hợp", type="primary", use_container_width=True)
    go = go or bool(st.session_state.pop("auto_run", False))

    location = st.session_state.get("flt_location", "Tất cả")
    remote_only = bool(st.session_state.get("flt_remote", False))

    if go:
        if not skills:
            st.warning("Upload CV hoặc chọn ít nhất một kỹ năng.")
        else:
            level_ord = LEVEL_OPTIONS[level_name]
            extra = (parsed or {}).get("text") or ""
            vec, canon = embed_user(pipe, space, skills, years, level_ord, extra)
            gap = suitable_max_level_gap(level_ord)
            wide = recommend(
                user_vec=vec[0],
                user_skills=canon,
                user_years=years,
                user_level=level_ord,
                df=df,
                knn=knn,
                kmeans=kmeans,
                top_n=10,
                max_level_gap=None,
                apply_filters=False,
                rank_by="cosine",
            )
            tight = recommend(
                user_vec=vec[0],
                user_skills=canon,
                user_years=years,
                user_level=level_ord,
                df=df,
                knn=knn,
                kmeans=kmeans,
                top_n=10,
                location=location,
                remote_only=remote_only,
                max_level_gap=gap,
                apply_filters=True,
                rank_by="score",
                min_skill_overlap=1,
            )
            user_cid = int(kmeans.predict(vec)[0])
            profiles = meta.get("cluster_profiles", {})
            cname = profiles.get(str(user_cid), {}).get("name", f"Cụm {user_cid}")
            dropped = pd.DataFrame()
            if not wide.empty:
                tight_ids = set(tight["job_id"].tolist()) if not tight.empty else set()
                dropped = wide[~wide["job_id"].isin(tight_ids)].copy()
                if not dropped.empty:
                    dropped["ly_do"] = dropped.apply(
                        lambda r: _drop_reason(r, level_ord, gap, location, remote_only), axis=1
                    )
            st.session_state["results"] = {
                "canon": canon,
                "cluster_name": cname,
                "level_name": level_name,
                "level_ord": level_ord,
                "years": years,
                "gap": gap,
                "wide": wide,
                "tight": tight,
                "dropped": dropped,
                "location": location,
                "remote_only": remote_only,
                "form_sig": (tuple(skills), float(years), level_name, location, remote_only),
            }

    results = st.session_state.get("results")
    if not results:
        st.markdown(
            '<div class="empty-box">Chưa có kết quả. Tải CV hoặc chọn persona, rồi bấm <b>Tìm việc phù hợp</b>.</div>',
            unsafe_allow_html=True,
        )
        return

    current_sig = (tuple(skills), float(years), level_name, location, remote_only)
    if results.get("form_sig") != current_sig:
        st.info("Hồ sơ hoặc bộ lọc đã đổi so với kết quả đang hiện. Bấm **Tìm việc phù hợp** để chạy lại.")

    st.divider()
    allowed = _allowed_levels(results["level_ord"], results["gap"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cụm gần nhất", results["cluster_name"].split(":")[0])
    m2.metric("Skill dùng match", len(results["canon"]))
    m3.metric("Cấp hồ sơ", results["level_name"])
    m4.metric("max_level_gap", results["gap"])
    st.caption(f"Kỹ năng: {', '.join(results['canon'][:18]) or '—'}  ·  Việc thích hợp gồm cấp: {allowed or '—'}")

    st.markdown(
        """
        <div class="section-banner wide">
          <h3>1. Top 10 liên quan nhất — không lọc điều kiện</h3>
          <p>Xếp theo cosine KNN thuần. Không lọc cấp bậc, địa điểm hay remote — để thấy việc “gần skill” nhất, kể cả lệch senior/lead.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_job_list(
        results["wide"],
        "Không tìm được JD lân cận trong không gian vector.",
        "wide",
    )

    gap = results["gap"]
    intern_note = (
        f"Hồ sơ {results['level_name']} chỉ giữ JD lệch tối đa <b>1 cấp</b> ({allowed}). "
        if results["level_ord"] <= 1
        else f"Hồ sơ {results['level_name']} giữ JD lệch tối đa <b>2 cấp</b> ({allowed}). "
    )
    loc_note = "Có lọc địa điểm / remote từ thanh bên." if results["location"] != "Tất cả" or results["remote_only"] else "Chưa lọc địa điểm."
    st.markdown(
        f"""
        <div class="section-banner tight">
          <h3>2. Việc thích hợp nhất — siết cấp bậc</h3>
          <p>{intern_note}{loc_note} Xếp theo điểm khớp (cosine × cấp × năm × coverage).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_job_list(
        results["tight"],
        "Không còn JD trong khoảng cấp cho phép. Nới skill hoặc xem danh sách liên quan phía trên.",
        "tight",
    )

    dropped = results.get("dropped")
    if dropped is not None and not dropped.empty:
        with st.expander(f"Việc trong Top 10 liên quan nhưng không vào danh sách thích hợp ({len(dropped)})"):
            view = dropped[["rank", "title", "level_norm", "level_gap", "ly_do", "cosine"]].copy()
            view.columns = ["#", "Title", "Cấp JD", "Lệch cấp", "Lý do loại", "Cosine"]
            st.dataframe(view, use_container_width=True, hide_index=True)


def page_clusters(art: dict) -> None:
    df = art["df"]
    meta = art["meta"]
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">K-Means · PCA 2D</div>
          <h1>Khám phá cụm việc làm</h1>
          <p>Mỗi cụm ≈ một nhóm kỹ năng cốt lõi. Dùng để giải thích, không dùng để xếp hạng CV.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Số JD", meta.get("n_jobs", len(df)))
    k2.metric("Số cụm K", meta.get("best_k", "—"))
    k3.metric("PCA 50-d giữ", f"{(meta.get('space') or {}).get('pca_explained_variance', 0):.0%}")
    filt = meta.get("corpus_filter") or {}
    k4.metric("Đã lọc bỏ", filt.get("n_dropped", "—"))
    if filt:
        st.caption(
            f"Corpus IT: {filt.get('n_before')} → {filt.get('n_after')} JD "
            f"(title non-IT {filt.get('drop_reasons', {}).get('non_it_title', 0)}, "
            f"ít skill {filt.get('drop_reasons', {}).get('too_few_it_skills', 0)}). "
            "Space: PCA(50)+L2, không StandardScaler. Plot 2D chỉ để xem — K-Means chạy trên 50 chiều."
        )

    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="cluster_name",
        hover_data=["title", "company", "role_family", "level_norm"],
        opacity=0.72,
        height=540,
    )
    fig.update_layout(
        legend_title="Cụm",
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,.8)",
        font=dict(family="DM Sans, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    profiles = list(meta.get("cluster_profiles", {}).values())
    if profiles:
        table = pd.DataFrame(profiles)
        if "top_skills" in table.columns:
            table["top_skills"] = table["top_skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        keep = [c for c in ["name", "size", "role_family", "top_skills", "example_title", "mean_years"] if c in table.columns]
        st.dataframe(table[keep] if keep else table, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Role family")
        st.bar_chart(df["role_family"].value_counts())
    with right:
        st.subheader("Cấp bậc JD")
        st.bar_chart(df["level_norm"].value_counts())

    search = st.selectbox("JD mẫu trong cụm", sorted(df["cluster_name"].unique()))
    sample = df[df["cluster_name"] == search][["title", "company", "level_norm", "salary_raw", "skills_norm"]].head(12)
    sample = sample.copy()
    sample["skills_norm"] = sample["skills_norm"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    st.dataframe(sample, use_container_width=True, hide_index=True)


def main() -> None:
    _inject_css()
    art = _load()
    with st.sidebar:
        st.markdown("### IT Match")
        st.caption("Gợi ý JD IT từ CV · KNN Cosine")
        page = st.radio("Trang", ["Gợi ý từ CV", "Khám phá cụm"], label_visibility="collapsed")
        st.divider()
        if art is None:
            st.warning("Chưa có model.")
        else:
            meta = art["meta"]
            st.metric("JD", meta.get("n_jobs", len(art["df"])))
            st.metric("Cụm K", meta.get("best_k", "—"))
            sources = meta.get("sources") or {}
            st.caption(" · ".join(f"{k} {v}" for k, v in sources.items()) or "—")
            if page == "Gợi ý từ CV":
                st.divider()
                st.markdown("**Lọc danh sách thích hợp**")
                st.caption("Chỉ áp dụng cho phần 2. Top 10 liên quan bỏ qua lọc này.")
                locations = ["Tất cả"] + sorted(art["df"]["location_norm"].dropna().unique().tolist())
                st.selectbox("Địa điểm", locations, key="flt_location")
                st.checkbox("Chỉ remote", key="flt_remote")
                st.caption("Intern / Fresher: max_level_gap = 1. Cấp khác: 2.")
    if art is None:
        st.error(
            "Chưa có model. Trong folder `IT_Job_Recommendation` chạy:\n\n"
            "`python scripts/run_all.py`\n\n"
            "hoặc notebook `notebooks/01_full_pipeline.ipynb`."
        )
        return
    if page == "Gợi ý từ CV":
        page_recommend(art)
    else:
        page_clusters(art)


if __name__ == "__main__":
    main()
