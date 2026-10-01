# Gợi ý việc làm IT từ CV

Pipeline **không giám sát**: cào JD (ITviec + TopCV) → làm sạch → rút skill theo từ điển → **lọc corpus IT** → vector TF-IDF/SVD → **PCA + L2** → **K-Means** gom cụm → **KNN Cosine** xếp hạng. Demo Streamlit nhận CV (PDF / DOCX / TXT).

Không dùng LLM embedding / GPU. Cấu hình tập trung ở `config.yaml`. Repo này là bản chạy được (pipeline + Streamlit + model đã train), không kèm các folder thí nghiệm nội bộ.

**Lần train hiện tại** (xem `reports/evaluation.json`):

| | |
|---|---|
| JD thô → sạch → **IT dùng để train** | 1711 → 1699 → **1015** |
| Nguồn (sau lọc) | ITviec 538 · TopCV 477 |
| Vector | 686-d (skill 644 + text SVD 40 + năm/cấp 2) → PCA 50-d (~**49%** phương sai) |
| K-Means | **K = 13** (argmax silhouette 8…14) |
| Silhouette / Davies–Bouldin | 0.13 / 2.37 |
| Precision@10 (`role_family`, 40 query) | **0.46** |

Silhouette thấp hơn corpus cũ (0.24 trên 1699 JD) vì đã **bỏ cụm non-IT dễ tách** (design, sales, CNC, JD 0 skill). Cụm còn lại đọc được hơn (QA ~81% đúng family, AI/ML ~84%). Plot PCA 2D **không** đo chất lượng cụm — K-Means chạy trên 50 chiều.

---

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Luồng hệ thống](#2-luồng-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt](#4-cài-đặt)
5. [Chạy demo Streamlit](#5-chạy-demo-streamlit)
6. [Chạy full pipeline](#6-chạy-full-pipeline)
7. [Chi tiết từng bước](#7-chi-tiết-từng-bước)
8. [Hai danh sách gợi ý](#8-hai-danh-sách-gợi-ý)
9. [Công thức xếp hạng](#9-công-thức-xếp-hạng)
10. [Số liệu đánh giá](#10-số-liệu-đánh-giá)
11. [Giới hạn](#11-giới-hạn)
12. [Stack](#12-stack)

---

## 1. Bài toán

Ứng viên IT phải lướt hàng trăm JD trên ITviec / TopCV. Hệ thống:

- Biến **file CV** thành vector **cùng không gian** với JD đã train.
- Trả **hai danh sách**: Top 10 theo độ liên quan (cosine, không lọc cấp) và việc **thích hợp cấp bậc** (`max_level_gap`).
- Gom JD thành cụm skill (Java/Spring, QA, DevOps, …) để giải thích “nhóm nghề gần nhất”.

---

## 2. Luồng hệ thống

```
ITviec / TopCV
        │  Selenium + BeautifulSoup
        ▼
 data/raw/combined.jsonl                 (1) Crawl
        │
        ▼
 data/interim/jobs_clean.pkl             (2) Preprocess
        │  lương, năm KN (clip), cấp JD title-first, địa điểm, role_family
        ▼
 (3) Features
        │  SkillExtractor (regex alias, family)
        │  Lọc corpus IT  (bỏ sales / design / CNC / ít skill)
        │  TF-IDF skill 80% + SVD text 10% + năm/cấp 10% → L2
        ▼
 models/job_vectors.npy                  ~686-d
        │
        ▼
 models/space.joblib                     (4) PCA(50) + L2  — không StandardScaler
 models/kmeans.joblib                    K = argmax silhouette
        │
        ▼
 models/knn.joblib                       (5) KNN Cosine trên 50-d
 data/processed/jobs.pkl
        │
        ├────────────────────────────── (6) Evaluate
        ▼
 app/streamlit_app.py                    (7) Upload CV → 2 danh sách việc
```

App **không crawl live** và **không train lại**. Thiếu model: `python scripts/run_all.py`.

---

## 3. Cấu trúc thư mục

```
Machine_learning_assignment/
├── README.md
├── config.yaml                      # crawl, features, space, corpus, knn, cv_extract
├── requirements.txt
├── notebooks/01_full_pipeline.ipynb
├── app/
│   ├── streamlit_app.py             # demo CV
│   └── artifacts.py                 # load model + parse lại cấp JD
├── scripts/
│   ├── crawl.py
│   ├── preprocess.py
│   ├── build_features.py            # extract skill + lọc corpus + vector
│   ├── train_kmeans.py
│   ├── train_knn.py
│   ├── evaluate.py
│   ├── compare_pca.py               # plot PCA trước/sau lọc
│   └── run_all.py                   # preprocess → evaluate
├── src/
│   ├── paths.py
│   ├── levels.py                    # cấp JD (title-first) vs CV (intern-first)
│   ├── crawl/
│   ├── preprocess/
│   ├── features/
│   │   ├── skills.py                # taxonomy + alias_policy
│   │   ├── corpus_filter.py         # lọc JD IT
│   │   └── vectorizer.py
│   ├── clustering/                  # space + K-Means
│   ├── recommend/knn.py
│   ├── evaluation/
│   └── cv_parse/                    # PDF / DOCX (kèm bảng) / TXT; ưu tiên khối Skills
├── data/
│   ├── raw/combined.jsonl           # ~1711 dòng
│   ├── dictionaries/skills.yaml
│   ├── dictionaries/alias_policy.yaml  # siết alias rộng, gộp skill trùng
│   ├── samples/cv_fresher_python.txt
│   ├── interim/jobs_clean.pkl       # 1699 JD sạch
│   └── processed/jobs.pkl           # 1015 JD IT + cụm — app đọc file này
├── models/                          # pipeline, space, kmeans, knn, vectors
└── reports/                         # evaluation.json, PCA HTML so sánh
```

---

## 4. Cài đặt

Python **3.10–3.12**. Chrome chỉ cần khi **cào lại** JD. Demo không cần Chrome.

```powershell
cd Machine_learning_assignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
cd Machine_learning_assignment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Chạy demo Streamlit

Repo đã kèm model. Từ thư mục project:

```powershell
streamlit run app/streamlit_app.py
# hoặc:
uv run streamlit run app/streamlit_app.py
```

**Không** chạy `uv run app/streamlit_app.py` (thiếu `streamlit run` thì không mở browser). File app tự chuyển sang `streamlit run` nếu gõ nhầm.

1. **Gợi ý từ CV** — upload PDF/DOCX/TXT hoặc CV mẫu / persona.
2. Form skill · năm · cấp là nguồn sự thật (CV chỉ điền sẵn).
3. Sidebar: địa điểm / remote — **chỉ áp dụng danh sách thích hợp**.
4. Kết quả: (1) Top 10 liên quan, (2) việc thích hợp cấp, expander việc bị loại.
5. **Khám phá cụm** — scatter PCA 2D (chỉ để xem) + profile cụm.

Đổi `skills.yaml` / `alias_policy.yaml` / parser cấp: restart app (`artifacts.py` reload extractor). Đổi vector/KNN: phải train lại.

---

## 6. Chạy full pipeline

### Notebook

[`notebooks/01_full_pipeline.ipynb`](notebooks/01_full_pipeline.ipynb) — `SKIP_CRAWL = True` dùng `combined.jsonl` có sẵn.

### Script (đã có `combined.jsonl`)

```powershell
python scripts/run_all.py
```

Từng bước:

```powershell
python scripts/preprocess.py
python scripts/build_features.py      # lọc corpus + vector
python scripts/train_kmeans.py
python scripts/train_knn.py
python scripts/evaluate.py
streamlit run app/streamlit_app.py
```

Cào thêm (không bắt buộc; cần Chrome):

```powershell
python scripts/crawl.py --source both --max-pages 1 --max-jobs 5
```

Cào xong **train lại** từ `preprocess.py`.

---

## 7. Chi tiết từng bước

### (1) Crawl

Selenium headless, ITviec + TopCV. JSONL incremental theo URL. Không login, không vượt captcha.

### (2) Preprocess — 1711 → 1699

- Bỏ JD &lt; 80 ký tự; HTML; Unicode.
- Tách mô tả / yêu cầu / nice-to-have / quyền lợi (quyền lợi **không** vào TF-IDF).
- Lương → VND (USD × 26000). Lương **không** vào vector — chỉ hiện UI.
- Năm KN: regex; **chặn nhầm năm dương lịch** (`từ 2020` không còn thành 2020 năm); clip 0–15.
- **Cấp JD**: đọc **title trước** — `Trưởng nhóm` / Team lead → lead. Không dùng `alias in blob` (tránh `lead` ⊂ `leadership`).
- `role_family`: rule trên title — **nhãn yếu**, không phải target train.

### (3) Features + lọc corpus — 1699 → 1015

`data/dictionaries/skills.yaml`: canonical + alias, alias dài trước, biên Unicode (`java` ≠ `javascript`). `alias_policy.yaml` siết alias rộng (`cache`, `solid`, `s3` ≠ AWS, GitHub ≠ git) và gộp skill trùng (redis_cache → redis, minio_s3 → minio, ingress_nginx → nginx). Family hiển thị: language / frontend / backend / mobile / data / ai_ml / devops / cloud / qa / security. Bỏ `soft` / `domain` / `industrial`. Blocklist alias ngắn (`ui`, `cv`, `json`, …). Photoshop/Figma **không** tính là skill IT khi lọc.

Upload CV (`cv_extract` trong `config.yaml`): ưu tiên khối Technical Skills; bỏ intro/education; intern/fresher tối đa 3 skill từ project; thứ tự theo vị trí trên CV.

Giữ JD nếu:

- Title không phải sales / thiết kế / CNC / …, **và**
- ≥ 2 skill IT nếu `role_family` là Backend, QA, DevOps, …; **hoặc** ≥ 3 skill IT với các family khác.

Lần train này: drop **684** (267 title + 417 ít skill).

Mỗi JD còn lại một vector L2:

| Khối | Chiều (lần train) | Trọng số |
|---|---:|---:|
| TF-IDF skill canonical | 644 | 0.80 |
| TF-IDF 1–2 gram phần yêu cầu + SVD | 40 | 0.10 |
| `years_mid` (clip 0–12), `level_ordinal` (MinMax) | 2 | 0.10 |
| **Tổng** | **686** | L2 lần cuối |

Query CV: bỏ `git` / `html` / `css` khỏi khối skill khi còn skill khác. Không có text CV thì zero khối SVD.

**Cấp CV**: intern/fresher **trước** (sinh viên hay viết “lead a team” trong đồ án).

### (4) Không gian + K-Means

**Không** StandardScaler (z-score trên TF-IDF thưa làm PCA đuổi skill hiếm, JD IT bị nén một cụm).

686-d → **PCA(50)** → L2 (~49% phương sai). Euclidean trên L2 ≈ cosine. **K = argmax silhouette** trên 8…14 → **13**.

Tên cụm gán *sau* (mode `role_family` + top skill). Cụm **không lọc** KNN — chỉ hiện “nhóm nghề gần nhất”.

Cụm lần train này (ví dụ): Java/Spring, C#/.NET, Fullstack JS, AI/ML, ETL, QA, DevOps, Security, Mobile, C++/embedded, Sysadmin.

### (5) KNN

`NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)` trên 50-d. Query kéo tối đa ~200 (hoặc cả corpus khi siết intern/fresher).

### (6) Evaluate

Không có nhãn vàng từng JD:

- Nội tại: silhouette, Davies–Bouldin, size.
- Retrieval: hold-out Precision@10 trên `role_family`.
- 3 persona: Fresher Python, Mid React, Senior DevOps.

---

## 8. Hai danh sách gợi ý

| | 1. Top 10 liên quan | 2. Thích hợp cấp bậc |
|---|---|---|
| Lọc cấp | **Không** (`max_level_gap=None`) | Intern/Fresher: **gap ≤ 1**; cấp khác: gap ≤ 2 |
| Địa điểm / remote | Không | Có (sidebar) |
| Xếp hạng | Cosine thuần | `score` (công thức dưới) |
| Skill trùng | Không bắt buộc | ≥ 1 skill trùng |

Intern (0) phần 2 chỉ Intern–Fresher; Fresher (1) thêm Junior. Expander liệt kê việc có trong (1) nhưng không vào (2) kèm lý do.

---

## 9. Công thức xếp hạng

Danh sách (2):

```
score = cosine
      × (0.75 + 0.25 × level_factor)
      × (0.75 + 0.25 × exp_factor)
      × (0.80 + 0.20 × skill_coverage)
```

- `cosine` = 1 − khoảng cách cosine sklearn.
- Lệch cấp ≤ 1 → 1.0; = 2 → 0.55; hơn → 0.22 (phần 1 không hard-filter; phần 2 cắt theo `max_level_gap`).
- Năm trong khoảng JD → 1.0; ngoài → `exp(-0.35 × gap)`.
- Coverage = tỷ lệ must-have của JD mà CV có.

---

## 10. Số liệu đánh giá

`reports/evaluation.json` — sau lọc corpus + bỏ scaler:

| Metric | Giá trị |
|---|---|
| JD train | 1015 |
| K | 13 |
| PCA 50-d explained | 0.49 |
| Silhouette | 0.13 |
| Davies–Bouldin | 2.37 |
| Size cụm | 38 … 129 |
| Precision@10 | 0.46 (40 query) |
| % `Other` | 21% |

So với corpus 1699 + StandardScaler: silhouette 0.24 (phình vì cụm design/rỗng skill), P@10 0.41, PCA 50-d 29%, size 72–369. File plot: `reports/pca_compare_filter.html`.

Persona Mid React / Senior DevOps khớp family ổn; Fresher Python vẫn lẫn Data/Backend vì Python phổ biến.

`role_family` là proxy — Precision@K **không** phải accuracy có giám sát.

---

## 11. Giới hạn

- Selector website đổi là crawler gãy; không vượt captcha.
- Regex KN/cấp vẫn sai trên JD mơ hồ.
- Từ điển không phủ hết stack; JD IT ít keyword vẫn bị lọc.
- Cụm lai (uiux + openai + sql) còn tồn tại; PCA 2D không tách blob IT.
- Không LLM embedding; không cá nhân hóa lịch sử ứng tuyển; app không cập nhật JD realtime.

---

## 12. Stack

Python, pandas, scikit-learn, Selenium, BeautifulSoup, Streamlit, Plotly, pdfplumber, python-docx, PyYAML, joblib.

---

## License

Đồ án học tập / nghiên cứu. Dữ liệu JD thuộc ITviec và TopCV — chỉ dùng local, không redistribute hay dùng crawl cho mục đích thương mại.
