# Gợi ý việc làm IT từ CV

Hệ thống **gợi ý tin tuyển dụng IT** (ITviec + TopCV) dựa trên nội dung CV. Pipeline học không giám sát: rút kỹ năng → vector TF-IDF/SVD → **K-Means** gom cụm JD → **KNN Cosine** xếp hạng việc phù hợp.

> Đồ án: crawl JD Việt Nam → làm sạch → feature engineering → clustering → recommendation → demo Streamlit (upload CV).

**Kết quả lần train hiện tại** (có thể chạy lại): **1699 JD**, **K = 13**, silhouette **0.186**, Precision@10 (nhãn yếu `role_family`) **0.41**.

---

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Luồng hệ thống](#2-luồng-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt](#4-cài-đặt)
5. [Chạy demo Streamlit (upload CV)](#5-chạy-demo-streamlit-upload-cv)
6. [Chạy full pipeline](#6-chạy-full-pipeline)
7. [Chi tiết từng bước](#7-chi-tiết-từng-bước)
8. [Công thức xếp hạng](#8-công-thức-xếp-hạng)
9. [Số liệu đánh giá](#9-số-liệu-đánh-giá)
10. [Giới hạn](#10-giới-hạn)
11. [Stack](#11-stack)

---

## 1. Bài toán

Ứng viên IT thường phải lướt hàng trăm JD trên ITviec / TopCV. Mục tiêu:

- Biến **file CV** (PDF / DOCX / TXT) thành vector **cùng không gian** với các JD đã train.
- Trả về **Top-N việc** kèm điểm khớp, kỹ năng trùng, kỹ năng còn thiếu, link nguồn.
- Gom JD thành cụm kỹ năng (Backend Java, AI/ML, DevOps, …) để giải thích “nhóm nghề gần nhất”.

Không dùng embedding deep learning / GPU. Feature = từ điển skill + TF-IDF + SVD + năm/cấp.

---

## 2. Luồng hệ thống

```
ITviec / TopCV
        │  Selenium + BeautifulSoup
        ▼
 data/raw/combined.jsonl          (1) Crawl
        │
        ▼
 data/interim/jobs_clean.pkl      (2) Preprocess — lương, KN, cấp, địa điểm, role_family
        │
        ▼
 models/job_vectors.npy           (3) Features — 722 chiều (skill 80% + text SVD 10% + năm/cấp 10%)
        │
        ▼
 models/space.joblib              (4) K-Means — StandardScaler → PCA(50) → L2, chọn K theo silhouette
 models/kmeans.joblib
        │
        ▼
 models/knn.joblib                (5) KNN Cosine trên 50 chiều
 data/processed/jobs.pkl
        │
        ├──────────────────────── (6) Evaluate (tuỳ chọn, báo cáo)
        ▼
 app/streamlit_app.py             (7) Demo: upload CV → gợi ý
```

App **không crawl live** và **không train lại**. Thiếu model thì chạy bước 2→5 (hoặc `scripts/run_all.py`).

---

## 3. Cấu trúc thư mục

```
IT_Job_Recommendation/
├── README.md
├── config.yaml                 # một file config cho cả pipeline
├── requirements.txt
├── notebooks/
│   └── 01_full_pipeline.ipynb  # chạy từng bước từ JD thô → gợi ý CV
├── app/
│   └── streamlit_app.py        # demo upload CV
├── scripts/
│   ├── crawl.py
│   ├── preprocess.py
│   ├── build_features.py
│   ├── train_kmeans.py
│   ├── train_knn.py
│   ├── evaluate.py
│   └── run_all.py              # preprocess → … → evaluate
├── src/
│   ├── paths.py                # mọi đường dẫn tập trung ở đây
│   ├── crawl/                  # ITviec + TopCV
│   ├── preprocess/             # làm sạch JD
│   ├── features/               # skill taxonomy + vectorizer
│   ├── clustering/             # K-Means + PCA space
│   ├── recommend/              # KNN + scoring
│   ├── evaluation/             # silhouette, Precision@K, personas
│   └── cv_parse/               # đọc PDF/DOCX/TXT
├── data/
│   ├── raw/combined.jsonl      # JD đã cào (1711 dòng)
│   ├── sources/                # URL listing
│   ├── dictionaries/skills.yaml
│   ├── samples/cv_fresher_python.txt
│   ├── interim/                # jobs_clean.pkl (sau preprocess)
│   └── processed/jobs.pkl      # bảng JD kèm cụm — app đọc file này
├── models/                     # pipeline, space, kmeans, knn, vectors
└── reports/evaluation.json
```

---

## 4. Cài đặt

Python **3.10–3.12**. Chrome cần khi **cào lại** JD (bước 1). Demo app **không** cần Chrome.

```powershell
cd IT_Job_Recommendation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
cd IT_Job_Recommendation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Chạy demo Streamlit (upload CV)

Repo đã kèm model đã train. Từ thư mục project:

```powershell
streamlit run app/streamlit_app.py
# hoặc, nếu dùng uv:
uv run streamlit run app/streamlit_app.py
```

Không chạy `uv run app/streamlit_app.py` (thiếu `streamlit run` thì không mở được trình duyệt). Khối `if __name__` trong file app sẽ tự chuyển sang `streamlit run` nếu bạn vẫn gõ nhầm.

1. Trang **Gợi ý từ CV**: upload PDF/DOCX/TXT (hoặc bấm “Dùng CV mẫu”).
2. Hệ thống điền sẵn skill / năm / cấp — chỉnh lại nếu parser nhầm.
3. Lọc địa điểm / remote / số gợi ý → **Gợi ý công việc**.
4. Trang **Khám phá cụm**: scatter PCA 2D + profile từng cụm.

---

## 6. Chạy full pipeline

### Cách 1 — Notebook (khuyến nghị khi báo cáo)

Mở [`notebooks/01_full_pipeline.ipynb`](notebooks/01_full_pipeline.ipynb), chọn kernel Python của venv, chạy lần lượt từng cell.

- `SKIP_CRAWL = True` (mặc định): dùng `data/raw/combined.jsonl` có sẵn.
- Đặt `SKIP_CRAWL = False` nếu muốn cào lại (cần Chrome, chậm, dễ bị chặn).

### Cách 2 — Script

Đã có `combined.jsonl`:

```powershell
python scripts/run_all.py
```

Từng bước:

```powershell
python scripts/preprocess.py
python scripts/build_features.py
python scripts/train_kmeans.py
python scripts/train_knn.py
python scripts/evaluate.py          # tuỳ chọn
streamlit run app/streamlit_app.py
```

Cào thêm JD (không bắt buộc):

```powershell
python scripts/crawl.py --source both --max-pages 1 --max-jobs 5
```

Cào xong **phải train lại** từ `preprocess.py`. App không tự cập nhật.

---

## 7. Chi tiết từng bước

### (1) Crawl

Selenium headless, 2 luồng ITviec + TopCV. Listing → card → trang chi tiết. Ghi JSONL incremental (resume theo URL). Không login, không vượt captcha.

Schema `combined.jsonl`: `title`, `company`, `jd_text`, `skills_raw`, `location_raw`, `salary_raw`, `source_url`, …

### (2) Preprocess

- Bỏ JD < 80 ký tự; cắt HTML; Unicode NFC.
- Tách mô tả / yêu cầu / nice-to-have / quyền lợi (quyền lợi **không** vào TF-IDF).
- Lương → VND (USD × 26000). Lương **không** vào vector — chỉ lọc UI.
- Kinh nghiệm, cấp intern→lead (0–5), địa điểm (HCM / Hà Nội / Đà Nẵng / Remote / Other).
- `role_family`: rule trên title — **nhãn yếu**, không phải target train.

1711 thô → **1699** sạch.

### (3) Features

Từ điển `data/dictionaries/skills.yaml` (canonical + alias, khớp alias dài trước; `java` ≠ `javascript`).

Mỗi JD một vector L2:

| Khối | Chiều (lần train) | Trọng số |
|---|---|---|
| TF-IDF skill canonical | 680 | 0.80 |
| TF-IDF 1–2 gram yêu cầu + SVD | 40 | 0.10 |
| `years_mid`, `level_ordinal` (MinMax) | 2 | 0.10 |
| **Tổng** | **722** | L2 lần cuối |

Query CV: bỏ `git` / `html` / `css` khỏi vector tìm. Không có text CV thì zero khối SVD.

### (4) K-Means

Vector 722-d → `StandardScaler` → `PCA(50)` → L2 (~28% phương sai). Chọn **K = argmax silhouette** trên 8…14 → **K = 13**.

Tên cụm gán *sau* (mode `role_family` + top skill), ví dụ `AI/ML: python, llm, langchain_agents`. Cụm **không lọc** kết quả KNN — chỉ hiện “nhóm nghề gần nhất”.

### (5) KNN

`NearestNeighbors(metric="cosine", algorithm="brute")` trên 50 chiều. Fetch dư `top_n × 6` rồi lọc địa điểm / remote / lương.

### (6) Evaluate

Học không giám sát không có nhãn vàng từng JD. Ba lớp số:

- Nội tại: silhouette, Davies–Bouldin, size cụm.
- Retrieval: hold-out Precision@K trên `role_family` (query = skill của chính JD, bỏ JD đó khỏi index).
- 3 persona cố định: Fresher Python, Mid React, Senior DevOps.

---

## 8. Công thức xếp hạng

```
score = cosine
      × (0.75 + 0.25 × level_factor)
      × (0.75 + 0.25 × exp_factor)
      × (0.80 + 0.20 × skill_coverage)
```

- `cosine` = 1 − khoảng cách cosine sklearn.
- Lệch cấp ≤ 1 → 1.0; = 2 → 0.72; hơn → 0.42.
- Năm trong khoảng JD → 1.0; ngoài → `exp(-0.35 × gap)`.
- Coverage = tỷ lệ must-have của JD mà CV có.

---

## 9. Số liệu đánh giá

Lần chạy trong `reports/evaluation.json`:

| Metric | Giá trị |
|---|---|
| Số JD | 1699 |
| Nguồn | TopCV 1100 · ITviec 599 |
| K | 13 |
| Silhouette | 0.186 |
| Davies–Bouldin | 1.99 |
| Precision@10 | 0.41 (30 query) |

Persona Senior DevOps khớp DevOps ổn; Fresher Python dễ lẫn Backend/AI vì Python phổ biến.

`role_family` lệch: ~43% `Other` (title không khớp rule) — Precision@K chỉ là proxy, không phải accuracy có giám sát.

---

## 10. Giới hạn

- Selector website đổi là crawler gãy — không vượt captcha.
- Parser KN/cấp dựa trên regex, JD mơ hồ sẽ sai (một số cụm `mean_years` bất thường).
- Từ điển skill có thể khớp nhầm token ngắn (CAD/CAM, design, sales lẫn vào “IT”).
- Không dùng LLM embedding; không cá nhân hóa theo lịch sử ứng tuyển.
- App không cập nhật JD realtime.

---

## 11. Stack

Python, pandas, scikit-learn, Selenium, BeautifulSoup, Streamlit, Plotly, pdfplumber, python-docx, PyYAML, joblib.

---

## License

Đồ án học tập / nghiên cứu. Dữ liệu JD thuộc ITviec và TopCV — chỉ dùng local, không redistribute hay dùng crawl cho mục đích thương mại.
