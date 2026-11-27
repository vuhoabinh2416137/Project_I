# Hệ thống Phân tích Dữ liệu Học tập LMS & Cảnh báo Sớm Sinh viên (OULAD × SEDL)

> **Đồ án Project I — Đại học Bách khoa Hà Nội (HUST)**  
> **Sinh viên thực hiện:** Vũ Hoà Bình (MSSV: `202416137`)  
> **Phạm vi:** Khám phá dữ liệu đa chiều (EDA), Tiền xử lý chống rò rỉ, Trích xuất đặc trưng theo khung **SEDL**, Huấn luyện & Đối chuẩn 10 mô hình Học máy **5-Fold Stratified Group CV**, Giải thích mô hình (**Explainable AI — Exact TreeSHAP, LIME, PDP/ICE, Subgroup Fairness**) qua **4 mốc thời gian (`25%, 50%, 75%, 100%`)** và Xây dựng **Ứng dụng Demo Cảnh báo Sớm Streamlit**.

---

## 1. Tổng quan Đề tài & Câu hỏi Nghiên cứu

Trong các hệ thống quản lý học tập trực tuyến (LMS), tỷ lệ sinh viên bỏ học (*Withdrawn*) hoặc không đạt (*Fail*) thường rất cao ngay từ nửa đầu khóa học. Đề tài này xây dựng một quy trình Khoa học Dữ liệu & Học máy hoàn chỉnh trên bộ dữ liệu mở **Open University Learning Analytics Dataset (OULAD)** [Kuzilek et al., 2017] kết hợp khung lý thuyết **Student Engagement in Digital Learning (SEDL)** [Bergdahl et al., 2024] để giải quyết 3 câu hỏi nghiên cứu:

1. **RQ1:** Các chỉ số tương tác nào trên LMS (lượt click theo loại tài nguyên, nhịp độ học tập, tiến độ nộp bài, điểm số sớm) có mối liên hệ mạnh nhất với kết quả học tập cuối kỳ?
2. **RQ2:** Có thể dự đoán chính xác sinh viên có nguy cơ gặp khó khăn (*At-Risk = Fail + Withdrawn*) ngay từ giai đoạn sớm (`25%` và `50%` thời lượng khóa học) mà vẫn đạt độ nhạy mục tiêu $\text{Recall} \ge 0.75$ hay không?
3. **RQ3:** Những đặc trưng và chiều tương tác nào (Behavioural, Cognitive, Demographic & Context, Social) đóng vai trò quyết định đối với dự báo rủi ro qua các mốc thời gian, và mô hình giải thích được cả ở mức toàn cục lẫn từng cá nhân như thế nào?

---

## 2. Cơ sở Lý thuyết & Dữ liệu

### 2.1. Bộ dữ liệu OULAD
- **Nguồn:** Open University Learning Analytics Dataset (The Open University, UK).
- **Quy mô:** **32,593** lượt đăng ký học phần (`28,785` sinh viên duy nhất) qua **7 học phần** (`AAA`–`GGG`) và **4 học kỳ** (`2013B`, `2013J`, `2014B`, `2014J`), bao gồm 7 bảng chuẩn (`courses`, `assessments`, `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle`, `vle`).
- **Biến mục tiêu nhị phân (`at_risk`):**
  - `1` (**At-Risk**): `Fail` + `Withdrawn` (`52.80%` trên toàn bộ `32,593` lượt đăng ký).
  - `0` (**Not At-Risk / Safe**): `Pass` + `Distinction` (`47.20%`).
- **Chống rò rỉ dữ liệu theo thời gian (`Dynamic Active Cohort`):** Tại mỗi lát cắt $T \in \{25\%, 50\%, 75\%, 100\%\}$ thời lượng môn học, quy trình chỉ giữ lại dữ liệu tương tác phát sinh đến ngày $T_{\text{cutoff}}$ và lọc các sinh viên chưa hủy đăng ký trước mốc cắt (`date_unregistration.isna() | date_unregistration > cutoff_day`). Quy mô `dynamic_active` qua 4 mốc lần lượt là **$N = 26,248$** (`25%`), **$N = 24,601$** (`50%`), **$N = 23,159$** (`75%`) và **$N = 22,522$** (`100%`).

### 2.2. Khung Tương tác Đa chiều SEDL
Nghiên cứu trích xuất **92 đặc trưng** (`114` chiều sau mã hóa One-Hot và gắn cờ thiếu dữ liệu) ánh xạ theo khung SEDL [Bergdahl et al., 2024]:
- **Behavioural (Hành vi):** Tổng lượt click, số ngày hoạt động, tỷ lệ hoàn thành bài tập (`submission_rate`, `course_progress_submission_rate`), thời gian vắng mặt gần nhất (`days_since_last_activity`, `max_inactivity_streak`), tính đều đặn hàng tuần (`consistency_score`, `login_regularity`), hệ số suy giảm tương tác (`engagement_decay`).
- **Cognitive (Nhận thức & Học thuật):** Tương tác học liệu lõi (`oucontent`, `quiz`, `resource`), độ đa dạng và Entropy tài nguyên (`activity_diversity`, `activity_entropy`), điểm trung bình có trọng số (`weighted_avg_score`, `avg_score`), quỹ đạo điểm số (`score_trend_slope`).
- **Social (Xã hội):** Tương tác diễn đàn thảo luận (`clicks_forumng`, `forumng_ratio`), wiki và phòng học trực tuyến (`clicks_ouwiki`, `clicks_oucollaborate`).
- **Demographic & Context (Bối cảnh):** Học vấn đầu vào (`highest_education`), chỉ số kinh tế-xã hội vùng (`imd_band`), độ tuổi (`age_band`), tín chỉ (`studied_credits`), số lần học lại (`num_of_prev_attempts`) và học phần (`code_module`).

---

## 3. Cấu trúc Dự án (Kiến trúc Notebook-Only)

Dự án tuân thủ nghiêm ngặt kiến trúc **Notebook-Only** (tự chủ 100% trong `notebooks/`, không sử dụng thư mục `src/`):

```text
Project_I/
├── AGENTS.md                              # Quy tắc dự án bắt buộc
├── KE_HOACH_DU_AN_LMS_LEARNING_ANALYTICS.md
├── README.md                              # Tài liệu hướng dẫn tổng quan (tệp này)
├── requirements.txt                       # Danh sách thư viện Python tái lập môi trường
├── notebooks/                             # Chuỗi 7 Jupyter Notebooks tự chủ hoàn toàn
│   ├── 01_data_loading.ipynb              # Tuần 1-2: Tải, kiểm tra tính toàn vẹn 7 bảng OULAD
│   ├── 02_eda.ipynb                       # Tuần 3: EDA đa chiều, kiểm định thống kê & SEDL
│   ├── 03_preprocessing.ipynb             # Tuần 4: Imputation, IQR capping, Log1p, RobustScaler, SMOTE
│   ├── 04_feature_engineering.ipynb       # Tuần 5-6: Tạo 4 lát cắt 25/50/75/100% & 92 đặc trưng SEDL
│   ├── 05_baseline_models.ipynb           # Tuần 7A: 5-Fold Stratified Group CV cho 5 Baseline
│   ├── 06_advanced_models.ipynb           # Tuần 7B: 5-Fold CV + Nested Tuning + 10 mô hình đa mốc
│   └── 07_model_explanation.ipynb         # Tuần 8A: XAI đa mốc 25%–100% (Gain, Perm, Exact TreeSHAP, LIME, PDP/ICE, Fairness)
├── models/
│   ├── best_model.pkl                     # Mô hình XGBoost (Tuned) đồng thuận tại 25% dynamic_active
│   ├── best_model_5fold_ensemble.joblib   # Bộ 5 mô hình + tiền xử lý In-Fold mốc 25%
│   ├── best_model_5fold_4milestones.joblib# Bộ 5-Fold OOF + tiền xử lý In-Fold + ngưỡng tau*_k cho cả 4 mốc 25%–100%
│   └── best_model_metadata.json           # Danh sách 114 đặc trưng, siêu tham số & metrics CV
├── demo/
│   ├── app.py                             # Ứng dụng Streamlit Cảnh báo Sớm & XAI 5-Fold OOF Đa mốc (5 Tab)
│   └── assets/                            # Ảnh chụp giao diện Demo thực tế
├── data/
│   ├── raw/                               # Dữ liệu OULAD gốc (7 tệp CSV)
│   └── processed/                         # Dữ liệu & kết quả thực nghiệm phân theo từng tuần
│       ├── week3_eda/
│       ├── week4_preprocessing/
│       ├── week5_6_snapshots/             # 4 tệp lát cắt 25%, 50%, 75%, 100%
│       ├── week5_feature_engineering/
│       ├── week6_advanced_features/
│       ├── week7_modeling/
│       └── week8_xai/                     # Metrics XAI, Top Consensus, Fairness, Case Studies & OOF summary 4 mốc
├── reports/                               # Báo cáo Markdown từng tuần & Hình vẽ (01..44)
│   └── figures/
└── báo_cáo/                               # Báo cáo học thuật LaTeX & PDF từng tuần
```

---

## 4. Tóm tắt Kết quả Chính (Tuần 3 – Tuần 8)

### 4.1. Thẩm định 5-Fold Stratified Group OOF qua 4 Mốc (`XGBoost (Tuned)`, Dynamic Active)
Nhờ khóa chặt `id_student` qua `StratifiedGroupKFold(n_splits=5)` và tiền xử lý **100% In-Fold (Zero Data Leakage)** cùng tối ưu ngưỡng theo từng fold:

| Mốc | Quy mô ($N$) | Ngưỡng $\bar{\tau}^*$ | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Brier | Max SHAP Err | LIME $R^2$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **25%** | 26,248 | 0.432 | **0.8480** | 0.8286 | **0.7696** | 0.6853 | 0.7249 | 0.1544 | `3.32e-07` | 0.6755 |
| **50%** | 24,601 | 0.496 | **0.9068** | 0.8863 | **0.7813** | 0.7813 | 0.7808 | 0.1158 | `4.03e-07` | 0.6764 |
| **75%** | 23,159 | 0.620 | **0.9457** | 0.9246 | **0.7817** | 0.8850 | 0.8299 | 0.0839 | `4.42e-07` | 0.6744 |
| **100%** | 22,522 | 0.574 | **0.9655** | 0.9486 | **0.8334** | 0.8987 | 0.8645 | 0.0633 | `3.58e-07` | 0.6939 |

### 4.2. Giải thích Mô hình Đa mốc (XAI) & Ứng dụng Demo (`Tuần 8`)
- Triển khai 5 kỹ thuật XAI bổ trợ qua cả **4 mốc (`25%, 50%, 75%, 100%`)**: **XGBoost Gain**, **5-Fold OOF Permutation Importance**, **Exact TreeSHAP** (`pred_contribs=True` trên C++ core của XGBoost), **Local Surrogate LIME** (`200` SV OOF/mốc), và **PDP/ICE + SHAP Dependence** cùng **Kiểm định Công bằng Nhóm (Subgroup Fairness Audit)**.
- Tỷ trọng nhóm đặc trưng SEDL dịch chuyển rõ nét theo tiến độ khóa học: `Behavioural` tăng từ `40.82%` (`25%`) lên `59.38%` (`100%`), trong khi `Demographic & Context` giảm mạnh từ `21.02%` xuống còn `7.84%`.
- Xây dựng ứng dụng **Streamlit Dashboard (`demo/app.py`)** chấm điểm **5-Fold Out-of-Fold chuẩn cho cả 4 mốc (`25%–100%`)** gồm **5 Tab chuyên sâu**:
  1. **Tổng quan Cảnh báo & Top 10 XAI:** KPI cards theo mốc, phân tầng 4 mức rủi ro, breakdown theo `code_module` và `code_presentation`, ma trận nhầm lẫn và bảng **Global Consensus Top 10 XAI** + tương quan hạng Spearman theo từng mốc.
  2. **Danh sách Ưu tiên Can thiệp:** Bảng tra cứu sinh viên `At-Risk` sắp xếp theo `Risk Score`, tìm kiếm theo `id_student`, lọc đa tiêu chí và xuất file `.csv`.
  3. **Chẩn đoán Cá nhân, Exact TreeSHAP & Giả lập What-If:** Hỗ trợ chọn nhanh các **Case Studies TP/FP/TN/FN** theo từng mốc, đồng hồ đo rủi ro so với $\tau^*$, biểu đồ **Exact TreeSHAP** dùng đúng OOF fold model, **Radar 4 chiều SEDL**, khuyến nghị sư phạm tự động và **What-If Simulator**.
  4. **Xu hướng SEDL & Kiểm định Công bằng:** Phân bố đặc trưng, **diễn tiến tỷ trọng 4 chiều SEDL (`25% → 100%`)** và kiểm định Fairness (`Recall/TPR`, `FPR`, `ROC-AUC`, `Disparate Impact Ratio`) theo `gender`, `age_band`, `imd_band`, `disability`, `region` cùng bảng tổng hợp 4 mốc.
  5. **Đối sánh Đa mốc (25% → 100%):** Đường cong hiệu năng 10 mô hình qua 4 mốc, bảng thẩm định OOF & LIME 4 mốc và **quỹ đạo rủi ro OOF dọc thời gian** cho từng sinh viên cụ thể.

---

## 5. Hướng dẫn Cài đặt & Vận hành

### 5.1. Yêu cầu Môi trường
- Python **3.10+** (đã kiểm chứng trên **Python 3.12.10**, Windows).
- Cài đặt toàn bộ thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 5.2. Tái tạo Pipeline Notebook (Tùy chọn)
Nếu muốn chạy lại toàn bộ quy trình từ dữ liệu gốc OULAD (`data/raw/`), thực thi tuần tự các notebook trong `notebooks/`:
1. `01_data_loading.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_baseline_models.ipynb`
6. `06_advanced_models.ipynb`
7. `07_model_explanation.ipynb`

### 5.3. Khởi chạy Ứng dụng Demo Streamlit
Từ thư mục gốc của dự án (`Project_I`), chạy lệnh sau:

```bash
streamlit run demo/app.py
```

Sau đó mở trình duyệt tại địa chỉ mặc định `http://localhost:8501`.

---

## 6. Tài liệu Tham khảo Chính
- Bergdahl, N., Nouri, J., Fors, U., & Knutsson, O. (2024). *Engagements in digital learning: A systematic review of learning analytics research on student engagement*. International Journal of Educational Technology in Higher Education, 21(1), 63.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open university learning analytics dataset*. Scientific Data, 4, 170171.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020). *From local explanations to global understanding with explainable AI for trees*. Nature Machine Intelligence, 2(1), 56–67.
