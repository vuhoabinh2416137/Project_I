# BÁO CÁO TỔNG KẾT TUẦN 1: KHỞI ĐỘNG NGHIÊN CỨU & THIẾT LẬP DỰ ÁN
## Đề tài: LMS Tiên tiến và Learning Analytics

---

## 1. MỤC TIÊU VÀ TRẠNG THÁI HOÀN THÀNH

Tuần 1 đã hoàn thành toàn bộ các mục tiêu đề ra theo kế hoạch tổng thể:

| STT | Nhiệm vụ | Trạng thái | Tài liệu / Sản phẩm bàn giao |
|:---:|---|:---:|---|
| **1.1** | Nghiên cứu bài báo Bergdahl et al. (2024) và khung SEDL (4 chiều tương tác) | **HOÀN THÀNH** | [01_literature_review_summary.md](../references/01_literature_review_summary.md) |
| **1.2** | Tổng hợp 4 bài báo kinh điển về OULAD & Early Prediction (Kuzilek 2017, Hlosta 2017, Umer 2017, Jayaprakash 2014) | **HOÀN THÀNH** | [01_literature_review_summary.md](../references/01_literature_review_summary.md) (Mục 2 & 3) |
| **1.3** | Tìm hiểu cấu trúc OULAD và xây dựng sơ đồ thực thể liên kết | **HOÀN THÀNH** | [02_oulad_data_architecture.md](../references/02_oulad_data_architecture.md) (Kèm Mermaid ERD) |
| **1.4** | Xác định câu hỏi nghiên cứu, giả thuyết và vận hành hóa biến đo lường | **HOÀN THÀNH** | [03_research_questions_and_methodology.md](../references/03_research_questions_and_methodology.md) |
| **1.5** | Thiết lập môi trường lập trình (Python, packages ML/XAI, cấu trúc thư mục) | **HOÀN THÀNH** | [requirements.txt](../requirements.txt) & [check_env.py](../src/check_env.py) |

---

## 2. TÓM TẮT KẾT QUẢ ĐẠT ĐƯỢC

### 2.1. Cấu trúc thư mục dự án chuẩn mực

```text
Project_I/    
├── data/    
│   ├── raw/                 # Dữ liệu 7 file CSV của OULAD (chuẩn bị nạp ở Tuần 2)    
│   └── processed/           # Dữ liệu sạch, features sau khi xử lý    
├── demo/    
│   └── assets/              # Tài nguyên hình ảnh, logo cho dashboard Streamlit    
├── models/                  # Lưu trữ model weights (.pkl / .joblib / .json)    
├── notebooks/               # Jupyter notebooks thực nghiệm (01_data_loading.ipynb, ...)    
├── references/              # Báo cáo tổng quan tài liệu, kiến trúc dữ liệu, RQ    
│   ├── 01_literature_review_summary.md    
│   ├── 02_oulad_data_architecture.md    
│   └── 03_research_questions_and_methodology.md    
├── reports/    
│   ├── figures/             # Biểu đồ EDA, SHAP plots xuất bản    
│   └── WEEK1_EXECUTIVE_SUMMARY.md    
├── src/                     # Source code module hóa (data pipeline, features, train)    
│   └── check_env.py    
├── requirements.txt         # Danh mục thư viện Python    
└── KE_HOACH_DU_AN_LMS_LEARNING_ANALYTICS.md  # Kế hoạch chi tiết 12 tuần    
```

### 2.2. Môi trường công nghệ
- **Python Version**: `3.12.10`
- **Thư viện nền tảng**: `numpy 2.4.4`, `pandas 3.0.3`, `scipy 1.17.1`
- **Trực quan hóa**: `matplotlib 3.10.9`, `seaborn 0.13.2`
- **Học máy & XAI**: `scikit-learn 1.8.0`, `xgboost 3.4.1`, `lightgbm 4.7.0`, `shap 0.53.0`
- **Giao diện Dashboard**: `streamlit 1.66.0`

### 2.3. Sẵn sàng cho Tuần 2
Tuần 2 sẽ tập trung vào:
1. Tải và giải nén bộ dữ liệu OULAD vào `data/raw/`.
2. Xây dựng notebook `notebooks/01_data_loading.ipynb` để kiểm tra toàn vẹn dữ liệu (missing values, types, distribution).
3. Đánh giá tính phân bố của nhãn `final_result` và kiểm tra liên kết giữa 7 bảng.
