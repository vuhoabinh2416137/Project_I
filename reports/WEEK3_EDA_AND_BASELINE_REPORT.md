# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & ĐỐI CHUẨN MÔ HÌNH BASELINE
## Tuần 3: Khám phá Dữ liệu (EDA), Kiểm định Thống kê & Chuyển đổi Mô hình 4-Sample sang 2-Sample
## Đề tài: LMS Tiên tiến và Learning Analytics

---

## 1. TỔNG QUAN TIẾN ĐỘ TUẦN 3

Toàn bộ 7 hạng mục công việc của Tuần 3 (từ T3.1 đến T3.7) đã được hoàn thành thông qua 2 pipeline thực nghiệm chính ([src/eda_analysis.py](file:///c:/Users/mrore/Project_I/src/eda_analysis.py) và [src/baseline_4to2_model.py](file:///c:/Users/mrore/Project_I/src/baseline_4to2_model.py)), đóng gói vào Jupyter Notebook [notebooks/02_eda.ipynb](file:///c:/Users/mrore/Project_I/notebooks/02_eda.ipynb) và xuất bản **11 biểu đồ phân tích chuyên sâu** (`04_*.png` đến `14_*.png` tại `reports/figures/`).

| Mã CV | Nội dung công việc Tuần 3 | Trạng thái | Sản phẩm đầu ra |
|:---:|---|:---:|---|
| **T3.1** | Phân tích phân bố clicks và ngày hoạt động theo `final_result` (4 lớp & 2 lớp) | Hoàn thành | [04_clicks_by_final_result.png](file:///c:/Users/mrore/Project_I/reports/figures/04_clicks_by_final_result.png) |
| **T3.2** | Phân tích điểm đánh giá (`avg_score`), điểm bài đầu tiên (`TMA1`) & tính kịp thời | Hoàn thành | [05_assessment_score_vs_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/05_assessment_score_vs_outcome.png), [06_submission_timeliness_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/06_submission_timeliness_analysis.png) |
| **T3.3** | Phân tích chuỗi thời gian (Temporal Patterns) theo tuần học, tách riêng `Fail` và `Withdrawn` | Hoàn thành | [07_temporal_weekly_clicks_trend.png](file:///c:/Users/mrore/Project_I/reports/figures/07_temporal_weekly_clicks_trend.png) |
| **T3.4** | Phân tích clicks theo `activity_type` và đối chiếu 3 chiều Khung lý thuyết SEDL | Hoàn thành | [08_activity_type_by_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/08_activity_type_by_outcome.png), [09_sedl_dimensions_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/09_sedl_dimensions_comparison.png) |
| **T3.5** | Phân tích ảnh hưởng của nhân khẩu học (Demographics Cross-tabulation) | Hoàn thành | [10_demographics_vs_at_risk.png](file:///c:/Users/mrore/Project_I/reports/figures/10_demographics_vs_at_risk.png) |
| **T3.6** | Kiểm tra bản ghi trùng 6 cột `studentVle`, ma trận tương quan & kiểm định thống kê (t-test, ANOVA, $\chi^2$) | Hoàn thành | [11_correlation_matrix_heatmap.png](file:///c:/Users/mrore/Project_I/reports/figures/11_correlation_matrix_heatmap.png), [12_statistical_tests_effect_size.png](file:///c:/Users/mrore/Project_I/reports/figures/12_statistical_tests_effect_size.png) |
| **T3.7** | Chạy thử mô hình bài báo dự đoán cũ, chuyển cấu hình từ **4 sample** sang **2 sample** | Hoàn thành | [13_baseline_4class_vs_2class_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/13_baseline_4class_vs_2class_comparison.png), [14_baseline_confusion_matrices_and_roc.png](file:///c:/Users/mrore/Project_I/reports/figures/14_baseline_confusion_matrices_and_roc.png) |

---

## 2. KIỂM ĐỊNH MỞ RỘNG BẢNG `studentVle` & KHÓA NGOẠI TỔ HỢP (T3.6)

Tiếp nối khuyến nghị từ Báo cáo Tuần 2, nhóm nghiên cứu đã kiểm tra chi tiết bản chất trùng lặp của `studentVle` (10,655,280 dòng) và đối chiếu khóa ngoại tổ hợp:

1. **Phân biệt trùng lặp hoàn toàn (6 cột) và trùng khóa phiên (5 cột)**:
   - Số dòng trùng hoàn toàn trên cả 6 cột (`code_module`, `code_presentation`, `id_student`, `id_site`, `date`, `sum_click`): **787,170 dòng (7.39%)**.
   - Số dòng trùng trên khóa 5 cột `(code_module, code_presentation, id_student, id_site, date)`: **2,195,960 dòng (20.61%)**. Điều này chứng minh hệ thống logging của The Open University ghi nhận từng phiên truy cập (session-level logging) trong cùng một ngày của sinh viên vào một tài nguyên; có 787,170 phiên tình cờ có cùng số lượt click (thường là `sum_click = 1` hoặc `2`).
   - Do đó, phép cộng gộp `groupby(['code_module', 'code_presentation', 'id_student', 'id_site', 'date'])['sum_click'].sum()` phản ánh đúng tổng tương tác thực tế (**39,605,099 clicks**).
2. **Kiểm tra Khóa ngoại Tổ hợp & Đa đăng ký (Multi-Enrollment)**:
   - Đối chiếu bộ ba khóa `(code_module, code_presentation, id_student)` giữa `studentRegistration`, `studentVle` với `studentInfo`: **0 bản ghi lệch (100% khớp)**.
   - Đối chiếu `(code_module, code_presentation, id_site)` giữa `studentVle` và `vle`: **0 bản ghi lệch (100% khớp)**.
   - Trong 32,593 bản ghi đăng ký môn học, có **28,785 định danh `id_student` duy nhất**, tức có **3,538 sinh viên** đăng ký từ 2 môn/kỳ trở lên $\rightarrow$ Củng cố bắt buộc phải dùng **Group Split theo `id_student`** ở Tuần 4 để chống rò rỉ hồ sơ cá nhân.
   - Có **29,228 lượt đăng ký (89.68%)** phát sinh tương tác VLE và **3,365 lượt đăng ký (10.32%)** hoàn toàn không có click nào (`total_clicks = 0`, hầu hết thuộc nhóm rút môn sớm `Withdrawn`).

---

## 3. CÁC PHÁT HIỆN CHÍNH TỪ PHÂN TÍCH KHÁM PHÁ (KEY EDA FINDINGS)

### 3.1. Chiều Hành vi (Behavioural Engagement) - Kiểm định Giả thuyết H1a
Biểu đồ minh họa: [reports/figures/04_clicks_by_final_result.png](file:///c:/Users/mrore/Project_I/reports/figures/04_clicks_by_final_result.png)

| Nhóm kết quả | Số lượng SV | Mean Clicks | Median Clicks | IQR Clicks | Mean Active Days | Median Active Days | Clicks / Active Day |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Distinction** | 3,024 | 2,666.8 | 1,896.0 | 2,823.0 | 110.0 ngày | 103.0 ngày | 21.94 |
| **Pass** | 12,361 | 1,921.8 | 1,343.0 | 1,885.0 | 87.0 ngày | 78.0 ngày | 20.46 |
| **Fail** | 7,052 | 651.9 | 317.0 | 698.3 | 33.2 ngày | 22.0 ngày | 16.29 |
| **Withdrawn** | 10,156 | 314.0 | 89.0 | 373.0 | 16.3 ngày | 7.0 ngày | 12.05 |
| **Not At-Risk (0)** | **15,385** | **2,068.2** | **1,425.0** | — | **91.6 ngày** | **82.0 ngày** | **20.76** |
| **At-Risk (1)** | **17,208** | **452.4** | **175.0** | — | **23.2 ngày** | **12.0 ngày** | **13.79** |

> **Kết luận H1a (ĐƯỢC XÁC NHẬN)**: Sinh viên nhóm `Not At-Risk` có tổng lượt click trung bình cao gấp **4.57 lần** (`2,068.2` vs `452.4`, Welch's $t = 93.96, p < 0.0001$, Cohen's $d = 1.086$) và số ngày hoạt động cao gấp **3.95 lần** (`91.6` vs `23.2` ngày, Welch's $t = 140.71, p < 0.0001$, Cohen's $d = 1.608$) so với nhóm `At-Risk`. Đặc biệt, **số ngày duy trì hoạt động (`active_days`, $r = -0.626$)** có khả năng phân tách mạnh hơn tổng số click thô (`total_clicks`, $r = -0.477$), chứng tỏ tính bền bỉ theo ngày quan trọng hơn việc click dồn trong vài phiên.

---

### 3.2. Chiều Nhận thức (Cognitive Engagement), Điểm `TMA1` & Tính Kịp thời - Kiểm định Giả thuyết H1b
Biểu đồ minh họa: [reports/figures/05_assessment_score_vs_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/05_assessment_score_vs_outcome.png) và [reports/figures/06_submission_timeliness_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/06_submission_timeliness_analysis.png)

1. **Vai trò quyết định của Bài kiểm tra đầu tiên (`TMA1`)**:
   - Có tới **97.01%** sinh viên `Not At-Risk` nộp bài `TMA1` (điểm trung bình `77.98/100`), trong khi chỉ có **55.21%** sinh viên `At-Risk` nộp bài `TMA1` (`76.80%` ở nhóm `Fail` và chỉ `40.22%` ở nhóm `Withdrawn`).
   - Phân tầng nguy cơ theo kết quả `TMA1`:
     - **Không nộp `TMA1`** ($n = 8,167$): Tỷ lệ rơi vào `At-Risk` lên tới **94.37%**!
     - **Điểm `TMA1 < 40`** ($n = 1,410$): Tỷ lệ `At-Risk` là **62.62%**.
     - **Điểm `TMA1 40 - 60`** ($n = 3,093$): Tỷ lệ `At-Risk` là **63.60%**.
     - **Điểm `TMA1 60 - 75`** ($n = 6,563$): Tỷ lệ `At-Risk` giảm xuống **43.70%**.
     - **Điểm `TMA1 75 - 100`** ($n = 13,360$): Tỷ lệ `At-Risk` chỉ còn **28.32%**.
2. **Hành vi chuẩn bị trước khai giảng (`pre_course_clicks > 0`, `date < 0`)**:
   - Sinh viên chủ động truy cập LMS trước ngày mở lớp ($n = 23,809$) có tỷ lệ `At-Risk` là **44.66%**, thấp hơn **30.19 điểm phần trăm** so với nhóm không truy cập trước khai giảng ($n = 8,784$, tỷ lệ `At-Risk` lên tới **74.85%**, $\chi^2 = 2,345.86, p < 0.0001$).
3. **Tính kịp thời khi nộp bài (`avg_days_early`)**:
   - Nhóm `Distinction` nộp bài sớm trung bình **17.95 ngày**, nhóm `Pass` sớm **13.17 ngày**, trong khi nhóm `Fail` (`9.28 ngày`) và `Withdrawn` (`8.06 ngày`) sát hạn chót hơn đáng kể ($t = 40.65, p < 0.0001$).

> **Kết luận H1b (ĐƯỢC XÁC NHẬN)**: Các chỉ số nhận thức và tự điều chỉnh học tập (`submission_rate` với $d = 2.829$, `avg_score` với $d = 1.507$, `tma1_score` với $d = 1.219$, `clicks_oucontent` với $d = 0.823$, và `pre_course_clicks`) đều có ý nghĩa thống kê cực cao ($p < 0.0001$).

---

### 3.3. Phân tích Chuỗi thời gian (Temporal Patterns) & Sự Phân hóa `Fail` vs `Withdrawn` (T3.2 & T3.3)
Biểu đồ minh họa: [reports/figures/07_temporal_weekly_clicks_trend.png](file:///c:/Users/mrore/Project_I/reports/figures/07_temporal_weekly_clicks_trend.png)

| Mốc thời gian | Distinction (TB Clicks \| % Active) | Pass (TB Clicks \| % Active) | Fail (TB Clicks \| % Active) | Withdrawn (TB Clicks \| % Active) |
|---|:---:|:---:|:---:|:---:|
| **Tuần -1 (Trước khai giảng)** | 62.7 \| 80.3% | 40.9 \| 73.2% | 21.8 \| 55.0% | 17.4 \| 43.9% |
| **Tuần 0 (Tuần khai giảng)** | 113.6 \| 89.8% | 81.1 \| 86.5% | 44.7 \| 71.2% | 34.7 \| 51.5% |
| **Tuần 2 (Đỉnh tương tác đầu kỳ)** | 117.9 \| 91.2% | 92.4 \| 89.9% | 55.7 \| 72.1% | 40.9 \| 47.9% |
| **Tuần 4 (Sau ~1 tháng)** | 87.4 \| 89.9% | 65.0 \| 86.2% | 32.1 \| 62.2% | 20.8 \| 38.5% |
| **Tuần 9 (Mốc 25% khóa học)** | **71.4 \| 85.9%** | **48.1 \| 78.0%** | **20.1 \| 44.0%** | **10.9 \| 21.8%** |
| **Tuần 18 (Mốc 50% khóa học)** | **80.6 \| 88.0%** | **60.0 \| 82.1%** | **16.2 \| 33.9%** | **4.0 \| 9.8%** |
| **Tuần 27 (Mốc 75% khóa học)** | 56.7 \| 82.6% | 41.3 \| 71.4% | 7.4 \| 19.7% | 0.5 \| 2.3% |

> **Phát hiện cốt lõi**: Giả thuyết từ Tuần 2 về sự phân hóa giữa `Withdrawn` và `Fail` đã được chứng minh định lượng:
> - **Nhóm `Withdrawn` (Rút môn)**: Rơi rụng cực nhanh ngay từ Tuần 2 đến Tuần 9. Tại **mốc 25% (Tuần 9)**, gần **80% sinh viên Withdrawn đã ngừng truy cập hoàn toàn** (chỉ còn `21.8%` active).
> - **Nhóm `Fail` (Thi trượt)**: Vẫn duy trì `71.2% - 72.1%` hoạt động ở 2 tuần đầu và còn `44.0%` hoạt động ở Tuần 9, `33.9%` ở Tuần 18, nhưng cường độ click mỗi tuần chỉ bằng ~1/3 so với nhóm `Pass`.

---

### 3.4. Chiều Xã hội (Social Engagement - `forumng`) & Đa chiều SEDL - Kiểm định Giả thuyết H1c
Biểu đồ minh họa: [reports/figures/08_activity_type_by_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/08_activity_type_by_outcome.png) và [reports/figures/09_sedl_dimensions_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/09_sedl_dimensions_comparison.png)

| Mức độ tương tác Diễn đàn (`forumng_clicks`) | Distinction (%) | Pass (%) | Fail (%) | Withdrawn (%) | Tổng At-Risk (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **0 clicks (Cô lập xã hội hoàn toàn)** | 0.87% | 6.04% | 22.15% | **70.94%** | **93.09%** |
| **1 - 20 clicks (Tương tác thấp)** | 3.64% | 24.80% | 33.37% | 38.19% | **71.56%** |
| **21 - 100 clicks (Tương tác trung bình)** | 6.24% | 38.75% | 27.35% | 27.66% | **55.01%** |
| **> 100 clicks (Tương tác tích cực)** | 16.60% | **56.69%** | 14.62% | **12.09%** | **26.71%** |

> **Kết luận H1c (ĐƯỢC XÁC NHẬN)**: Sự cô lập xã hội trên môi trường học tập số (`forumng_clicks = 0`) gắn liền với tỷ lệ bỏ học giữa chừng lên tới **70.94%** (tổng nguy cơ `At-Risk` là **93.09%**). Ngược lại, sinh viên tham gia diễn đàn trên 100 clicks có tỷ lệ đỗ (`Pass + Distinction`) đạt **73.29%** và tỷ lệ bỏ học giảm gần 6 lần xuống chỉ còn **12.09%** ($F = 1,238.15, p < 0.0001$).

---

### 3.5. Ảnh hưởng của Nhân khẩu học & Kiểm định Thống kê Tổng hợp (T3.5 & T3.6)
Biểu đồ minh họa: [reports/figures/10_demographics_vs_at_risk.png](file:///c:/Users/mrore/Project_I/reports/figures/10_demographics_vs_at_risk.png), [reports/figures/11_correlation_matrix_heatmap.png](file:///c:/Users/mrore/Project_I/reports/figures/11_correlation_matrix_heatmap.png), [reports/figures/12_statistical_tests_effect_size.png](file:///c:/Users/mrore/Project_I/reports/figures/12_statistical_tests_effect_size.png)

- **Các nhóm nhân khẩu học có rủi ro cao nhất**:
  1. **Trình độ học vấn đầu vào (`highest_education`)**: Nhóm `No Formal quals` có tỷ lệ At-Risk **70.32%** và `Lower Than A Level` là **61.15%**, cao gấp đôi so với `Post Graduate Qualification` (**34.50%**, $\chi^2 = 737.25, p < 0.0001$, Cramer's $V = 0.1504$).
  2. **Chỉ số kinh tế xã hội vùng (`imd_band`)**: Giảm đơn điệu đều đặn từ vùng nghèo nhất `0-10%` (**64.84% At-Risk**) xuống vùng khá giả nhất `90-100%` (**42.47% At-Risk**, $\chi^2 = 721.97, p < 0.0001$, Cramer's $V = 0.1488$).
  3. **Tiền sử học lại (`num_of_prev_attempts`)**: Sinh viên học lần đầu có tỷ lệ At-Risk **50.72%**, tăng lên **65.44%** nếu đã trượt 1 lần và **72.51%** nếu trượt $\ge 2$ lần.
  4. **Áp lực tín chỉ (`studied_credits`)**: Sinh viên đăng ký quá tải (`> 120 tín chỉ`) có tỷ lệ At-Risk **69.64%**, so với **48.20%** ở nhóm học vừa sức (`<= 60 tín chỉ`).
  5. **Khuyết tật (`disability`)**: Nhóm có khuyết tật (`Y`) có tỷ lệ At-Risk **61.85%** so với **51.82%** ở nhóm không khuyết tật ($\chi^2 = 114.90, p < 0.0001$).
- **So sánh Độ lớn Hiệu ứng (Effect Size)**: Mặc dù mọi biến nhân khẩu học đều có $p < 0.0001$ (do kích thước mẫu lớn $N = 32,593$), hệ số tương quan và độ lớn hiệu ứng của nhân khẩu học ($|r| \in [0.10, 0.15]$, Cramer's $V \le 0.15$) **nhỏ hơn rất nhiều** so với các đặc trưng tương tác LMS theo khung SEDL (`submission_rate`: $d = 2.829$, `active_days`: $d = 1.608$, `avg_score`: $d = 1.507$, `tma1_score`: $d = 1.219$, `total_clicks`: $d = 1.086$, `clicks_oucontent`: $d = 0.823$). Điều này chứng minh **hành vi nỗ lực thực tế trong khóa học quyết định kết quả nhiều hơn xuất phát điểm nhân khẩu học**.

---

## 4. THỰC NGHIỆM MÔ HÌNH BÀI BÁO CŨ: CHUYỂN CẤU HÌNH 4-SAMPLE SANG 2-SAMPLE (NHIỆM VỤ T3.7)

Biểu đồ minh họa: [reports/figures/13_baseline_4class_vs_2class_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/13_baseline_4class_vs_2class_comparison.png) và [reports/figures/14_baseline_confusion_matrices_and_roc.png](file:///c:/Users/mrore/Project_I/reports/figures/14_baseline_confusion_matrices_and_roc.png)

### 4.1. Thiết kế thực nghiệm đối chuẩn
Nhóm nghiên cứu tái hiện bộ mô hình phân loại cơ sở từ các bài báo gốc trên OULAD (Kuzilek et al., 2017; Hlosta et al., 2017; Jayaprakash et al., 2014) gồm 4 thuật toán (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`) trên cùng phép chia Stratified Train/Test (80% Train = 26,074 mẫu; 20% Test = 6,519 mẫu) để so sánh 3 cấu hình nhãn:
1. **Cấu hình 4-Sample Gốc (4-Class Multiclass)**: Dự đoán 4 lớp `Distinction`, `Pass`, `Fail`, `Withdrawn`.
2. **Cấu hình 4-sang-2 Hậu nghiệm (Post-hoc Probability Aggregation)**: Huấn luyện mô hình 4 lớp nhưng gộp xác suất dự đoán $P(\text{At-Risk}) = P(\text{Fail}) + P(\text{Withdrawn})$.
3. **Cấu hình 2-Sample Trực tiếp (Direct Binary Target Adaptation)**: Chuyển đổi trực tiếp mục tiêu bài toán sang 2 lớp `0: Not At-Risk` (`Pass + Distinction`) và `1: At-Risk` (`Fail + Withdrawn`).

### 4.2. Bảng đối sánh kết quả giữa Cấu hình 4-Sample và 2-Sample (Toàn khóa)

| Mô hình Baseline | 4-Sample Accuracy | 4-Sample Macro-F1 | 4-Sample F1 (`Fail`) | 4$\rightarrow$2 Mapped F1 | **2-Sample Accuracy** | **2-Sample Precision** | **2-Sample Recall** | **2-Sample F1-Score** | **2-Sample ROC-AUC** | **Mức tăng F1 (+%)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 70.82% | 63.39% | 41.89% | 90.46% | **90.12%** | 92.27% | 88.73% | **90.46%** | 0.9600 | **+27.07%** |
| **Decision Tree** | 70.32% | 62.26% | 39.98% | 90.87% | **91.07%** | 95.98% | 86.72% | **91.12%** | 0.9669 | **+28.86%** |
| **Random Forest** | 71.38% | 62.37% | 40.78% | 91.40% | **91.64%** | 95.45% | 88.38% | **91.78%** | 0.9720 | **+29.41%** |
| **Gradient Boosting** | 72.82% | 66.22% | 47.53% | 92.84% | **92.28%** | 95.04% | **90.09%** | **92.50%** | **0.9776** | **+26.28%** |

### 4.3. Tại sao chuyển từ 4-Sample sang 2-Sample mang lại bước nhảy vọt hiệu năng (+26% đến +29% F1)?
Phân tích Ma trận nhầm lẫn (Confusion Matrix) của mô hình `Random Forest` trên tập Test ($N = 6,519$):
- **Trong cấu hình 4-Sample**:
  - Giữa `Distinction` và `Pass`: Trong 605 sinh viên `Distinction` thực tế, có tới **381 sinh viên (63.0%) bị đoán nhầm thành `Pass`** do cả hai nhóm đều có hành vi tương tác và nộp bài đầy đủ suốt khóa học.
  - Giữa `Fail` và `Withdrawn`: Trong 1,411 sinh viên `Fail` thực tế, có **539 sinh viên (38.2%) bị đoán nhầm thành `Withdrawn`** và **431 sinh viên (30.5%) bị đoán nhầm thành `Pass`**, khiến F1-Score của lớp `Fail` chỉ đạt **40.78%**.
- **Khi chuyển sang cấu hình 2-Sample (`Not At-Risk` vs `At-Risk`)**:
  - Mọi sự nhầm lẫn nội bộ giữa `Distinction <-> Pass` (cùng thuộc nhóm An toàn `0`) và giữa `Fail <-> Withdrawn` (cùng thuộc nhóm Nguy cơ cần can thiệp `1`) được triệt tiêu hợp lý về mặt nghiệp vụ sư phạm!
  - Ma trận nhầm lẫn 2-Sample của `Random Forest` đạt độ phân tách sắc nét: nhận diện đúng **2,932 / 3,077 (95.3%)** sinh viên `Not At-Risk` và phát hiện đúng **3,042 / 3,442 (88.4%)** sinh viên `At-Risk`.

### 4.4. Kết quả đối chuẩn Cảnh báo sớm (Early Warning Snapshots: 25%, 50%, 100%)

| Mốc thời gian dự đoán | Mô hình Baseline | 4-Sample Acc | 4-Sample Macro-F1 | **2-Sample Acc** | **2-Sample Precision** | **2-Sample Recall** | **2-Sample F1** | **2-Sample ROC-AUC** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mốc 25% (~Tuần 9)** | Logistic Regression | 60.88% | 49.68% | **79.35%** | 83.53% | 75.86% | **79.51%** | 0.8770 |
| **Mốc 25% (~Tuần 9)** | Random Forest | 61.44% | 48.93% | **80.89%** | 86.26% | 75.89% | **80.74%** | 0.8896 |
| **Mốc 25% (~Tuần 9)** | Gradient Boosting | 62.11% | 53.99% | **81.32%** | 85.48% | **77.83%** | **81.48%** | **0.8957** |
| **Mốc 50% (~Tuần 18)** | Logistic Regression | 64.60% | 55.45% | **83.40%** | 86.20% | 81.64% | **83.86%** | 0.9171 |
| **Mốc 50% (~Tuần 18)** | Random Forest | 65.39% | 54.10% | **85.41%** | 89.58% | 81.90% | **85.57%** | 0.9309 |
| **Mốc 50% (~Tuần 18)** | Gradient Boosting | 67.07% | 59.82% | **86.33%** | 89.77% | **83.64%** | **86.60%** | **0.9384** |
| **Mốc 100% (Cuối khóa)** | Gradient Boosting | 72.82% | 66.22% | **92.28%** | 95.04% | **90.09%** | **92.50%** | **0.9776** |

> **Ý nghĩa đối với dự án**: Ngay tại **mốc 25% thời lượng khóa học (~Tuần 9)**, khi chỉ sử dụng dữ liệu tương tác và điểm số trước ngày cắt (`date <= 0.25 * L`), cấu hình 2-Sample đã đạt **Recall = 75.89% - 77.83%** và **ROC-AUC = 0.8896 - 0.8957**, vượt ngưỡng mục tiêu ban đầu ($\text{Recall} \ge 75\%$) của Câu hỏi nghiên cứu **RQ2** và thiết lập đường cơ sở (baseline benchmark) vững chắc cho các tuần tiếp theo.

---

## 5. DANH MỤC SẢN PHẨM BÀN GIAO TUẦN 3 (DELIVERABLES CHECKLIST)

- [x] Notebook phân tích khám phá đầy đủ (nhúng sẵn bảng và 11 biểu đồ): [notebooks/02_eda.ipynb](file:///c:/Users/mrore/Project_I/notebooks/02_eda.ipynb)
- [x] Script phân tích EDA & Kiểm định thống kê: [src/eda_analysis.py](file:///c:/Users/mrore/Project_I/src/eda_analysis.py)
- [x] Script thử nghiệm mô hình bài báo cũ chuyển đổi 4-sample sang 2-sample: [src/baseline_4to2_model.py](file:///c:/Users/mrore/Project_I/src/baseline_4to2_model.py)
- [x] Bộ 11 biểu đồ phân tích chất lượng cao (300 DPI) trong `reports/figures/` (nâng tổng số biểu đồ của dự án lên **14 biểu đồ**):
  1. [04_clicks_by_final_result.png](file:///c:/Users/mrore/Project_I/reports/figures/04_clicks_by_final_result.png)
  2. [05_assessment_score_vs_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/05_assessment_score_vs_outcome.png)
  3. [06_submission_timeliness_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/06_submission_timeliness_analysis.png)
  4. [07_temporal_weekly_clicks_trend.png](file:///c:/Users/mrore/Project_I/reports/figures/07_temporal_weekly_clicks_trend.png)
  5. [08_activity_type_by_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/08_activity_type_by_outcome.png)
  6. [09_sedl_dimensions_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/09_sedl_dimensions_comparison.png)
  7. [10_demographics_vs_at_risk.png](file:///c:/Users/mrore/Project_I/reports/figures/10_demographics_vs_at_risk.png)
  8. [11_correlation_matrix_heatmap.png](file:///c:/Users/mrore/Project_I/reports/figures/11_correlation_matrix_heatmap.png)
  9. [12_statistical_tests_effect_size.png](file:///c:/Users/mrore/Project_I/reports/figures/12_statistical_tests_effect_size.png)
  10. [13_baseline_4class_vs_2class_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/13_baseline_4class_vs_2class_comparison.png)
  11. [14_baseline_confusion_matrices_and_roc.png](file:///c:/Users/mrore/Project_I/reports/figures/14_baseline_confusion_matrices_and_roc.png)
- [x] File JSON lưu trữ toàn bộ thông số thống kê & kết quả thực nghiệm: [week3_eda_metrics.json](file:///c:/Users/mrore/Project_I/reports/week3_eda_metrics.json) & [week3_baseline_metrics.json](file:///c:/Users/mrore/Project_I/reports/week3_baseline_metrics.json)
- [x] Báo cáo tổng hợp Tuần 3 (Markdown & LaTeX): [WEEK3_EDA_AND_BASELINE_REPORT.md](file:///c:/Users/mrore/Project_I/reports/WEEK3_EDA_AND_BASELINE_REPORT.md) & [báo_cáo_tuần_3.tex](file:///c:/Users/mrore/Project_I/báo_cáo_tuần_3.tex)
