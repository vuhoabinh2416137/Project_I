# BÁO CÁO KỸ THUẬT TUẦN 6: ĐẶC TRƯNG NÂNG CAO (TEMPORAL, REGULARITY, TRAJECTORY) & HỆ THỐNG CẢNH BÁO SỚM (EARLY WARNING)

**Dự án:** Học máy dự báo Sinh viên có Nguy cơ Bỏ học / Trượt môn (OULAD Learning Analytics)  
**Giai đoạn:** Tuần 6 — Feature Engineering Nâng cao & Tối ưu Ngưỡng Cảnh báo Sớm  
**Mã nguồn chính:** [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb) (lớp `OULADFeatureEngineer` & pipeline Tuần 5–6 tự chủ 100%)  
**Kiểm thử tự động:** Tích hợp trong [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb) (**18/18 PASSED**)

---

## 1. Tóm tắt Kết quả Đạt được (Executive Summary)

Tiếp nối hệ sinh thái 76 đặc trưng nền tảng của Tuần 5 (`week5_expanded`, 98 biến mã hóa), Tuần 6 bổ sung **16 đặc trưng động học thời gian, nhịp độ học tập và quỹ đạo điểm số** (nâng danh mục lên **92 đặc trưng** và ma trận đầu vào `week6_advanced` lên **114 biến mã hóa** sau One-Hot + Ordinal + Missing Flags), đồng thời tích hợp **LightGBM** và chiến lược **Cost-Sensitive + Tinh chỉnh ngưỡng $\tau^*$ thuần trên tập Validation (Zero Test Leakage)** cho bài toán **Early Warning**.

Các cột mốc nổi bật được kiểm chứng trên bộ chia `group_stratified` (0 trùng lặp `id_student` giữa Train/Val/Test):

1. **Sức mạnh áp đảo của Đặc trưng Suy giảm & Vắng mặt tại Mốc sớm 25%:**
   - Trong danh mục 92 đặc trưng tại lát cắt **25% thời lượng khóa**, **`engagement_decay`** vươn lên đứng **#1 toàn cục** theo Random Forest Gini Importance (`0.0671`), vượt qua cả điểm trung bình trọng số (`weighted_avg_score`, `0.0585`) và điểm cao nhất (`max_score`, `0.0538`).
   - Có **7 đặc trưng mới Tuần 6 lọt Top 20** ở mốc 25%: `engagement_decay` (#1, `0.0671`), `days_since_last_activity` (#4, `0.0530`), `clicks_last_2_weeks` (#9, `0.0307`), `consistency_score` (#11, `0.0263`), `max_inactivity_streak` (#14, `0.0226`), `login_regularity` (#18, `0.0157`), `active_weeks` (#20, `0.0138`).
   - Tổng tỷ trọng Importance của 3 nhóm Tuần 6 (`6.1 Temporal & Decay`, `6.2 Regularity & Burst`, `6.3 Score Trajectory`) đạt **29.09%** ở mốc 25%, tăng lên **37.98%** ở mốc 50% và **35.20%** ở mốc 100%.

2. **Đột phá Early Warning Recall $\ge 75\%$ trên Cohort `dynamic_active` ngay từ Mốc 25%:**
   - Dưới ngưỡng mặc định ($\tau = 0.50$), mô hình chuẩn (không cân bằng lớp) bị thiên lệch về Precision do tỷ lệ At-Risk ở Cohort `dynamic_active` 25% là `41.39%` (LightGBM chuẩn chỉ đạt Test Recall = `0.6617`).
   - Với **Cost-Sensitive Learning (`class_weight='balanced'`)** kết hợp tinh chỉnh ngưỡng $\tau^*$ **chỉ trên tập Validation** (tối đa $F_1$ sao cho $\text{Recall}_{\text{val}} \ge 0.76$), **`LightGBM (Balanced)`** tại `25%` (`tau* = 0.43`) đạt:
     - **Test Recall = `0.7713`** (tăng **+10.96 điểm %**, vượt mốc mục tiêu $\ge 75\%$).
     - **Test F1 = `0.7272`** (vượt mốc đề cương $\ge 0.70$), **Test ROC-AUC = `0.8492`**, **Test PR-AUC = `0.8291`**.
   - Tại mốc `50%` (`tau* = 0.48`): **Test Recall = `0.7798`**, **Test F1 = `0.7774`**, **Test ROC-AUC = `0.9070`**.
   - Tại mốc `75%` (`tau* = 0.50`): **Test Recall = `0.8020`**, **Test F1 = `0.8220`**, **Test ROC-AUC = `0.9429`**.
   - Tại mốc `100%` (`tau* = 0.66`): **Test Recall = `0.7765`**, **Test F1 = `0.8466`**, **Test ROC-AUC = `0.9623`**.

3. **Kiểm soát Rò rỉ Dữ liệu & Kiểm thử Tự động (18/18 PASSED):**
   - Tất cả 16 đặc trưng Tuần 6 tính toán nghiêm ngặt trên cửa sổ `date <= T_cutoff` và `date_submitted <= T_cutoff`, loại bỏ hoàn toàn bài thi cuối kỳ `Exam`.
   - Đặc biệt xử lý chuẩn xác **`clicks_trend`** (hệ số góc OLS có thể âm) để **không** bị kẹp `[0, upper]` và `log1p` nhầm trong `prepare_ml_matrices`.
   - Toàn bộ **18/18 unit tests** (12 tests nền Tuần 5 + 6 tests mới Tuần 6) vượt qua 100% trong `43.666s`.

---

## 2. Kiến trúc 16 Đặc trưng Mới Tuần 6 (`week6_advanced`)

Trong phương thức `extract_snapshot_features` của [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb), sau khi tính xong các đặc trưng Tuần 5 và đồng bộ biến `days_before_start = -date_registration`, pipeline gọi 2 phương thức chuyên biệt: `_add_week6_temporal_regularity_features` và `_add_week6_score_trajectory_features`.

| Nhóm | Tên Đặc trưng | Công thức Toán học (Zero Leakage tại $T = T_{\text{cutoff}}$) | Ý nghĩa Sư phạm & Early Warning | RF Imp (25%) | RF Imp (100%) |
|---|---|---|---|---:|---:|
| **6.1 Temporal & Decay** | `clicks_week_1_to_4` | $\sum_{0 \le t \le \min(27, T)} \text{click}_t$ | Lượng tương tác nền 4 tuần đầu khóa học | `0.0076` | `0.0030` |
| **6.1 Temporal & Decay** | `clicks_last_2_weeks` | $\sum_{T - 14 < t \le T} \text{click}_t$ | Động lượng hoạt động ngay sát mốc dự đoán | `0.0307` | `0.0072` |
| **6.1 Temporal & Decay** | `clicks_trend` | $\beta_{\text{OLS}}(\text{weekly\_clicks}_{0..W})$ | Độ dốc tăng/giảm click theo tuần (tính cả tuần 0 click) | `0.0087` | `0.0085` |
| **6.1 Temporal & Decay** | `engagement_decay` | $\text{clicks\_last\_2\_weeks} / \max(1, \text{clicks\_week\_1\_to\_4})$ | Tỷ lệ duy trì/suy giảm tương tác (**#1 tại mốc 25%**) | **`0.0671`** | `0.0127` |
| **6.1 Temporal & Decay** | `days_since_last_activity` | $\max(0, T - \text{last\_click\_day})$ (hoặc $T+1$ nếu chưa từng click) | Số ngày vắng mặt liên tục tính đến mốc $T$ (**#4 tại 25%**) | **`0.0530`** | **`0.0952`** |
| **6.1 Temporal & Decay** | `inactivity_ratio` | $(\lfloor T \rfloor + 1 - \text{active\_days\_in\_course}) / (\lfloor T \rfloor + 1)$ | Tỷ lệ ngày hoàn toàn vắng bóng trong kỳ chính thức | `0.0082` | `0.0127` |
| **6.1 Temporal & Decay** | `max_inactivity_streak` | $\max_i (\Delta t_{\text{inactive}, i})$ trên $[0, \lfloor T \rfloor]$ | Chuỗi ngày "mất tích" liên tiếp dài nhất (**#2 tại 100%**) | `0.0226` | **`0.1000`** |
| **6.1 Temporal & Decay** | `consistency_score` | $-\sum_{w=0}^{W} p_w \ln(p_w) / \ln(W+1)$ | Shannon Entropy chuẩn hóa độ đều đặn theo tuần $[0,1]$ | `0.0263` | `0.0358` |
| **6.2 Regularity & Burst** | `active_weeks` | $|\{w \in [0, W] : \text{weekly\_click}_w > 0\}|$ | Số tuần thực sự tham gia LMS tính đến mốc cắt | `0.0138` | `0.0277` |
| **6.2 Regularity & Burst** | `login_regularity` | $\text{active\_weeks} / (W + 1)$ | Tỷ lệ tuần có hoạt động trên tổng số tuần đã qua $[0,1]$ | `0.0157` | `0.0323` |
| **6.2 Regularity & Burst** | `weekend_clicks` | $\sum_{t \bmod 7 \in \{5, 6\}} \text{click}_t$ | Tổng lượt click vào ngày nghỉ cuối tuần | `0.0056` | `0.0027` |
| **6.2 Regularity & Burst** | `weekend_ratio` | $\text{weekend\_clicks} / \text{total\_clicks}$ | Tỷ trọng học tập cuối tuần (bổ sung cho SV đi làm) | `0.0061` | `0.0022` |
| **6.2 Regularity & Burst** | `burst_count` | $|\{d : c_d > \bar{c} + 2s_c\}|$ (`active_days >= 3`) | Số ngày học dồn bất thường (cramming spikes) | `0.0017` | `0.0035` |
| **6.2 Regularity & Burst** | `burst_ratio` | $\text{burst\_count} / \text{active\_days}$ | Mức độ phụ thuộc vào các phiên học dồn nén $[0,1]$ | `0.0056` | `0.0020` |
| **6.3 Score Trajectory** | `last_minus_first_score` | $s_{\text{last}} - s_{\text{first}}$ (`0.0` nếu nộp $<2$ bài) | Xu hướng biến thiên điểm giữa bài gần nhất và bài đầu | `0.0083` | `0.0029` |
| **6.3 Score Trajectory** | `score_trend_slope` | $\beta_{\text{OLS}}(s_0, \dots, s_{m-1})$ (`0.0` nếu $m < 2$) | Độ dốc tiến bộ hay sa sút điểm số qua các bài tập | `0.0098` | `0.0037` |

*(Ghi chú kỹ thuật: Trong ma trận `week6_advanced`, biến `days_before_start = -date_registration` được chuẩn hóa từ giá trị trung vị `Train` của `date_registration`, thay thế `date_registration` gốc để giữ nguyên không gian 84 biến số + 3 biến ordinal + 24 cột One-Hot + 3 cờ thiếu dữ liệu = **114 biến mã hóa**, danh mục tổng **92 đặc trưng** tại [`feature_catalog_week6.csv`](file:///c:/Users/mrore/Project_I/data/processed/feature_catalog_week6.csv)).*

---

## 3. Phân tích Chuyển biến Đặc trưng qua Thời gian & Tín hiệu Sớm

Trích xuất từ [`week6_experiment_metrics.json`](file:///c:/Users/mrore/Project_I/data/processed/week6_experiment_metrics.json) trên Cohort thực tế **`dynamic_active`** (loại sinh viên đã rút trước/tại mốc cắt):

| Mốc Snapshot | Quy mô `dynamic_active` | Tỷ lệ At-Risk | `days_since_last` (0 vs 1) | `max_inactivity_streak` (0 vs 1) | `consistency_score` (0 vs 1) | `login_regularity` (0 vs 1) | `clicks_trend` (0 vs 1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **25%** | 26,248 | 41.39% | 3.97 vs **15.04** ngày | 10.60 vs **21.43** ngày | 0.7732 vs **0.5792** | 0.8497 vs **0.6354** | -5.473 vs -4.135 |
| **50%** | 24,601 | 37.46% | 4.26 vs **38.10** ngày | 17.06 vs **47.99** ngày | 0.7818 vs **0.5417** | 0.8053 vs **0.5053** | -2.715 vs -2.311 |
| **75%** | 23,159 | 33.57% | 6.31 vs **75.39** ngày | 19.56 vs **83.32** ngày | 0.7978 vs **0.5063** | 0.8032 vs **0.4190** | -1.139 vs -1.322 |
| **100%** | 22,522 | 31.69% | 14.22 vs **121.33** ngày | 25.12 vs **129.04** ngày | 0.7820 vs **0.4658** | 0.7593 vs **0.3389** | -1.038 vs -0.893 |

**Phát hiện chuyên sâu (Domain Insights):**
- **Tại sao `clicks_trend` âm ở cả nhóm Đạt nhưng `engagement_decay` lại xếp #1?** Nhóm `Distinction`/`Pass` bùng nổ rất mạnh ngay tuần 0–3 (trung bình ~339 clicks ở 4 tuần đầu), nên khi nhịp học trở về mức ổn định, độ dốc tuyệt đối (`clicks/tuần`) âm sâu hơn về mặt trị số (`-5.47` vs `-4.14`). Ngược lại, `engagement_decay` đo **tỷ lệ tương đối** so với chính sinh viên đó: sinh viên Đạt giữ được nhịp tương tác cuối cửa sổ, còn sinh viên Nguy cơ suy giảm về `0` (trung vị `0.0`), tạo ranh giới phân tách cực sắc bén cho mô hình cây (`RF`, `LightGBM`, `GBDT`).
- **Quy luật Ngưỡng Vắng mặt (`max_inactivity_streak`):** Ở mốc 25% `dynamic_active` (Hình 27D), tỷ lệ At-Risk tăng đơn điệu cực mạnh theo chuỗi ngày vắng mặt: **16.2%** (0–3 ngày) $\rightarrow$ **26.5%** (4–7 ngày) $\rightarrow$ **38.7%** (8–14 ngày) $\rightarrow$ **54.5%** (15–30 ngày) $\rightarrow$ **78.9%** ($>30$ ngày). Chỉ cần sinh viên vắng mặt $>14$ ngày liên tiếp ngay trong 1/4 chặng đường đầu, xác suất rơi vào nhóm nguy cơ đã vượt ngưỡng báo động đỏ $>54\%$.

---

## 4. Kết quả Thực nghiệm Đối sánh Mô hình (`week5_expanded` vs `week6_advanced`)

Tất cả mô hình đều được đánh giá trên tập **Test độc lập** dưới chiến lược **`group_stratified`** (0 trùng lặp sinh viên giữa Train/Val/Test).

### 4.1. Bảng Đối sánh Mô hình Chuẩn ($\tau = 0.50$) trên Cohort `dynamic_active`

| Snapshot | Bộ Đặc trưng | Mô hình | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **25%** | `week5_expanded` (98) | Logistic Regression | 0.8449 | 0.8227 | 0.7097 | 0.6418 | 0.7936 | 0.1526 |
| **25%** | `week6_advanced` (114) | Logistic Regression | 0.8472 | 0.8265 | 0.7148 | 0.6501 | 0.7939 | 0.1510 |
| **25%** | `week5_expanded` (98) | Random Forest | 0.8411 | 0.8176 | 0.6899 | 0.6134 | 0.7881 | 0.1556 |
| **25%** | `week6_advanced` (114) | Random Forest | 0.8431 | 0.8235 | 0.6896 | 0.6076 | 0.7971 | 0.1541 |
| **25%** | `week5_expanded` (98) | Gradient Boosting | 0.8459 | 0.8239 | 0.7073 | 0.6424 | 0.7869 | 0.1527 |
| **25%** | `week6_advanced` (114) | Gradient Boosting | 0.8509 | 0.8313 | 0.7153 | 0.6572 | 0.7846 | 0.1498 |
| **25%** | `week5_expanded` (98) | LightGBM | 0.8473 | 0.8250 | 0.7140 | 0.6572 | 0.7816 | 0.1521 |
| **25%** | **`week6_advanced` (114)** | **LightGBM** | **0.8515** | **0.8323** | **0.7172** | **0.6617** | 0.7828 | **0.1498** |
| **50%** | `week5_expanded` (98) | LightGBM | 0.9002 | 0.8746 | 0.7612 | 0.7031 | 0.8297 | 0.1170 |
| **50%** | **`week6_advanced` (114)** | **LightGBM** | **0.9070** | **0.8816** | **0.7688** | **0.7145** | 0.8320 | **0.1141** |
| **75%** | `week5_expanded` (98) | LightGBM | 0.9392 | 0.9169 | 0.8147 | 0.7480 | 0.8945 | 0.0831 |
| **75%** | **`week6_advanced` (114)** | **LightGBM** | **0.9431** | **0.9214** | **0.8266** | **0.7597** | **0.9066** | **0.0809** |
| **100%** | `week5_expanded` (98) | LightGBM | 0.9577 | 0.9375 | 0.8488 | 0.7814 | 0.9289 | 0.0669 |
| **100%** | **`week6_advanced` (114)** | **LightGBM** | **0.9626** | **0.9441** | **0.8541** | **0.7863** | **0.9347** | **0.0630** |

### 4.2. Tổng hợp Toàn cục trên `Full Cohort` (`N = 32,593`) và `Pre-Course Active` (`N = 29,915`) với `week6_advanced`

| Cohort | Snapshot | Mô hình tốt nhất | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Brier |
|---|---|---|---:|---:|---:|---:|---:|
| **`full`** (`N=32,593`) | 25% | LightGBM / GBDT | 0.9004 / **0.9010** | 0.9268 / **0.9269** | **0.8260** / 0.8239 | **0.7868** / 0.7811 | 0.1240 / **0.1239** |
| **`full`** (`N=32,593`) | 50% | Gradient Boosting | **0.9484** | **0.9625** | **0.8822** | **0.8527** | **0.0866** |
| **`full`** (`N=32,593`) | 75% | Gradient Boosting | **0.9741** | **0.9809** | **0.9257** | **0.8975** | **0.0583** |
| **`full`** (`N=32,593`) | 100% | Gradient Boosting / LGBM | **0.9844** / 0.9843 | 0.9884 / **0.9885** | 0.9415 / **0.9442** | 0.9142 / **0.9158** | 0.0437 / **0.0434** |
| **`pre_course_active`** (`N=29,915`) | 25% | Gradient Boosting / LGBM | **0.8761** / 0.8753 | 0.8935 / **0.8941** | **0.7790** / 0.7783 | **0.7182** / 0.7177 | **0.1385** |
| **`pre_course_active`** (`N=29,915`) | 100% | LightGBM | **0.9809** | **0.9838** | **0.9306** | **0.8984** | **0.0483** |

---

## 5. Đột phá Early Warning: Cost-Sensitive + Validation-Tuned Threshold

Trong ứng dụng Cảnh báo Sớm giáo dục, **bỏ sót sinh viên có nguy cơ (False Negative)** đắt giá hơn nhiều so với cảnh báo nhầm nhẹ (gửi email nhắc nhở). Pipeline tối ưu hóa ngưỡng $\tau^*$ **hoàn toàn trên tập Validation** theo nguyên tắc:
$$\tau^* = \arg\max_{\tau \in [0.15, 0.70]} F_1^{\text{val}}(\tau) \quad \text{sao cho} \quad \text{Recall}^{\text{val}}(\tau) \ge 0.76$$
sau đó đóng băng $\tau^*$ để chấm điểm một lần duy nhất trên tập Test (Zero Test Leakage):

| Snapshot | Cohort | Mô hình (`week6_advanced`) | Default $\tau=0.50$ Recall | Tuned $\tau^*$ (Val) | Val Recall ($\tau^*$) | **Test Recall** | Test Precision | **Test F1** | Test ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **25%** | `dynamic_active` | Logistic Regression (Balanced) | 0.7146 | 0.44 | 0.7668 | **0.7713** | 0.6919 | **0.7294** | 0.8475 |
| **25%** | `dynamic_active` | Random Forest (Balanced) | 0.6688 | 0.40 | 0.7771 | **0.7751** | 0.6755 | **0.7219** | 0.8441 |
| **25%** | `dynamic_active` | **LightGBM (Balanced)** | 0.7146 | **0.43** | 0.7790 | **0.7713** | 0.6879 | **0.7272** | **0.8492** |
| **50%** | `dynamic_active` | **LightGBM (Balanced)** | 0.7699 | **0.48** | 0.7844 | **0.7798** | 0.7751 | **0.7774** | **0.9070** |
| **75%** | `dynamic_active` | Random Forest (Balanced) | 0.7642 | 0.41 | 0.8209 | **0.8227** | 0.8487 | **0.8355** | **0.9440** |
| **75%** | `dynamic_active` | **LightGBM (Balanced)** | 0.8020 | **0.50** | 0.8056 | **0.8020** | 0.8430 | **0.8220** | **0.9429** |
| **100%** | `dynamic_active` | **LightGBM (Balanced)** | 0.8294 | **0.66** | 0.8165 | **0.7765** | 0.9307 | **0.8466** | **0.9623** |
| **25%** | `pre_course_active` | LightGBM (Balanced) | 0.7259 | 0.38 | 0.8246 | **0.7924** | 0.7709 | **0.7815** | **0.8758** |
| **25%** | `full` | LightGBM (Balanced) | 0.7681 | 0.46 | 0.8024 | **0.7950** | 0.8627 | **0.8274** | **0.9005** |

Tất cả các mô hình Early Warning ngay từ **mốc 25% trên `dynamic_active`** đều vượt xa đồng thời 3 tiêu chuẩn: **AUC $\ge 0.849$ (mục tiêu $\ge 0.80$)**, **Recall $\ge 0.771$ (mục tiêu $\ge 0.75$)** và **F1 $\ge 0.727$ (mục tiêu $\ge 0.70$)**.

---

## 6. Hệ thống Biểu đồ & Kiểm thử Tự động (Verification)

- **5 Biểu đồ phân tích Tuần 6 (300 DPI):**
  - [`26_temporal_trend_and_engagement_decay.png`](file:///c:/Users/mrore/Project_I/reports/figures/26_temporal_trend_and_engagement_decay.png)
  - [`27_inactivity_gap_and_consistency_analysis.png`](file:///c:/Users/mrore/Project_I/reports/figures/27_inactivity_gap_and_consistency_analysis.png)
  - [`28_regularity_weekend_and_burst_patterns.png`](file:///c:/Users/mrore/Project_I/reports/figures/28_regularity_weekend_and_burst_patterns.png)
  - [`29_week6_early_warning_breakthrough_25_50_75_100.png`](file:///c:/Users/mrore/Project_I/reports/figures/29_week6_early_warning_breakthrough_25_50_75_100.png)
  - [`30_week6_feature_importance_evolution.png`](file:///c:/Users/mrore/Project_I/reports/figures/30_week6_feature_importance_evolution.png)
- **Kiểm thử Unit Tests:** Bộ kiểm thử tích hợp trong [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb) — **18/18 PASSED**, xác nhận 100% Zero Temporal Leakage, miền giá trị toán học, kích thước ma trận 114 biến, danh mục 92 đặc trưng và Test Recall $\ge 0.75$ tại mốc 25% `dynamic_active`.
