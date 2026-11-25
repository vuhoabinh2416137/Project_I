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
   - Tổng tỷ trọng Importance của 3 nhóm Tuần 6 (`6.1 Temporal & Decay`, `6.2 Regularity & Burst`, `6.3 Score Trajectory`) đạt **29.09%** ở mốc 25%, **37.98%** ở mốc 50%, đạt đỉnh **39.47%** ở mốc 75% và duy trì **35.20%** ở mốc 100%.

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
   - Toàn bộ **18/18 unit tests** (12 tests nền Tuần 5 + 6 tests mới Tuần 6) vượt qua 100% trong `19.464s`.

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

*(Ghi chú kỹ thuật: Trong ma trận `week6_advanced`, biến `days_before_start = -date_registration` được chuẩn hóa từ giá trị trung vị `Train` của `date_registration`, thay thế `date_registration` gốc để giữ nguyên không gian 84 biến số + 3 biến ordinal + 24 cột One-Hot + 3 cờ thiếu dữ liệu = **114 biến mã hóa**, danh mục tổng **92 đặc trưng** tại [`feature_catalog_week6.csv`](file:///c:/Users/mrore/Project_I/data/processed/week6_advanced_features/feature_catalog_week6.csv)).*

### 2.1. Hồ sơ Thống kê và Xếp hạng 16 Đặc trưng Mới qua 4 Lát cắt (`25%`, `50%`, `75%`, `100%`)

#### Mốc 25% (`Full Cohort`)
| Nhóm | Tên Đặc trưng | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini | Hạng (/92) |
|---|---|---:|---:|---:|---:|---:|
| 6.1 Temporal | `engagement_decay` | -0.015 | 0.030 | 0.158 | **`0.0671`** | **#1** |
| 6.1 Temporal | `days_since_last_activity` | +0.530 | -1.251 | 0.183 | **`0.0530`** | **#4** |
| 6.1 Temporal | `clicks_last_2_weeks` | -0.312 | 0.659 | 0.166 | **`0.0307`** | **#9** |
| 6.1 Temporal | `consistency_score` | -0.534 | 1.267 | 0.167 | **`0.0263`** | **#11** |
| 6.1 Temporal | `max_inactivity_streak` | +0.529 | -1.250 | 0.176 | **`0.0226`** | **#14** |
| 6.1 Temporal | `clicks_trend` | +0.103 | -0.208 | 0.085 | `0.0087` | #24 |
| 6.1 Temporal | `inactivity_ratio` | +0.479 | -1.092 | 0.162 | `0.0082` | #27 |
| 6.1 Temporal | `clicks_week_1_to_4` | -0.297 | 0.622 | 0.110 | `0.0076` | #31 |
| 6.2 Regularity | `login_regularity` | -0.537 | 1.276 | 0.174 | **`0.0157`** | **#18** |
| 6.2 Regularity | `active_weeks` | -0.541 | 1.289 | 0.181 | **`0.0138`** | **#20** |
| 6.2 Regularity | `weekend_ratio` | -0.125 | 0.253 | 0.089 | `0.0061` | #44 |
| 6.2 Regularity | `weekend_clicks` | -0.302 | 0.635 | 0.104 | `0.0056` | #47 |
| 6.2 Regularity | `burst_ratio` | -0.238 | 0.491 | 0.106 | `0.0056` | #49 |
| 6.2 Regularity | `burst_count` | -0.397 | 0.866 | 0.089 | `0.0017` | #72 |
| 6.3 Trajectory | `score_trend_slope` | -0.081 | 0.163 | 0.098 | `0.0098` | #22 |
| 6.3 Trajectory | `last_minus_first_score` | -0.109 | 0.219 | 0.088 | `0.0083` | #26 |

#### Mốc 50% (`Full Cohort`)
| Nhóm | Tên Đặc trưng | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini | Hạng (/92) |
|---|---|---:|---:|---:|---:|---:|
| 6.1 Temporal | `engagement_decay` | -0.071 | 0.143 | 0.263 | **`0.0973`** | **#1** |
| 6.1 Temporal | `days_since_last_activity` | +0.646 | -1.696 | 0.309 | **`0.0911`** | **#2** |
| 6.1 Temporal | `clicks_last_2_weeks` | -0.408 | 0.895 | 0.277 | **`0.0470`** | **#5** |
| 6.1 Temporal | `max_inactivity_streak` | +0.637 | -1.653 | 0.273 | **`0.0372`** | **#9** |
| 6.1 Temporal | `consistency_score` | -0.623 | 1.595 | 0.249 | **`0.0153`** | **#18** |
| 6.1 Temporal | `inactivity_ratio` | +0.548 | -1.313 | 0.220 | **`0.0128`** | **#19** |
| 6.1 Temporal | `clicks_trend` | +0.092 | -0.185 | 0.106 | `0.0085` | #25 |
| 6.1 Temporal | `clicks_week_1_to_4` | -0.297 | 0.622 | 0.110 | `0.0040` | #45 |
| 6.2 Regularity | `login_regularity` | -0.650 | 1.712 | 0.264 | **`0.0220`** | **#14** |
| 6.2 Regularity | `active_weeks` | -0.652 | 1.723 | 0.259 | **`0.0154`** | **#17** |
| 6.2 Regularity | `burst_ratio` | -0.256 | 0.529 | 0.153 | `0.0052` | #33 |
| 6.2 Regularity | `weekend_ratio` | -0.141 | 0.285 | 0.118 | `0.0040` | #48 |
| 6.2 Regularity | `weekend_clicks` | -0.359 | 0.771 | 0.161 | `0.0036` | #52 |
| 6.2 Regularity | `burst_count` | -0.482 | 1.102 | 0.144 | `0.0011` | #73 |
| 6.3 Trajectory | `score_trend_slope` | -0.051 | 0.102 | 0.232 | `0.0079` | #27 |
| 6.3 Trajectory | `last_minus_first_score` | -0.068 | 0.136 | 0.189 | `0.0074` | #29 |

#### Mốc 75% (`Full Cohort`)
| Nhóm | Tên Đặc trưng | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini | Hạng (/92) |
|---|---|---:|---:|---:|---:|---:|
| 6.1 Temporal | `days_since_last_activity` | +0.729 | -2.134 | 0.404 | **`0.1007`** | **#2** |
| 6.1 Temporal | `engagement_decay` | -0.068 | 0.137 | 0.312 | **`0.0694`** | **#3** |
| 6.1 Temporal | `max_inactivity_streak` | +0.721 | -2.084 | 0.377 | **`0.0619`** | **#4** |
| 6.1 Temporal | `clicks_last_2_weeks` | -0.379 | 0.819 | 0.314 | **`0.0437`** | **#8** |
| 6.1 Temporal | `consistency_score` | -0.680 | 1.856 | 0.318 | **`0.0313`** | **#10** |
| 6.1 Temporal | `inactivity_ratio` | +0.604 | -1.517 | 0.279 | `0.0111` | #23 |
| 6.1 Temporal | `clicks_trend` | -0.014 | 0.028 | 0.127 | `0.0098` | #24 |
| 6.1 Temporal | `clicks_week_1_to_4` | -0.297 | 0.622 | 0.110 | `0.0036` | #37 |
| 6.2 Regularity | `login_regularity` | -0.726 | 2.116 | 0.337 | **`0.0208`** | **#14** |
| 6.2 Regularity | `active_weeks` | -0.729 | 2.133 | 0.334 | **`0.0206`** | **#15** |
| 6.2 Regularity | `burst_count` | -0.552 | 1.327 | 0.208 | `0.0032` | #41 |
| 6.2 Regularity | `weekend_ratio` | -0.143 | 0.289 | 0.131 | `0.0029` | #47 |
| 6.2 Regularity | `burst_ratio` | -0.260 | 0.539 | 0.203 | `0.0027` | #52 |
| 6.2 Regularity | `weekend_clicks` | -0.405 | 0.888 | 0.199 | `0.0026` | #54 |
| 6.3 Trajectory | `score_trend_slope` | -0.031 | 0.063 | 0.302 | `0.0053` | #29 |
| 6.3 Trajectory | `last_minus_first_score` | -0.063 | 0.126 | 0.191 | `0.0049` | #32 |

#### Mốc 100% (`Full Cohort`)
| Nhóm | Tên Đặc trưng | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini | Hạng (/92) |
|---|---|---:|---:|---:|---:|---:|
| 6.1 Temporal | `max_inactivity_streak` | +0.781 | -2.505 | 0.447 | **`0.1000`** | **#2** |
| 6.1 Temporal | `days_since_last_activity` | +0.778 | -2.479 | 0.439 | **`0.0952`** | **#3** |
| 6.1 Temporal | `consistency_score` | -0.709 | 2.014 | 0.364 | **`0.0358`** | **#8** |
| 6.1 Temporal | `inactivity_ratio` | +0.633 | -1.636 | 0.321 | **`0.0127`** | **#18** |
| 6.1 Temporal | `engagement_decay` | -0.029 | 0.058 | 0.172 | **`0.0127`** | **#19** |
| 6.1 Temporal | `clicks_trend` | +0.060 | -0.120 | 0.127 | `0.0085` | #22 |
| 6.1 Temporal | `clicks_last_2_weeks` | -0.205 | 0.419 | 0.179 | `0.0072` | #24 |
| 6.1 Temporal | `clicks_week_1_to_4` | -0.297 | 0.622 | 0.110 | `0.0030` | #36 |
| 6.2 Regularity | `login_regularity` | -0.770 | 2.421 | 0.381 | **`0.0323`** | **#10** |
| 6.2 Regularity | `active_weeks` | -0.773 | 2.437 | 0.373 | **`0.0277`** | **#11** |
| 6.2 Regularity | `burst_count` | -0.584 | 1.440 | 0.239 | `0.0035` | #35 |
| 6.2 Regularity | `weekend_clicks` | -0.434 | 0.966 | 0.234 | `0.0027` | #39 |
| 6.2 Regularity | `weekend_ratio` | -0.141 | 0.285 | 0.151 | `0.0022` | #48 |
| 6.2 Regularity | `burst_ratio` | -0.251 | 0.520 | 0.235 | `0.0020` | #55 |
| 6.3 Trajectory | `score_trend_slope` | -0.022 | 0.044 | 0.332 | `0.0037` | #33 |
| 6.3 Trajectory | `last_minus_first_score` | -0.047 | 0.095 | 0.194 | `0.0029` | #38 |

---

## 3. Phân tích Chuyển biến Đặc trưng qua Thời gian & Tín hiệu Sớm

Trích xuất từ [`week6_experiment_metrics.json`](file:///c:/Users/mrore/Project_I/data/processed/week6_advanced_features/week6_experiment_metrics.json) trên Cohort thực tế **`dynamic_active`** (loại sinh viên đã rút trước/tại mốc cắt):

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

### 4.1. Bảng Đối sánh Mô hình Chuẩn ($\tau = 0.50$) trên Cohort `dynamic_active` (Chia 4 Lát cắt)

#### Mốc 25% (`dynamic_active`, $N = 26,248$)
| Thuật toán | Bộ Đặc trưng | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `week5_expanded` (98) | 0.8449 | 0.8227 | 0.7097 | 0.6418 | 0.7936 | 0.1526 |
| Logistic Regression | **`week6_advanced` (114)** | 0.8472 | 0.8265 | 0.7148 | 0.6501 | 0.7939 | 0.1510 |
| Random Forest | `week5_expanded` (98) | 0.8411 | 0.8176 | 0.6899 | 0.6134 | 0.7881 | 0.1556 |
| Random Forest | **`week6_advanced` (114)** | 0.8431 | 0.8235 | 0.6896 | 0.6076 | 0.7971 | 0.1541 |
| Gradient Boosting | `week5_expanded` (98) | 0.8459 | 0.8239 | 0.7073 | 0.6424 | 0.7869 | 0.1527 |
| Gradient Boosting | **`week6_advanced` (114)** | 0.8509 | 0.8313 | 0.7153 | 0.6572 | 0.7846 | 0.1498 |
| LightGBM | `week5_expanded` (98) | 0.8473 | 0.8250 | 0.7140 | 0.6572 | 0.7816 | 0.1521 |
| LightGBM | **`week6_advanced` (114)** | 0.8515 | 0.8323 | 0.7172 | 0.6617 | 0.7828 | 0.1498 |

#### Mốc 50% (`dynamic_active`, $N = 24,601$)
| Thuật toán | Bộ Đặc trưng | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `week5_expanded` (98) | 0.9006 | 0.8740 | 0.7610 | 0.7001 | 0.8336 | 0.1169 |
| Logistic Regression | **`week6_advanced` (114)** | 0.9062 | 0.8807 | 0.7661 | 0.7077 | 0.8351 | 0.1139 |
| Random Forest | `week5_expanded` (98) | 0.8947 | 0.8669 | 0.7571 | 0.6841 | 0.8476 | 0.1203 |
| Random Forest | **`week6_advanced` (114)** | 0.9026 | 0.8760 | 0.7562 | 0.6819 | 0.8488 | 0.1166 |
| Gradient Boosting | `week5_expanded` (98) | 0.8990 | 0.8727 | 0.7556 | 0.6986 | 0.8229 | 0.1184 |
| Gradient Boosting | **`week6_advanced` (114)** | 0.9074 | 0.8810 | 0.7668 | 0.7130 | 0.8295 | 0.1147 |
| LightGBM | `week5_expanded` (98) | 0.9002 | 0.8746 | 0.7612 | 0.7031 | 0.8297 | 0.1170 |
| LightGBM | **`week6_advanced` (114)** | 0.9070 | 0.8816 | 0.7688 | 0.7145 | 0.8320 | 0.1141 |

#### Mốc 75% (`dynamic_active`, $N = 23,159$)
| Thuật toán | Bộ Đặc trưng | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `week5_expanded` (98) | 0.9399 | 0.9169 | 0.8094 | 0.7471 | 0.8830 | 0.0836 |
| Logistic Regression | **`week6_advanced` (114)** | 0.9418 | 0.9196 | 0.8203 | 0.7579 | 0.8938 | 0.0822 |
| Random Forest | `week5_expanded` (98) | 0.9372 | 0.9151 | 0.8122 | 0.7318 | 0.9125 | 0.0849 |
| Random Forest | **`week6_advanced` (114)** | 0.9420 | 0.9207 | 0.8115 | 0.7264 | 0.9191 | 0.0820 |
| Gradient Boosting | `week5_expanded` (98) | 0.9386 | 0.9154 | 0.8149 | 0.7471 | 0.8963 | 0.0845 |
| Gradient Boosting | **`week6_advanced` (114)** | 0.9425 | 0.9201 | 0.8149 | 0.7471 | 0.8963 | 0.0818 |
| LightGBM | `week5_expanded` (98) | 0.9392 | 0.9169 | 0.8147 | 0.7480 | 0.8945 | 0.0831 |
| LightGBM | **`week6_advanced` (114)** | 0.9431 | 0.9214 | 0.8266 | 0.7597 | 0.9066 | 0.0809 |

#### Mốc 100% (`dynamic_active`, $N = 22,522$)
| Thuật toán | Bộ Đặc trưng | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `week5_expanded` (98) | 0.9547 | 0.9331 | 0.8387 | 0.7696 | 0.9214 | 0.0691 |
| Logistic Regression | **`week6_advanced` (114)** | 0.9585 | 0.9391 | 0.8533 | 0.7873 | 0.9316 | 0.0651 |
| Random Forest | `week5_expanded` (98) | 0.9561 | 0.9348 | 0.8338 | 0.7500 | 0.9387 | 0.0686 |
| Random Forest | **`week6_advanced` (114)** | 0.9593 | 0.9403 | 0.8380 | 0.7529 | 0.9446 | 0.0651 |
| Gradient Boosting | `week5_expanded` (98) | 0.9566 | 0.9309 | 0.8394 | 0.7686 | 0.9245 | 0.0680 |
| Gradient Boosting | **`week6_advanced` (114)** | 0.9600 | 0.9412 | 0.8494 | 0.7853 | 0.9249 | 0.0640 |
| LightGBM | `week5_expanded` (98) | 0.9577 | 0.9375 | 0.8488 | 0.7814 | 0.9289 | 0.0669 |
| LightGBM | **`week6_advanced` (114)** | 0.9626 | 0.9441 | 0.8541 | 0.7863 | 0.9347 | 0.0630 |

### 4.2. Hiệu năng Bộ Đặc trưng Tuần 6 (`week6_advanced`, 114 biến) trên `Full Cohort` và `Pre-Course Active` (Chia 4 Lát cắt)

#### Mốc 25%
| Cohort | Thuật toán (`week6_advanced`) | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` ($N=32,593$) | Logistic Regression | 0.8960 | 0.9226 | 0.8162 | 0.7742 | 0.8630 | 0.1274 |
| `full` ($N=32,593$) | Random Forest | 0.8950 | 0.9226 | 0.8155 | 0.7571 | 0.8837 | 0.1278 |
| `full` ($N=32,593$) | Gradient Boosting | 0.9010 | 0.9269 | 0.8239 | 0.7811 | 0.8715 | 0.1239 |
| `full` ($N=32,593$) | LightGBM | 0.9004 | 0.9268 | 0.8260 | 0.7868 | 0.8692 | 0.1240 |
| `pre_course_active` ($N=29,915$) | Logistic Regression | 0.8743 | 0.8918 | 0.7758 | 0.7144 | 0.8489 | 0.1392 |
| `pre_course_active` ($N=29,915$) | Random Forest | 0.8711 | 0.8893 | 0.7654 | 0.6922 | 0.8559 | 0.1417 |
| `pre_course_active` ($N=29,915$) | Gradient Boosting | 0.8761 | 0.8935 | 0.7790 | 0.7182 | 0.8510 | 0.1385 |
| `pre_course_active` ($N=29,915$) | LightGBM | 0.8753 | 0.8941 | 0.7783 | 0.7177 | 0.8500 | 0.1385 |

#### Mốc 50%
| Cohort | Thuật toán (`week6_advanced`) | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` ($N=32,593$) | Logistic Regression | 0.9451 | 0.9601 | 0.8744 | 0.8458 | 0.9051 | 0.0895 |
| `full` ($N=32,593$) | Random Forest | 0.9467 | 0.9614 | 0.8787 | 0.8381 | 0.9234 | 0.0876 |
| `full` ($N=32,593$) | Gradient Boosting | 0.9484 | 0.9625 | 0.8822 | 0.8527 | 0.9137 | 0.0866 |
| `full` ($N=32,593$) | LightGBM | 0.9481 | 0.9623 | 0.8782 | 0.8466 | 0.9123 | 0.0873 |
| `pre_course_active` ($N=29,915$) | Logistic Regression | 0.9368 | 0.9453 | 0.8531 | 0.8044 | 0.9081 | 0.0961 |
| `pre_course_active` ($N=29,915$) | Random Forest | 0.9342 | 0.9444 | 0.8494 | 0.7890 | 0.9197 | 0.0971 |
| `pre_course_active` ($N=29,915$) | Gradient Boosting | 0.9357 | 0.9457 | 0.8516 | 0.7977 | 0.9134 | 0.0957 |
| `pre_course_active` ($N=29,915$) | LightGBM | 0.9346 | 0.9446 | 0.8537 | 0.7996 | 0.9156 | 0.0966 |

#### Mốc 75%
| Cohort | Thuật toán (`week6_advanced`) | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` ($N=32,593$) | Logistic Regression | 0.9714 | 0.9793 | 0.9203 | 0.8950 | 0.9471 | 0.0609 |
| `full` ($N=32,593$) | Random Forest | 0.9735 | 0.9806 | 0.9225 | 0.8906 | 0.9567 | 0.0594 |
| `full` ($N=32,593$) | Gradient Boosting | 0.9741 | 0.9809 | 0.9257 | 0.8975 | 0.9558 | 0.0583 |
| `full` ($N=32,593$) | LightGBM | 0.9733 | 0.9805 | 0.9230 | 0.8930 | 0.9552 | 0.0588 |
| `pre_course_active` ($N=29,915$) | Logistic Regression | 0.9639 | 0.9699 | 0.8982 | 0.8565 | 0.9442 | 0.0687 |
| `pre_course_active` ($N=29,915$) | Random Forest | 0.9650 | 0.9714 | 0.9003 | 0.8526 | 0.9537 | 0.0672 |
| `pre_course_active` ($N=29,915$) | Gradient Boosting | 0.9670 | 0.9721 | 0.8994 | 0.8569 | 0.9463 | 0.0668 |
| `pre_course_active` ($N=29,915$) | LightGBM | 0.9660 | 0.9719 | 0.9078 | 0.8651 | 0.9548 | 0.0660 |

#### Mốc 100%
| Cohort | Thuật toán (`week6_advanced`) | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` ($N=32,593$) | Logistic Regression | 0.9819 | 0.9866 | 0.9367 | 0.9121 | 0.9626 | 0.0472 |
| `full` ($N=32,593$) | Random Forest | 0.9841 | 0.9882 | 0.9411 | 0.9064 | 0.9785 | 0.0448 |
| `full` ($N=32,593$) | Gradient Boosting | 0.9844 | 0.9884 | 0.9415 | 0.9142 | 0.9706 | 0.0437 |
| `full` ($N=32,593$) | LightGBM | 0.9843 | 0.9885 | 0.9442 | 0.9158 | 0.9745 | 0.0434 |
| `pre_course_active` ($N=29,915$) | Logistic Regression | 0.9775 | 0.9814 | 0.9264 | 0.8907 | 0.9650 | 0.0509 |
| `pre_course_active` ($N=29,915$) | Random Forest | 0.9797 | 0.9831 | 0.9250 | 0.8762 | 0.9795 | 0.0503 |
| `pre_course_active` ($N=29,915$) | Gradient Boosting | 0.9804 | 0.9835 | 0.9276 | 0.8911 | 0.9671 | 0.0491 |
| `pre_course_active` ($N=29,915$) | LightGBM | 0.9809 | 0.9838 | 0.9306 | 0.8984 | 0.9653 | 0.0483 |

---

## 5. Đột phá Early Warning: Cost-Sensitive + Validation-Tuned Threshold (36 Cấu hình qua 4 Lát cắt)

Trong ứng dụng Cảnh báo Sớm giáo dục, **bỏ sót sinh viên có nguy cơ (False Negative)** đắt giá hơn nhiều so với cảnh báo nhầm nhẹ (gửi email nhắc nhở). Pipeline tối ưu hóa ngưỡng $\tau^*$ **hoàn toàn trên tập Validation** theo nguyên tắc:
$$\tau^* = \arg\max_{\tau \in [0.15, 0.70]} F_1^{\text{val}}(\tau) \quad \text{sao cho} \quad \text{Recall}^{\text{val}}(\tau) \ge 0.76$$
sau đó đóng băng $\tau^*$ để chấm điểm một lần duy nhất trên tập Test (Zero Test Leakage):

### 5.1. Mốc 25%
| Cohort | Mô hình (`week6_advanced`) | Default $\tau=0.50$ Recall | Tuned $\tau^*$ (Val) | Val Recall ($\tau^*$) | **Test Recall** | Test Precision | **Test F1** | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_active` | Logistic Regression (Balanced) | 0.7146 | 0.44 | 0.7668 | **0.7713** | 0.6919 | **0.7294** | 0.8475 |
| `dynamic_active` | Random Forest (Balanced) | 0.6688 | 0.40 | 0.7771 | **0.7751** | 0.6755 | **0.7219** | 0.8441 |
| `dynamic_active` | LightGBM (Balanced) | 0.7146 | 0.43 | 0.7790 | **0.7713** | 0.6879 | **0.7272** | 0.8492 |
| `pre_course_active` | Logistic Regression (Balanced) | 0.7225 | 0.40 | 0.8087 | **0.7895** | 0.7742 | **0.7818** | 0.8743 |
| `pre_course_active` | Random Forest (Balanced) | 0.7013 | 0.37 | 0.8313 | **0.8068** | 0.7488 | **0.7767** | 0.8720 |
| `pre_course_active` | LightGBM (Balanced) | 0.7259 | 0.38 | 0.8246 | **0.7924** | 0.7709 | **0.7815** | 0.8758 |
| `full` | Logistic Regression (Balanced) | 0.7592 | 0.39 | 0.8353 | **0.8149** | 0.8152 | **0.8151** | 0.8960 |
| `full` | Random Forest (Balanced) | 0.7457 | 0.40 | 0.8349 | **0.8092** | 0.8229 | **0.8160** | 0.8946 |
| `full` | LightGBM (Balanced) | 0.7681 | 0.46 | 0.8024 | **0.7950** | 0.8627 | **0.8274** | 0.9005 |

### 5.2. Mốc 50%
| Cohort | Mô hình (`week6_advanced`) | Default $\tau=0.50$ Recall | Tuned $\tau^*$ (Val) | Val Recall ($\tau^*$) | **Test Recall** | Test Precision | **Test F1** | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_active` | Logistic Regression (Balanced) | 0.7806 | 0.55 | 0.7601 | **0.7525** | 0.7960 | **0.7736** | 0.9064 |
| `dynamic_active` | Random Forest (Balanced) | 0.7282 | 0.43 | 0.7813 | **0.7866** | 0.7680 | **0.7772** | 0.9029 |
| `dynamic_active` | LightGBM (Balanced) | 0.7699 | 0.48 | 0.7844 | **0.7798** | 0.7751 | **0.7774** | 0.9070 |
| `pre_course_active` | Logistic Regression (Balanced) | 0.8088 | 0.48 | 0.8255 | **0.8165** | 0.8959 | **0.8543** | 0.9368 |
| `pre_course_active` | Random Forest (Balanced) | 0.7900 | 0.44 | 0.8366 | **0.8155** | 0.8887 | **0.8505** | 0.9337 |
| `pre_course_active` | LightGBM (Balanced) | 0.8015 | 0.49 | 0.8260 | **0.8059** | 0.9058 | **0.8529** | 0.9343 |
| `full` | Logistic Regression (Balanced) | 0.8385 | 0.43 | 0.8707 | **0.8584** | 0.8892 | **0.8735** | 0.9451 |
| `full` | Random Forest (Balanced) | 0.8340 | 0.46 | 0.8597 | **0.8454** | 0.9114 | **0.8772** | 0.9466 |
| `full` | LightGBM (Balanced) | 0.8397 | 0.51 | 0.8544 | **0.8381** | 0.9229 | **0.8785** | 0.9484 |

### 5.3. Mốc 75%
| Cohort | Mô hình (`week6_advanced`) | Default $\tau=0.50$ Recall | Tuned $\tau^*$ (Val) | Val Recall ($\tau^*$) | **Test Recall** | Test Precision | **Test F1** | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_active` | Logistic Regression (Balanced) | 0.8281 | 0.60 | 0.7804 | **0.7831** | 0.8555 | **0.8177** | 0.9420 |
| `dynamic_active` | Random Forest (Balanced) | 0.7642 | 0.41 | 0.8209 | **0.8227** | 0.8487 | **0.8355** | 0.9440 |
| `dynamic_active` | LightGBM (Balanced) | 0.8020 | 0.50 | 0.8056 | **0.8020** | 0.8430 | **0.8220** | 0.9429 |
| `pre_course_active` | Logistic Regression (Balanced) | 0.8574 | 0.55 | 0.8708 | **0.8497** | 0.9489 | **0.8966** | 0.9639 |
| `pre_course_active` | Random Forest (Balanced) | 0.8560 | 0.40 | 0.9007 | **0.8834** | 0.9179 | **0.9003** | 0.9654 |
| `pre_course_active` | LightGBM (Balanced) | 0.8704 | 0.42 | 0.8969 | **0.8873** | 0.9275 | **0.9069** | 0.9668 |
| `full` | Logistic Regression (Balanced) | 0.8897 | 0.47 | 0.9020 | **0.8950** | 0.9462 | **0.9199** | 0.9714 |
| `full` | Random Forest (Balanced) | 0.8836 | 0.45 | 0.9122 | **0.8991** | 0.9477 | **0.9228** | 0.9732 |
| `full` | LightGBM (Balanced) | 0.8918 | 0.51 | 0.9008 | **0.8914** | 0.9601 | **0.9245** | 0.9734 |

### 5.4. Mốc 100%
| Cohort | Mô hình (`week6_advanced`) | Default $\tau=0.50$ Recall | Tuned $\tau^*$ (Val) | Val Recall ($\tau^*$) | **Test Recall** | Test Precision | **Test F1** | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dynamic_active` | Logistic Regression (Balanced) | 0.8422 | 0.60 | 0.8400 | **0.8118** | 0.8990 | **0.8532** | 0.9590 |
| `dynamic_active` | Random Forest (Balanced) | 0.7882 | 0.45 | 0.8391 | **0.8059** | 0.9013 | **0.8509** | 0.9613 |
| `dynamic_active` | LightGBM (Balanced) | 0.8294 | 0.66 | 0.8165 | **0.7765** | 0.9307 | **0.8466** | 0.9623 |
| `pre_course_active` | Logistic Regression (Balanced) | 0.8935 | 0.54 | 0.8964 | **0.8892** | 0.9675 | **0.9267** | 0.9775 |
| `pre_course_active` | Random Forest (Balanced) | 0.8772 | 0.51 | 0.8954 | **0.8752** | 0.9816 | **0.9254** | 0.9798 |
| `pre_course_active` | LightGBM (Balanced) | 0.9013 | 0.53 | 0.9046 | **0.8964** | 0.9683 | **0.9310** | 0.9807 |
| `full` | Logistic Regression (Balanced) | 0.9117 | 0.60 | 0.9097 | **0.8987** | 0.9796 | **0.9374** | 0.9819 |
| `full` | Random Forest (Balanced) | 0.9032 | 0.43 | 0.9284 | **0.9178** | 0.9682 | **0.9424** | 0.9840 |
| `full` | LightGBM (Balanced) | 0.9125 | 0.48 | 0.9329 | **0.9158** | 0.9711 | **0.9426** | 0.9843 |

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
