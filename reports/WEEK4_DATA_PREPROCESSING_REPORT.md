# BÁO CÁO KỸ THUẬT TIỀN XỬ LÝ DỮ LIỆU & KIỂM SOÁT RÒ RỈ THÔNG TIN (DATA PREPROCESSING & ZERO LEAKAGE)
## Tuần 4: Xử lý Missing Values, Chuẩn hóa, Ngoại lai & Chia tập Train/Val/Test
## Đề tài: LMS Tiên tiến và Learning Analytics (Học máy Dự đoán Kết quả Học tập Sinh viên OULAD)

---

## 1. TỔNG QUAN TIẾN ĐỘ TUẦN 4

Toàn bộ **8 hạng mục công việc cốt lõi** của Tuần 4 (từ Nhiệm vụ 4.1 đến 4.8) theo Kế hoạch Dự án LMS Learning Analytics đã được hoàn thành 100% thông qua pipeline hướng đối tượng chuẩn mực [`src/preprocessing.py`](file:///c:/Users/mrore/Project_I/src/preprocessing.py), Jupyter Notebook [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb), bộ kiểm thử tự động [`src/tests/test_preprocessing.py`](file:///c:/Users/mrore/Project_I/src/tests/test_preprocessing.py) (**9/9 tests PASSED**) và xuất bản **6 biểu đồ phân tích chuyên sâu** (`15_*.png` đến `20_*.png` tại `reports/figures/`).

| Mã CV | Nội dung công việc Tuần 4 | Trạng thái | Sản phẩm đầu ra |
|:---:|---|:---:|---|
| **T4.1** | Xử lý missing values trong `studentAssessment` (điểm thiếu $\rightarrow 0$, cờ `has_no_submissions`) | Hoàn thành | Pipeline Imputer & [15_missing_values_heatmap_and_imputation.png](file:///c:/Users/mrore/Project_I/reports/figures/15_missing_values_heatmap_and_imputation.png) |
| **T4.2** | Xử lý `date_submitted` missing, `imd_band` missing (Mode / Unknown, cờ `imd_was_missing`) | Hoàn thành | Dữ liệu sạch, triệt tiêu toàn bộ NaN |
| **T4.3** | Phân tích & xử lý sinh viên Withdrawn sớm (`date_unregistration < 0`, 0 clicks) | Hoàn thành | [16_early_unregistration_impact.png](file:///c:/Users/mrore/Project_I/reports/figures/16_early_unregistration_impact.png) & Cấu hình lọc linh hoạt |
| **T4.4** | Phát hiện & xử lý ngoại lai (IQR Capping $Q_3 + 3 \times \text{IQR}$, Winsorization & Log1p) | Hoàn thành | [17_outliers_boxplots_and_capping.png](file:///c:/Users/mrore/Project_I/reports/figures/17_outliers_boxplots_and_capping.png) |
| **T4.5** | Encoding biến phân loại (Ordinal mapping thứ bậc & One-Hot Encoding danh mục) | Hoàn thành | 63 đặc trưng số đầu ra, không còn kiểu chuỗi/object |
| **T4.6** | Chuẩn hóa dữ liệu số (`RobustScaler` / `StandardScaler`) - **100% Zero Data Leakage** | Hoàn thành | [18_feature_scaling_distributions.png](file:///c:/Users/mrore/Project_I/reports/figures/18_feature_scaling_distributions.png) & `scaler.joblib` |
| **T4.7** | Chia tập Train/Val/Test (70/15/15) - **Stratified Split** & **Group-aware Split** theo `id_student` | Hoàn thành | [19_train_val_test_stratification.png](file:///c:/Users/mrore/Project_I/reports/figures/19_train_val_test_stratification.png) & 6 tập file CSV |
| **T4.8** | Xử lý class imbalance (SMOTE trên Train, Class Weight 'balanced', Random Under-sampling) | Hoàn thành | [20_class_imbalance_and_resampling.png](file:///c:/Users/mrore/Project_I/reports/figures/20_class_imbalance_and_resampling.png) & Bảng đối sánh ML |

---

## 2. CHI TIẾT CHIẾN LƯỢC XỬ LÝ MISSING VALUES (T4.1 & T4.2)
Biểu đồ minh họa: [reports/figures/15_missing_values_heatmap_and_imputation.png](file:///c:/Users/mrore/Project_I/reports/figures/15_missing_values_heatmap_and_imputation.png)

### 2.1. Thống kê tỷ lệ thiếu và cơ chế phát sinh
Trong bảng đặc trưng cấp độ Sinh viên - Khóa học ($N = 32,593$ bản ghi), các trường dữ liệu có tỷ lệ khuyết thiếu được phân loại theo 4 cơ chế bản chất:

1. **`date_unregistration` ($22,521$ missing, chiếm $69.10\%$)**:
   - *Bản chất*: Giá trị `NaN` biểu thị sinh viên **hoàn thành khóa học** hoặc duy trì đăng ký đến hết kỳ học (không rút lui giữa chừng).
   - *Quy tắc chống rò rỉ dữ liệu (Zero Leakage Rule)*: Cột `date_unregistration` **TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA VÀO BỘ FEATURE $X$** khi dự đoán nguy cơ rớt môn/rút lui trong thời gian thực. Việc sử dụng cột này làm đặc trưng dự đoán sẽ gây ra hiện tượng rò rỉ nhãn mục tiêu (Target Leakage), dẫn đến độ chính xác giả mạo 100% phi thực tế.
2. **Điểm kiểm tra (`avg_score`, `min_score`, `max_score`: $6,773$ missing, $20.78\%$ | `tma1_score`: $8,167$ missing, $25.06\%$)**:
   - *Bản chất*: Sinh viên không nộp bất kỳ bài tập nào trong khóa học hoặc bỏ bài kiểm tra đầu tiên (`TMA1`).
   - *Chiến lược xử lý*: Impute giá trị $0.0$ điểm (phản ánh đúng thực tế học thuật là 0 điểm nếu vắng thi/không nộp) kết hợp bổ sung cờ nhị phân `has_no_submissions` ($= 1$ nếu không nộp bài) và `submitted_tma1` ($= 0$ nếu không nộp TMA1).
3. **Độ kịp thời khi nộp bài (`avg_days_early`: $6,751$ missing, $20.71\%$ | `tma1_days_early`: $8,128$ missing, $24.94\%$)**:
   - *Bản chất*: Không thể tính số ngày nộp trước hạn đối với những bài tập sinh viên không hề nộp.
   - *Chiến lược xử lý*: Gán giá trị trễ danh nghĩa định lượng (Penalty Sentinel Value) là $-10.0$ ngày (biểu thị nộp muộn sau hạn chót).
4. **Vùng điều kiện kinh tế xã hội (`imd_band`: $1,111$ missing, $3.41\%$)**:
   - *Bản chất*: Sinh viên cư trú ngoài vùng nước Anh (Scotland, Wales, Bắc Ireland) hoặc không khai báo mã bưu chính.
   - *Chiến lược xử lý*: Impute theo giá trị `Mode` (phổ biến nhất) tính riêng từ tập Train (`20-30%`, decile 1) hoặc ánh xạ về mức trung vị decile 4, đồng thời tạo biến cờ `imd_was_missing = 1.0` để bảo toàn thông tin khuyết thiếu.
5. **Ngày đăng ký môn học (`date_registration`: $45$ missing, $0.14\%$)**:
   - *Chiến lược*: Impute bằng giá trị `Median` của tập Train ($-57.0$ ngày trước khai giảng).

Sau khi chạy qua bước tiền xử lý, tỷ lệ NaN của toàn bộ bảng đặc trưng $X$ đạt **chính xác 0.00%** trên cả 3 tập Train, Validation và Test.

---

## 3. PHÂN TÍCH VÀ CHIẾN LƯỢC XỬ LÝ SINH VIÊN WITHDRAWN SỚM (T4.3)
Biểu đồ minh họa: [reports/figures/16_early_unregistration_impact.png](file:///c:/Users/mrore/Project_I/reports/figures/16_early_unregistration_impact.png)

Tiếp nối khuyến nghị từ Tuần 2 và Tuần 3, nhóm nghiên cứu đã phân tích sâu hành vi của nhóm sinh viên rút môn sớm trước ngày khai giảng chính thức:

### 3.1. Các phát hiện thực nghiệm cốt lõi:
- **Số lượng rút trước ngày 0 (`date_unregistration < 0`)**: Có **$2,678$ lượt đăng ký**, chiếm **$8.22\%$** tổng số đăng ký OULAD.
- **Phân bố kết quả**: Trong số $2,678$ trường hợp này, có tới **$2,676$ sinh viên nhận kết quả `Withdrawn`** ($99.93\%$) và chỉ $2$ trường hợp nhận nhãn `Fail`.
- **Hành vi tương tác LMS**: Có tới **$2,213$ sinh viên hoàn toàn có $0$ lượt click** trên hệ thống ($82.64\%$ nhóm rút trước khai giảng). Nhóm này không tham gia bất kỳ hoạt động học tập nào trên LMS.

### 3.2. Đánh giá tính hai mặt (Trade-off) và Cấu hình Pipeline:
1. **Góc độ Quản trị Tuyển sinh / Retention Tổng thể (Full Cohort - $N = 32,593$)**:
   - Phù hợp với bài toán quản lý nhà trường khi muốn dự đoán khả năng sinh viên đăng ký xong nhưng bỏ khóa ngay từ vạch xuất phát dựa trên đặc điểm nhân khẩu học và ngày đăng ký sớm/muộn.
   - Đây là cấu hình mặc định trong các bài báo chuẩn OULAD (Kuzilek et al., 2017).
2. **Góc độ Learning Analytics Tương tác Thời gian thực (Active Learner Cohort - $N = 30,380$)**:
   - Nếu giữ $2,213$ sinh viên $0$ click này trong các bài toán dự đoán tại mốc Tuần 4, Tuần 9 hay 25%, mô hình có xu hướng học mẹo (trivial shortcut): *"cứ hễ 0 click thì 100% Withdrawn"*, làm suy giảm năng lực nhận diện các sinh viên *thực sự tham gia học nhưng có nguy cơ rớt môn dần về sau*.
3. **Giải pháp kiến trúc**: Pipeline `OULADPreprocessor` cung cấp tham số cấu hình:
   ```python
   preprocessor = OULADPreprocessor(filter_pre_course_unreg=False) # Full Cohort (Mặc định)
   preprocessor = OULADPreprocessor(filter_pre_course_unreg=True)  # Active Cohort (Lọc bỏ 2,213 SV rác)
   ```

---

## 4. PHÁT HIỆN VÀ XỬ LÝ NGOẠI LAI (OUTLIER TREATMENT - T4.4)
Biểu đồ minh họa: [reports/figures/17_outliers_boxplots_and_capping.png](file:///c:/Users/mrore/Project_I/reports/figures/17_outliers_boxplots_and_capping.png)

### 4.1. Khảo sát độ lệch và các giá trị cực trị
Nhiều đặc trưng tương tác click và học vụ có phân phối lệch phải nghiêm trọng (Heavy Right-skewed Distribution):

| Đặc trưng | Skewness gốc | Kurtosis gốc | Min | Max | Số mẫu > $Q_3 + 1.5 \times \text{IQR}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| `total_clicks` | **$+3.03$** | $14.95$ | $0.0$ | **$24,139.0$** | $2,503$ ($7.68\%$) |
| `daily_click_std` | **$+4.68$** | $77.70$ | $0.0$ | **$704.67$** | $1,479$ ($4.54\%$) |
| `studied_credits` | **$+1.88$** | $7.74$ | $30$ | **$655$** | $350$ ($1.07\%$) |
| `daily_click_mean` | $+1.56$ | $7.47$ | $0.0$ | $221.20$ | $1,023$ ($3.14\%$) |
| `avg_score` | $-1.27$ | $2.34$ | $0.0$ | $100.0$ | $762$ ($2.95\%$) |

### 4.2. Giải pháp Capping & Biến đổi hàm Log:
- **Tại sao không loại bỏ bản ghi (Hard Drop)?**: Trong giáo dục trực tuyến OULAD, sinh viên có tổng số click cực lớn ($> 10,000$) phần lớn thuộc nhóm **Distinction** (sinh viên xuất sắc, tích cực). Việc xóa bỏ họ sẽ làm mất đi các mẫu đại diện quý giá nhất của nhóm tích cực.
- **Kỹ thuật Capping (Winsorization) kết hợp Log1p**:
  1. *Ngưỡng chặn trên*: Tính toán trên tập Train theo công thức Extreme Outlier Boundary:
     $$\text{Upper Bound} = Q_3 + 3.0 \times \text{IQR}$$
     Đối với `total_clicks`, ngưỡng chặn trên là $\approx 5,200$ clicks; đối với `studied_credits`, ngưỡng chặn trên là $240$ tín chỉ (giữ nguyên tải học tập thực tế tối đa).
  2. *Biến đổi hàm Log (`np.log1p`)*: Áp dụng $x' = \ln(1 + x)$ trên toàn bộ các biến tương tác click sau khi capping. Kết quả làm giảm Skewness của `total_clicks` từ $+3.03$ xuống **$-0.21$**, đưa phân phối về dạng gần phân phối chuẩn (Gaussian-like), giúp các mô hình tuyến tính và mạng nơ-ron hội tụ tối ưu.

---

## 5. MÃ HÓA BIẾN PHÂN LOẠI & CHUẨN HÓA DỮ LIỆU SỐ (T4.5 & T4.6)
Biểu đồ minh họa: [reports/figures/18_feature_scaling_distributions.png](file:///c:/Users/mrore/Project_I/reports/figures/18_feature_scaling_distributions.png)

### 5.1. Mã hóa Thứ bậc (Ordinal Encoding) & Danh mục (One-Hot Encoding)
Bảo toàn tính chất tự nhiên của các nhóm biến:
- **Biến Thứ bậc (Ordinal Categorical)**:
  - `highest_education`: `No Formal quals` ($0$) $\rightarrow$ `Lower Than A Level` ($1$) $\rightarrow$ `A Level or Equivalent` ($2$) $\rightarrow$ `HE Qualification` ($3$) $\rightarrow$ `Post Graduate Qualification` ($4$).
  - `imd_band`: `0-10%` ($0$) $\rightarrow \dots \rightarrow$ `90-100%` ($9$), `Unknown` $\rightarrow 4$.
  - `age_band`: `0-35` ($0$) $\rightarrow$ `35-55` ($1$) $\rightarrow$ `55<=` ($2$).
- **Biến Danh mục (Nominal Categorical)**:
  - `gender` ($2$ cột), `disability` ($2$ cột), `region` ($13$ cột), `code_module` ($7$ môn học) được mã hóa bằng `OneHotEncoder(handle_unknown='ignore')`.

### 5.2. Chuẩn hóa Dữ liệu Số (Feature Scaling) & Quy tắc Zero Data Leakage
- **Lựa chọn Scaler**: Sử dụng `RobustScaler` (dựa trên Trung vị và Khoảng tứ phân vị IQR: $z = \frac{x - \text{median}}{\text{IQR}}$) làm bộ chuẩn hóa chính vì khả năng chống chịu ngoại lai vượt trội so với `StandardScaler`.
- **Bảo đảm Không rò rỉ dữ liệu (Zero Leakage Enforcement)**:
  - Scaler và Imputer **CHỈ ĐƯỢC FIT TRÊN TẬP HUẤN LUYỆN (Train set)**.
  - Các tham số $\text{median}$, $\text{IQR}$, $\text{mode}$ sau khi fit trên Train được đóng băng và lưu vào `scaler.joblib`, sau đó chỉ gọi phương thức `.transform()` trên Validation set và Test set.
  - Bộ unit test `test_03_zero_data_leakage_imputation` đã xác minh độc lập: việc thay đổi dữ liệu tập Test không làm thay đổi bất kỳ tham số nào trong Scaler và Imputer.

---

## 6. CHIẾN LƯỢC CHIA TẬP TRAIN / VAL / TEST & KIỂM SOÁT GROUP LEAKAGE (T4.7)
Biểu đồ minh họa: [reports/figures/19_train_val_test_stratification.png](file:///c:/Users/mrore/Project_I/reports/figures/19_train_val_test_stratification.png)

Tỷ lệ chia: **$70\%$ Train ($22,815$ bản ghi) | $15\%$ Validation ($4,889$ bản ghi) | $15\%$ Test ($4,889$ bản ghi)**.

### 6.1. Bảo toàn Tỷ lệ Phân tầng (Stratification Consistency)
Tỷ lệ nhãn nguy cơ `at_risk = 1` được duy trì đồng nhất tuyệt đối trên cả 3 tập phân chia:
- Tập Train: **$52.80\%$** At-Risk ($12,046$ sinh viên).
- Tập Validation: **$52.79\%$** At-Risk ($2,581$ sinh viên).
- Tập Test: **$52.79\%$** At-Risk ($2,581$ sinh viên).
- Độ lệch tỷ lệ nhãn giữa các tập: **$< 0.01\%$**, vượt xa yêu cầu khắt khe ($< 0.5\%$).

### 6.2. Phân tích Nguy cơ Rò rỉ Nhóm Sinh viên Đa môn (`id_student` Group Split)
- Trong $32,593$ lượt đăng ký, có $28,785$ sinh viên duy nhất, trong đó có **$3,538$ sinh viên đăng ký nhiều hơn $1$ môn học** (tối đa $5$ môn học cùng lúc).
- Khi chia ngẫu nhiên thông thường (`Stratified Random Split`): Có tới **$812$ sinh viên xuất hiện đồng thời ở cả tập Train và tập Test** ($17.5\%$ tập Test). Do đặc điểm nhân khẩu học (`gender`, `age_band`, `region`, `imd_band`) của cùng một sinh viên không đổi, điều này có thể dẫn đến rò rỉ thông tin cá nhân.
- Khi áp dụng `Stratified Group Split` (`StratifiedGroupKFold` theo `id_student`): Số sinh viên trùng lặp giữa Train - Test, Train - Val, Val - Test **chính xác bằng $0$ (Triệt tiêu rò rỉ danh tính 100%)**!
- Cả 2 tùy chọn chia tập đều được tích hợp đầy đủ trong [`src/preprocessing.py`](file:///c:/Users/mrore/Project_I/src/preprocessing.py) và sẵn sàng cho các vòng thực nghiệm mô hình tiếp theo.

---

## 7. XỬ LÝ MẤT CÂN BẰNG LỚP & ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH (T4.8)
Biểu đồ minh họa: [reports/figures/20_class_imbalance_and_resampling.png](file:///c:/Users/mrore/Project_I/reports/figures/20_class_imbalance_and_resampling.png)

### 7.1. Phân tích Mức độ Mất cân bằng:
- **Bài toán Nhị phân (`at_risk`)**: Tỷ lệ khá cân bằng: `Not At-Risk (0)` chiếm $47.20\%$ vs `At-Risk (1)` chiếm $52.80\%$. Khi áp dụng SMOTE trên tập Train, số lượng mẫu được cân bằng chính xác $12,046 : 12,046$ ($24,092$ mẫu).
- **Bài toán 4 lớp gốc (`final_result`)**: Mất cân bằng đáng kể giữa nhóm `Distinction` ($9.28\%$) so với `Pass` ($36.94\%$), `Fail` ($21.64\%$), `Withdrawn` ($32.15\%$).

### 7.2. Kết quả Thực nghiệm Định lượng trên Tập Test Độc lập:
Nhóm nghiên cứu đã huấn luyện và đánh giá đối kháng các giải pháp tiền xử lý trên tập Test gồm **$4,889$ mẫu**:

| Thuật toán | Biến thể Tiền xử lý | Accuracy | Precision | Recall (At-Risk) | F1-Score | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Chuẩn hóa Robust (Standard) | $0.9296$ | $0.9694$ | $0.8950$ | $0.9307$ | $0.9779$ |
| Logistic Regression | Trọng số Lớp (`class_weight='balanced'`) | $0.9296$ | $0.9737$ | $0.8907$ | $0.9304$ | $0.9779$ |
| Logistic Regression | SMOTE trên tập Train | $0.9294$ | $0.9721$ | $0.8919$ | $0.9303$ | $0.9779$ |
| **Decision Tree** | Độ sâu tối đa max_depth=6 | $0.9239$ | $0.9593$ | $0.8938$ | $0.9254$ | $0.9728$ |
| **Random Forest** | 100 Trees, max_depth=10 (Chuẩn) | **$0.9307$** | **$0.9722$** | **$0.8942$** | **$0.9316$** | **$0.9806$** |
| Random Forest | Trọng số Lớp (`class_weight='balanced'`) | $0.9298$ | **$0.9742$** | $0.8907$ | $0.9306$ | **$0.9812$** |
| Random Forest | SMOTE trên tập Train | $0.9282$ | $0.9733$ | $0.8884$ | $0.9289$ | $0.9808$ |

> **Nhận xét chuyên môn**:
> 1. Dữ liệu sau tiền xử lý Tuần 4 giúp mô hình đạt độ chính xác rất cao: **Accuracy đạt $93.07\%$**, **ROC-AUC đạt $0.9812$**, **F1-Score đạt $0.9316$**.
> 2. Đối với bài toán nhị phân At-Risk, do tỷ lệ gốc đã tương đối cân bằng ($47:53$), phương pháp sử dụng trọng số lớp (`class_weight='balanced'`) mang lại ROC-AUC cao nhất ($0.9812$) mà không làm tăng chi phí tính toán hay tạo mẫu tổng hợp giả mạo.

---

## 8. KẾT QUẢ KIỂM THỬ ĐỘC LẬP (TEST SUITE VERIFICATION)
Tệp kiểm thử: [`src/tests/test_preprocessing.py`](file:///c:/Users/mrore/Project_I/src/tests/test_preprocessing.py)  
Toàn bộ **9/9 test cases** đã chạy thành công tuyệt đối ($100\%$ Passed):

```text
Ran 9 tests in 13.884s
OK
```

1. `test_01_no_missing_values`: Xác nhận $0$ giá trị NaN sau khi tiền xử lý trên Train, Val, Test. (PASSED)
2. `test_02_feature_shapes_and_column_alignment`: Khớp $100\%$ tên và số lượng cột giữa 3 tập ($63$ đặc trưng). (PASSED)
3. `test_03_zero_data_leakage_imputation`: Khẳng định giá trị điền khuyết được học duy nhất từ Train set. (PASSED)
4. `test_04_stratification_consistency`: Độ lệch phân tầng nhãn At-Risk giữa các tập $< 0.005$. (PASSED)
5. `test_05_group_stratified_split_zero_student_overlap`: Số sinh viên trùng giữa Train và Test bằng $0$. (PASSED)
6. `test_06_all_features_are_numeric`: Tất cả $63$ cột đều thuộc kiểu dữ liệu số (`float64`/`int64`). (PASSED)
7. `test_07_smote_resampling_balance`: Cân bằng chính xác tỷ lệ $50:50$ trên tập Train sau SMOTE. (PASSED)
8. `test_08_outlier_bounds_applied`: Các cực trị được chặn chặt chẽ theo ngưỡng $Q_3 + 3 \times \text{IQR}$. (PASSED)
9. `test_09_artifacts_saved_and_reloadable`: Nạp lại toàn bộ Scaler, Encoders, Metadata từ disk cho kết quả tương thích tuyệt đối. (PASSED)

---

## 9. DANH MỤC SẢN PHẨM BÀN GIAO TUẦN 4 (DELIVERABLES)

- [x] **Mã nguồn Pipeline Tiền xử lý**: [`src/preprocessing.py`](file:///c:/Users/mrore/Project_I/src/preprocessing.py) (lớp `OULADPreprocessor`, hỗ trợ CLI runner, cấu hình linh hoạt).
- [x] **Jupyter Notebook Thực nghiệm**: [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb) (trực quan hóa chi tiết từng bước 4.1 - 4.8).
- [x] **Bộ Kiểm thử Độc lập**: [`src/tests/test_preprocessing.py`](file:///c:/Users/mrore/Project_I/src/tests/test_preprocessing.py) ($9/9$ bài test PASSED).
- [x] **Dữ liệu Tiền xử lý Hoàn chỉnh**:
  - `data/processed/train_features.csv` ($22,815 \times 63$)
  - `data/processed/train_target.csv` ($22,815 \times 1$)
  - `data/processed/train_features_smote.csv` ($24,092 \times 63$)
  - `data/processed/train_target_smote.csv` ($24,092 \times 1$)
  - `data/processed/val_features.csv` ($4,889 \times 63$)
  - `data/processed/val_target.csv` ($4,889 \times 1$)
  - `data/processed/test_features.csv` ($4,889 \times 63$)
  - `data/processed/test_target.csv` ($4,889 \times 1$)
- [x] **Artifacts Mô hình Đã Lưu**:
  - `data/processed/scaler.joblib`
  - `data/processed/one_hot_encoder.joblib`
  - `data/processed/preprocessor_state.joblib`
  - `data/processed/preprocessing_metadata.json`
- [x] **6 Biểu đồ Báo cáo Chất lượng Cao**:
  - `reports/figures/15_missing_values_heatmap_and_imputation.png`
  - `reports/figures/16_early_unregistration_impact.png`
  - `reports/figures/17_outliers_boxplots_and_capping.png`
  - `reports/figures/18_feature_scaling_distributions.png`
  - `reports/figures/19_train_val_test_stratification.png`
  - `reports/figures/20_class_imbalance_and_resampling.png`
- [x] **Tài liệu Báo cáo LaTeX**: [`báo_cáo_tuần_4.tex`](file:///c:/Users/mrore/Project_I/báo_cáo_tuần_4.tex) và tệp PDF biên dịch.
