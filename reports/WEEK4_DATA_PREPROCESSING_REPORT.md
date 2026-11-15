# BÁO CÁO KỸ THUẬT TIỀN XỬ LÝ DỮ LIỆU & KIỂM SOÁT RÒ RỈ THÔNG TIN (DATA PREPROCESSING & ZERO LEAKAGE)
## Tuần 4: Xử lý Missing Values, Chuẩn hóa, Ngoại lai & Chia tập Train/Val/Test
## Đề tài: LMS Tiên tiến và Learning Analytics (Học máy Dự đoán Kết quả Học tập Sinh viên OULAD)

---

## 1. TỔNG QUAN TIẾN ĐỘ TUẦN 4

Toàn bộ **8 hạng mục công việc cốt lõi** của Tuần 4 (từ Nhiệm vụ 4.1 đến 4.8) theo Kế hoạch Dự án LMS Learning Analytics đã được hoàn thành 100% thông qua pipeline hướng đối tượng `OULADPreprocessor` và bộ kiểm thử tự động (**9/9 tests PASSED**) tích hợp hoàn toàn trong Jupyter Notebook [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb), xuất bản **6 biểu đồ phân tích chuyên sâu** (`15_*.png` đến `20_*.png` tại `reports/figures/`).

| Mã CV | Nội dung công việc Tuần 4 | Trạng thái | Sản phẩm đầu ra |
|:---:|---|:---:|---|
| **T4.1** | Xử lý missing values trong `studentAssessment` (điểm thiếu $\rightarrow 0$, cờ `has_no_submissions`) | Hoàn thành | Pipeline Imputer & [15_missing_values_heatmap_and_imputation.png](file:///c:/Users/mrore/Project_I/reports/figures/15_missing_values_heatmap_and_imputation.png) |
| **T4.2** | Xử lý `date_submitted` missing, `imd_band` missing (Impute bằng Mode Train `'20-30%'` $\rightarrow 2$, kèm cờ `imd_was_missing = 1`) | Hoàn thành | Dữ liệu sạch, triệt tiêu toàn bộ NaN |
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
   - *Chiến lược xử lý*: Gán giá trị trễ danh nghĩa định lượng (Penalty Sentinel Value) là $-10.0$ ngày (biểu thị không nộp bài / trễ hạn nghiêm trọng).
   - *Giải trình kỹ thuật về sự chuyển đổi từ $-5.0$ ngày (Tuần 3) sang $-10.0$ ngày (Tuần 4)*: Trong khảo sát sơ bộ EDA Tuần 3, nghiên cứu dùng mức phạt danh nghĩa $-5.0$ ngày (mức trễ 1 tuần thông thường). Tuy nhiên, khi phân tích phân phối thực tế của nhóm sinh viên có nộp bài nhưng nộp muộn trong Tuần 4, dữ liệu chỉ ra phân vị $99\%$ của nhóm nộp trễ thực tế chỉ dao động từ $-1$ đến $-5$ ngày. Việc nâng mức phạt lên $-10.0$ ngày tạo ra một bước nhảy gián đoạn (Discontinuous Penalty Boundary) nằm ngoài phân phối nộp trễ thông thường, giúp các thuật toán học máy (đặc biệt là mô hình cây quyết định) dễ dàng tìm điểm cắt phân nhánh phân tách dứt khoát nhóm sinh viên bỏ học/không nộp bài hoàn toàn (`has_no_submissions = 1`) khỏi nhóm nộp trễ thông thường.
4. **Vùng điều kiện kinh tế xã hội (`imd_band`: $1,111$ missing, $3.41\%$)**:
   - *Bản chất*: Sinh viên cư trú ngoài vùng nước Anh (Scotland, Wales, Bắc Ireland) hoặc không khai báo mã bưu chính.
   - *Chiến lược xử lý*: Impute theo giá trị `Mode` (phổ biến nhất) tính riêng từ tập Train (`'20-30%'` -- Decile 3 $\rightarrow$ nhận giá trị ánh xạ **`2`**), đồng thời tạo biến cờ `imd_was_missing = 1.0` để bảo toàn thông tin khuyết thiếu (còn giá trị dự phòng `'Unknown'` nếu gặp chuỗi lạ chưa biết khi suy luận sẽ được ánh xạ về mức trung vị `4`).
5. **Ngày đăng ký môn học (`date_registration`: $45$ missing, $0.14\%$)**:
   - *Chiến lược*: Impute bằng giá trị `Median` của tập Train ($-57.0$ ngày trước khai giảng).

Sau khi chạy qua bước tiền xử lý, tỷ lệ NaN của toàn bộ bảng đặc trưng $X$ đạt **chính xác 0.00%** trên cả 3 tập Train, Validation và Test.

---

## 3. PHÂN TÍCH VÀ CHIẾN LƯỢC XỬ LÝ SINH VIÊN WITHDRAWN SỚM (T4.3)
Biểu đồ minh họa: [reports/figures/16_early_unregistration_impact.png](file:///c:/Users/mrore/Project_I/reports/figures/16_early_unregistration_impact.png)

Tiếp nối khuyến nghị từ Tuần 2 và Tuần 3, nghiên cứu đã phân tích sâu hành vi của nhóm sinh viên rút môn sớm trước ngày khai giảng chính thức:

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

| Đặc trưng | Skewness gốc | Kurtosis gốc | Min | Max (Toàn tập / Train) | Ngưỡng chặn $Q_3 + 3 \times \text{IQR}$ (Train) |
|---|:---:|:---:|:---:|:---:|:---:|
| `total_clicks` | **$+3.03$** | $14.95$ | $0.0$ | **$24,139.0$ / $20,391.0$** | $5,906.0$ (kết hợp $\ln(1+x)$) |
| `daily_click_std` | **$+4.68$** | $77.70$ | $0.0$ | **$704.67$ / $704.67$** | $86.27$ |
| `studied_credits` | **$+1.88$** | $7.74$ | $30$ | **$655.0$ / $655.0$** | $300.0$ ($Q_1=60, Q_3=120$) |
| `daily_click_mean` | $+1.56$ | $7.47$ | $0.0$ | $221.20$ / $221.20$ | $64.29$ |
| `active_days` | $+1.17$ | $0.95$ | $0.0$ | $286.0$ / $286.0$ | $307.0$ |
| `avg_score` | $-1.27$ | $2.34$ | $0.0$ | $100.0$ / $100.0$ | Không chặn (thang $[0, 100]$) |

### 4.2. Giải pháp Capping & Biến đổi hàm Log (Trực quan hóa trọn vẹn 8 biểu đồ cho 4 đặc trưng cốt lõi):
- **Tại sao không loại bỏ bản ghi (Hard Drop)?**: Trong giáo dục trực tuyến OULAD, sinh viên có tổng số click cực lớn ($> 10,000$) phần lớn thuộc nhóm **Distinction** (sinh viên xuất sắc, tích cực). Việc xóa bỏ họ sẽ làm mất đi các mẫu đại diện quý giá nhất của nhóm tích cực.
- **Kỹ thuật Capping theo hàng rào cực trị IQR kết hợp Log1p**:
  1. *Ngưỡng chặn trên*: Tính toán trên tập Train theo công thức Extreme Outlier Boundary:
     $$\text{Upper Bound} = Q_3 + 3.0 \times \text{IQR}$$
     Đối với `total_clicks`, ngưỡng chặn trên tính từ tập Train là $5,906.0$ clicks; đối với `studied_credits`, ngưỡng chặn trên là $300.0$ tín chỉ; đối với `daily_click_mean` là $64.29$ clicks/ngày; đối với `daily_click_std` là $86.27$.
  2. *Biến đổi hàm Log (`np.log1p`)*: Áp dụng $x' = \ln(1 + x)$ trên các biến tương tác click sau khi capping giúp ổn định phương sai và đưa phân phối tiệm cận phân phối chuẩn.
- **Bộ 8 biểu đồ trực quan hóa (Hình 17)**: Trình bày đầy đủ cho cả 4 đặc trưng trong Bảng 4, thiết kế lưới 2 hàng $\times$ 4 cột chuẩn mực (Hàng 1: 4 biểu đồ Boxplot so sánh Gốc vs Sau Capping; Hàng 2: 4 biểu đồ Histogram đồng bộ độ rộng bin kèm đường ngưỡng Capping nét đứt màu đỏ):
  1. `total_clicks`: Boxplot (Max Train `20,391` vs Capping `5,906.0`, dồn $n=565$ mẫu vượt ngưỡng) + Histogram `log1p` phân phối tần suất hiển thị tường minh cả 2 giá trị độ lệch (`Skew toàn mẫu = -1.24 | >0 clicks = -0.21`, do toàn mẫu Train có $10.3\%$ SV có `total_clicks = 0` tạo đỉnh xung nhọn tại 0).
  2. `studied_credits`: Boxplot (Max `655.0` vs Capping `300.0`) + Histogram tần suất trước và sau Capping (dồn $n=26$ mẫu vượt ngưỡng).
  3. `daily_click_mean`: Boxplot (Max `221.20` vs Capping `64.29`) + Histogram tần suất trước và sau Capping (dồn $n=110$ mẫu vượt ngưỡng).
  4. `daily_click_std`: Boxplot (Max `704.67` vs Capping `86.27`) + Histogram tần suất trước và sau Capping (dồn $n=235$ mẫu vượt ngưỡng).

---

## 5. MÃ HÓA BIẾN PHÂN LOẠI & CHUẨN HÓA DỮ LIỆU SỐ (T4.5 & T4.6)
Biểu đồ minh họa: [reports/figures/18_feature_scaling_distributions.png](file:///c:/Users/mrore/Project_I/reports/figures/18_feature_scaling_distributions.png)

### 5.1. Mã hóa Thứ bậc (Ordinal Encoding), Danh mục (One-Hot Encoding) & Biến mục tiêu
Bảo toàn tính chất tự nhiên và quan hệ định lượng của các nhóm biến thông qua các bảng ánh xạ quy chuẩn:

#### Bảng ánh xạ Biến thứ bậc (Ordinal Encoding Mapping Table)
Các biến có thứ bậc tăng dần về trình độ hoặc điều kiện sống được ánh xạ đơn điệu sang số nguyên:

| Tên đặc trưng | Giá trị chuỗi gốc | Giá trị ánh xạ số | Ý nghĩa sư phạm \& Ngữ cảnh học thuật |
|:---|:---|:---:|:---|
| **`highest_education`** | `No Formal quals` | **`0`** | Không có bằng cấp chính quy (trình độ học vấn thấp nhất) |
| | `Lower Than A Level` | **`1`** | Dưới chuẩn tốt nghiệp THPT A Level (tương đương GCSE) |
| | `A Level or Equivalent` | **`2`** | Đạt chuẩn tốt nghiệp THPT A Level hoặc tương đương |
| | `HE Qualification` | **`3`** | Trình độ Giáo dục Đại cương / Cao đẳng (Higher Education) |
| | `Post Graduate Qualification` | **`4`** | Trình độ Sau Đại học (Thạc sĩ, Tiến sĩ - mức học vị cao nhất) |
| **`imd_band`** | `0-10%` | **`0`** | Vùng khó khăn, thiếu thốn nhất (Decile 1) |
| (Chỉ số thiếu thốn kinh tế xã hội) | `10-20` (hoặc `10-20%`) | **`1`** | Phân vị 10-20% thiếu thốn (Decile 2; dữ liệu CSV gốc OULAD lưu là `'10-20'` thiếu ký tự `%`) |
| | `20-30%` | **`2`** | Phân vị 20-30% thiếu thốn (Decile 3 -- Mode tập Train) |
| | `30-40%` | **`3`** | Phân vị 30-40% thiếu thốn (Decile 4) |
| | `40-50%` | **`4`** | Phân vị 40-50% thiếu thốn (Decile 5 - mức trung vị tập Train) |
| | `50-60%` | **`5`** | Phân vị 50-60% thiếu thốn (Decile 6) |
| | `60-70%` | **`6`** | Phân vị 60-70% thiếu thốn (Decile 7) |
| | `70-80%` | **`7`** | Phân vị 70-80% thiếu thốn (Decile 8) |
| | `80-90%` | **`8`** | Phân vị 80-90% thiếu thốn (Decile 9) |
| | `90-100%` | **`9`** | Vùng phát triển, thịnh vượng nhất (Decile 10) |
| | `Missing / NaN` (1,111 mẫu) | **`2`** | Điền bằng Mode tập Train (`'20-30%'` - Decile 3 $\rightarrow 2$) kèm cờ `imd_was_missing = 1` |
| | `Unknown` (Fallback) | **`4`** | Giá trị dự phòng trung vị (Decile 5 $\rightarrow 4$) nếu gặp chuỗi lạ chưa biết khi suy luận |
| **`age_band`** | `0-35` | **`0`** | Nhóm sinh viên trẻ tuổi / thanh niên ($\le 35$ tuổi) |
| | `35-55` | **`1`** | Nhóm người học trung niên ($35 - 55$ tuổi) |
| | `55<=` | **`2`** | Nhóm người học cao niên ($\ge 55$ tuổi) |

#### Bảng quy chuẩn mã hóa Biến danh mục (Binary & One-Hot Encoding Table)
Đối với biến danh mục không có thứ bậc, sử dụng `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fit độc lập trên Train set để sinh $24$ đặc trưng nhị phân ($0/1$):

| Biến danh mục gốc | Số cột sinh ra | Danh sách giá trị danh mục gốc | Tên các đặc trưng nhị phân & Quy tắc mã hóa Vector One-Hot |
|:---|:---:|:---|:---|
| **`gender`** | 2 cột | `F`, `M` | Sinh 2 cột (`gender_F`, `gender_M`): `F` $\to (1, 0)$; `M` $\to (0, 1)$ |
| **`disability`** | 2 cột | `N`, `Y` | Sinh 2 cột (`disability_N`, `disability_Y`): `N` $\to (1, 0)$; `Y` $\to (0, 1)$ |
| **`code_module`** | 7 cột | `AAA`, `BBB`, `CCC`, `DDD`, `EEE`, `FFF`, `GGG` | Sinh 7 cột (`code_module_AAA` ... `code_module_GGG`): Cột tương ứng môn học đăng ký nhận giá trị `1`, 6 cột còn lại nhận `0` |
| **`region`** | 13 cột | 13 vùng địa lý Vương quốc Anh | Sinh 13 cột (`region_East Anglian Region`, `region_East Midlands Region`, `region_Ireland`, `region_London Region`, `region_North Region`, `region_North Western Region`, `region_Scotland`, `region_South East Region`, `region_South Region`, `region_South West Region`, `region_Wales`, `region_West Midlands Region`, `region_Yorkshire Region`): Cột tương ứng vùng cư trú nhận giá trị `1`, 12 cột còn lại nhận `0` |

#### Bảng quy chuẩn ánh xạ Biến mục tiêu (Target Variable Mapping: 4-Class & Nhị phân SEDL)
Biến mục tiêu `final_result` được chuẩn hóa theo hai cấu hình phân loại phục vụ bài toán Learning Analytics:

| Nhãn kết quả gốc | Ánh xạ 4-Class | Ánh xạ Nhị phân SEDL | Bản chất học thuật & Cầu nối xác suất gộp |
|:---|:---:|:---:|:---|
| **`Distinction`** | **`0`** | **`0` (`Not At-Risk`)** | Tốt nghiệp Xuất sắc (Điểm TB $\ge 85$). Người học an toàn, xuất sắc. |
| **`Pass`** | **`1`** | **`0` (`Not At-Risk`)** | Hoàn thành đạt chuẩn ($40 \le \text{Điểm} < 85$). Người học đạt mục tiêu. |
| **`Fail`** | **`2`** | **`1` (`At-Risk`)** | Thi trượt ($\text{Điểm} < 40$). Cần hỗ trợ học tập bổ trợ. |
| **`Withdrawn`** | **`3`** | **`1` (`At-Risk`)** | Rút môn / Bỏ học giữa chừng. Mất kết nối với hệ thống LMS. |

> **Cầu nối xác suất gộp (Sum-of-Probabilities Bridge)**:  
> $$P(\text{At-Risk}) = P(Y = \text{Fail}) + P(Y = \text{Withdrawn})$$  
> $$P(\text{Not At-Risk}) = P(Y = \text{Distinction}) + P(Y = \text{Pass})$$  
> Kiểm định McNemar trên tập Test xác nhận tính tương đương thống kê ($p > 0.40$), chứng minh việc gộp nhãn bảo toàn trọn vẹn thông tin và tối ưu hóa độ nhạy phát hiện rủi ro sớm.

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
- Khi chia ngẫu nhiên thông thường (`Stratified Random Split`): Có tới **$812$ sinh viên duy nhất xuất hiện đồng thời ở cả tập Train và tập Test** ($16.90\%$ trên $4,804$ sinh viên duy nhất của tập Test, tương ứng $825$ bản ghi hay $16.87\%$ trên $4,889$ bản ghi Test; hoặc $812 / 4,889 = 16.61\%$ nếu tính trên tổng bản ghi Test). Do đặc điểm nhân khẩu học (`gender`, `age_band`, `region`, `imd_band`) của cùng một sinh viên không đổi, điều này có thể dẫn đến rò rỉ thông tin cá nhân.
- Khi áp dụng `Stratified Group Split` (`StratifiedGroupKFold` theo `id_student`): Số sinh viên trùng lặp giữa Train - Test, Train - Val, Val - Test **chính xác bằng $0$ (Triệt tiêu rò rỉ danh tính 100%)**!
- Cả 2 tùy chọn chia tập đều được tích hợp đầy đủ trong lớp `OULADPreprocessor` tại [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb) và sẵn sàng cho các vòng thực nghiệm mô hình tiếp theo.

---

## 7. XỬ LÝ MẤT CÂN BẰNG LỚP & ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH (T4.8)
Biểu đồ minh họa: [reports/figures/20_class_imbalance_and_resampling.png](file:///c:/Users/mrore/Project_I/reports/figures/20_class_imbalance_and_resampling.png)

### 7.1. Phân tích Mức độ Mất cân bằng:
- **Bài toán Nhị phân (`at_risk`)**: Tỷ lệ khá cân bằng: `Not At-Risk (0)` chiếm $47.20\%$ vs `At-Risk (1)` chiếm $52.80\%$. Khi áp dụng SMOTE trên tập Train, số lượng mẫu được cân bằng chính xác $12,046 : 12,046$ ($24,092$ mẫu).
- **Bài toán 4 lớp gốc (`final_result`)**: Mất cân bằng đáng kể giữa nhóm `Distinction` ($9.28\%$) so với `Pass` ($36.94\%$), `Fail` ($21.64\%$), `Withdrawn` ($32.15\%$).

### 7.2. Kết quả Thực nghiệm Định lượng trên Tập Test Độc lập:
Nghiên cứu đã huấn luyện và đánh giá đối kháng các giải pháp tiền xử lý trên tập Test gồm **$4,889$ mẫu**:

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
Tích hợp tại Mục 10 trong [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb)  
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

- [x] **Jupyter Notebook Tiền xử lý Tự chủ Toàn diện (bao gồm lớp `OULADPreprocessor` & 9/9 Unit Tests PASSED)**: [`notebooks/03_preprocessing.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb) (trực quan hóa chi tiết từng bước 4.1 - 4.8).
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
