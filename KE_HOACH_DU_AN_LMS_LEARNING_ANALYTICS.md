# KẾ HOẠCH DỰ ÁN: LMS TIÊN TIẾN VÀ LEARNING ANALYTICS

## Phân tích dữ liệu học tập của Sinh viên trên LMS để xem Sinh viên đang học như thế nào và ai có nguy cơ gặp khó khăn

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Dataset OULAD](#3-dataset-oulad)
4. [Kế hoạch chi tiết 12 tuần](#4-kế-hoạch-chi-tiết-12-tuần)
5. [Công nghệ và công cụ](#5-công-nghệ-và-công-cụ)
6. [Đánh giá và tiêu chí thành công](#6-đánh-giá-và-tiêu-chí-thành-công)
7. [Rủi ro và giải pháp](#7-rủi-ro-và-giải-pháp)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Bài toán nghiên cứu

Phân tích dữ liệu học tập (Learning Analytics) của sinh viên trên hệ thống quản lý học tập (LMS) nhằm:
- **Hiểu hành vi học tập**: Sinh viên đang học như thế nào? Mức độ tương tác (engagement) ra sao?
- **Dự đoán nguy cơ**: Ai là sinh viên có nguy cơ gặp khó khăn hoặc bỏ học?
- **Cảnh báo sớm**: Xây dựng hệ thống phát hiện sớm sinh viên có nguy cơ để can thiệp kịp thời.

### 1.2. Câu hỏi nghiên cứu

| # | Câu hỏi nghiên cứu |
|---|---------------------|
| RQ1 | Các chỉ số tương tác nào trên LMS (số lần click, thời gian, số lần nộp bài) có mối tương quan mạnh nhất với kết quả học tập cuối kỳ? |
| RQ2 | Có thể dự đoán sinh viên có nguy cơ gặp khó khăn từ dữ liệu LMS sớm trong khóa học (trước 25%, 50% thời lượng) không? |
| RQ3 | Những đặc trưng nào đóng vai trò quan trọng nhất trong việc dự đoán kết quả học tập? |

### 1.3. Phạm vi

- **Dataset**: Open University Learning Analytics Dataset (OULAD)
- **Đối tượng**: Sinh viên đại học trong các khóa học trực tuyến
- **Phương pháp**: Kết hợp EDA, Machine Learning (Classification), và Explainable AI (XAI)
- **Kết quả đầu ra**: Mô hình dự đoán + Demo trực quan + Báo cáo nghiên cứu

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Student Engagement Framework (SEDL)

Dựa trên bài báo **Bergdahl et al. (2024)** - *"Engagements in digital learning: A systematic review of learning analytics research on student engagement"* (International Journal of Educational Technology in Higher Education, 21:63), student engagement gồm **4 chiều**:

| Chiều engagement | Chỉ số đo lường (Indicators) | Dữ liệu LMS tương ứng |
|---|---|---|
| **Behavioural** (Hành vi) | Participation, Time on task, Homework completion, Attendance, Effort | Số lần click, thời gian truy cập, số bài nộp, login frequency |
| **Cognitive** (Nhận thức) | Critical thinking, Self-regulation, Deep learning, Self-efficacy | Thời gian xem tài liệu vs. quiz, pattern truy cập có hệ thống |
| **Emotional** (Cảm xúc) | Enjoyment, Interest, Satisfaction | Tần suất truy cập tự nguyện, forum interactions |
| **Social** (Xã hội) | Interaction with peers/educators, Asking for help | Forum posts, collaborative activities |

> **Phát hiện chính từ bài báo**: 95% nghiên cứu LA chỉ đo **behavioural engagement** (clicks, time). Rất ít nghiên cứu khám phá đa chiều. Đây là **cơ hội nghiên cứu** để dự án khai thác.

### 2.2. Disengagement Indicators

| Behavioural | Cognitive | Emotional | Social |
|---|---|---|---|
| Task incompletion | Confusion | Boredom | Decreased interaction |
| Absence | Apathy | Disinterest | Social isolation |
| Lurking | Unfocussed | Frustration | Ignoring others |
| Drop out | Distracted | Anger | Giving up on inclusion |

### 2.3. Liên hệ với dự án

- Dự án sử dụng **OULAD** - dataset chứa dữ liệu tương tác LMS (clicks theo loại activity, thời gian nộp bài, kết quả assessment)
- Áp dụng framework SEDL để **thiết kế đặc trưng** (feature engineering) theo nhiều chiều engagement
- Xây dựng mô hình **dự đoán nguy cơ** (at-risk prediction) - một trong những ứng dụng phổ biến nhất của LA (theo survey của Bergdahl et al.)

---

## 3. DATASET OULAD

### 3.1. Giới thiệu

**Open University Learning Analytics Dataset (OULAD)** là bộ dữ liệu mở từ The Open University (Anh), chứa dữ liệu của **~32,000 sinh viên** trong **22 khóa học** qua **4 kỳ học**.

- **Nguồn**: https://analyse.kmi.open.ac.uk/open_dataset
- **Tham khảo**: Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017). "Open university learning analytics dataset." *Scientific Data*, 4, 170171.

### 3.2. Cấu trúc dữ liệu (7 bảng CSV)

```
OULAD/
├── courses.csv              # Thông tin khóa học (module + presentation)
├── assessments.csv          # Thông tin bài kiểm tra (TMA, CMA, Exam)
├── studentInfo.csv          # Thông tin sinh viên (demographics + final_result)
├── studentRegistration.csv  # Ngày đăng ký/hủy đăng ký
├── studentAssessment.csv    # Điểm bài kiểm tra của từng sinh viên
├── studentVle.csv           # Tương tác VLE (clicks) - DỮ LIỆU CHÍNH
└── vle.csv                  # Thông tin về các tài nguyên VLE (activity types)
```

### 3.3. Mô tả chi tiết từng bảng

#### courses.csv

| Cột | Mô tả |
|-----|--------|
| `code_module` | Mã môn học (AAA, BBB, CCC, DDD, EEE, FFF, GGG) |
| `code_presentation` | Mã kỳ học (2013B, 2013J, 2014B, 2014J) |
| `module_presentation_length` | Số ngày của khóa học |

#### assessments.csv

| Cột | Mô tả |
|-----|--------|
| `id_assessment` | ID bài đánh giá |
| `code_module`, `code_presentation` | Khóa ngoại đến courses |
| `assessment_type` | Loại: **TMA** (Tutor Marked), **CMA** (Computer Marked), **Exam** |
| `date` | Ngày deadline (tính từ ngày bắt đầu khóa học) |
| `weight` | Trọng số điểm (%) |

#### studentInfo.csv - Bảng quan trọng nhất (chứa biến mục tiêu)

| Cột | Mô tả |
|-----|--------|
| `id_student` | ID sinh viên |
| `code_module`, `code_presentation` | Khóa ngoại |
| `gender` | M / F |
| `region` | Vùng địa lý |
| `highest_education` | Trình độ học vấn cao nhất |
| `imd_band` | Chỉ số IMD (mức thu nhập vùng) |
| `age_band` | Nhóm tuổi (0-35, 35-55, 55<=) |
| `num_of_prev_attempts` | Số lần học lại |
| `studied_credits` | Số tín chỉ đang học |
| `disability` | Có khuyết tật? (Y/N) |
| **`final_result`** | **KẾT QUẢ CUỐI KỲ**: Distinction, Pass, Fail, Withdrawn |

#### studentRegistration.csv

| Cột | Mô tả |
|-----|--------|
| `date_registration` | Ngày đăng ký (có thể âm = đăng ký trước khóa học) |
| `date_unregistration` | Ngày hủy đăng ký (NULL nếu hoàn thành) |

#### studentAssessment.csv

| Cột | Mô tả |
|-----|--------|
| `id_assessment` | Khóa ngoại đến assessments |
| `id_student` | Khóa ngoại đến studentInfo |
| `date_submitted` | Ngày nộp bài thực tế |
| `is_banked` | Bài được chuyển từ kỳ trước? |
| `score` | Điểm (0-100) |

#### studentVle.csv - Bảng dữ liệu tương tác chính

| Cột | Mô tả |
|-----|--------|
| `code_module`, `code_presentation` | Khóa ngoại |
| `id_student` | Khóa ngoại |
| `id_site` | ID tài nguyên VLE |
| `date` | Ngày tương tác |
| `sum_click` | **Tổng số click trong ngày đó** |

#### vle.csv

| Cột | Mô tả |
|-----|--------|
| `id_site` | ID tài nguyên |
| `code_module`, `code_presentation` | Khóa ngoại |
| `activity_type` | Loại tài nguyên: oucontent, forumng, subpage, homepage, quiz, resource, url, ouwiki, glossary, questionnaire, page, ouelluminate, dualpane, dataplus, folder, oucollaborate, htmlactivity, externalquiz, repeatactivity, sharedsubpage |
| `week_from`, `week_to` | Tuần hiệu lực |

### 3.4. Thống kê tổng quan

| Thông số | Giá trị |
|----------|---------|
| Tổng sinh viên | ~32,593 |
| Tổng khóa học | 22 (7 modules x 2-4 presentations) |
| Tổng tương tác VLE | ~10,655,280 records |
| Tổng bài đánh giá | 206 |
| Kết quả Pass | ~38% |
| Kết quả Fail | ~16% |
| Kết quả Withdrawn | ~33% |
| Kết quả Distinction | ~13% |

### 3.5. Biến mục tiêu (Target Variable)

Chuyển đổi `final_result` thành bài toán **phân loại nhị phân**:

| Nhóm | Giá trị gốc | Label |
|------|-------------|-------|
| **Thành công** | Distinction, Pass | **0** (Not At-Risk) |
| **Gặp khó khăn** | Fail, Withdrawn | **1** (At-Risk) |

> **Lưu ý**: Có thể thử nghiệm cả phân loại đa lớp (4 class) ở tuần 8.

---

## 4. KẾ HOẠCH CHI TIẾT 12 TUẦN

---

### TUẦN 1: Khởi động nghiên cứu

**Mục tiêu**: Hiểu bài toán, xác định phạm vi, đọc tài liệu nền tảng.

#### Công việc chi tiết:

| # | Công việc | Thời gian | Output |
|---|-----------|-----------|--------|
| 1.1 | Đọc kỹ bài báo **Bergdahl et al. (2024)** - tập trung vào: SEDL Framework, RQ4 (indicators đo engagement), Table 5 (operationalisation) | 2 ngày | Ghi chú tóm tắt bài báo |
| 1.2 | Đọc thêm 3-5 bài báo liên quan (xem danh sách bên dưới) | 2 ngày | Bảng tổng hợp literature |
| 1.3 | Tìm hiểu tổng quan về OULAD dataset (đọc paper gốc của Kuzilek et al., 2017) | 1 ngày | Sơ đồ quan hệ các bảng |
| 1.4 | Xác định câu hỏi nghiên cứu chính thức (RQ1, RQ2, RQ3) | 0.5 ngày | Tài liệu RQ |
| 1.5 | Lập kế hoạch công cụ: cài đặt Python, Jupyter, thư viện | 0.5 ngày | Môi trường sẵn sàng |

#### Bài báo cần đọc:

1. **Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017)**. Open university learning analytics dataset. *Scientific Data*, 4, 170171.
2. **Bergdahl, N., Bond, M., Sjoden, J., and Dane, M. (2024)**. Engagements in digital learning: A systematic review of learning analytics research on student engagement. *IJETHE*, 21:63.
3. **Hlosta, M., Zdrahal, Z., and Zendulka, J. (2017)**. Ouroboros: early identification of at-risk students without models based on legacy data. *LAK*.
4. **Umer, R., Susnjak, T., Mathrani, A., and Suriadi, S. (2017)**. On predicting academic performance with process mining in learning analytics. *JRIS*.
5. **Jayaprakash, S. M., Moody, E. W., Lauria, E. J., Regan, J. R., and Baron, J. D. (2014)**. Early alert of academically at-risk students: An open source analytics initiative. *JLAL*, 1(1), 6-47.

#### Deliverable Tuần 1:
- [x] File tóm tắt literature review (bảng so sánh các nghiên cứu) - [01_literature_review_summary.md](file:///c:/Users/mrore/Project_I/references/01_literature_review_summary.md)
- [x] Sơ đồ quan hệ dữ liệu OULAD - [02_oulad_data_architecture.md](file:///c:/Users/mrore/Project_I/references/02_oulad_data_architecture.md)
- [x] Câu hỏi nghiên cứu chính thức - [03_research_questions_and_methodology.md](file:///c:/Users/mrore/Project_I/references/03_research_questions_and_methodology.md)
- [x] Môi trường lập trình sẵn sàng - [requirements.txt](file:///c:/Users/mrore/Project_I/requirements.txt) & [check_env.py](file:///c:/Users/mrore/Project_I/src/check_env.py)

---

### TUẦN 2: Thu thập và tìm hiểu dữ liệu

**Mục tiêu**: Tải, nạp OULAD, đánh giá tổng quan cấu trúc dữ liệu.

#### Công việc chi tiết:

| # | Công việc | Thời gian | Output |
|---|-----------|-----------|--------|
| 2.1 | Tải dataset OULAD từ trang chính thức | 0.5 ngày | Thư mục data/ chứa 7 file CSV |
| 2.2 | Nạp dữ liệu vào Pandas, kiểm tra shape, dtypes, head/tail | 0.5 ngày | Notebook 01_data_loading.ipynb |
| 2.3 | Kiểm tra missing values, duplicates cho từng bảng | 1 ngày | Bảng thống kê missing |
| 2.4 | Vẽ sơ đồ ERD (Entity Relationship Diagram) thực tế | 0.5 ngày | Sơ đồ ERD |
| 2.5 | Thực hiện merge cơ bản giữa các bảng, kiểm tra tính toàn vẹn dữ liệu | 1 ngày | DataFrame merged |
| 2.6 | Thống kê mô tả cơ bản (phân bố final_result, demographics) | 0.5 ngày | Biểu đồ phân bố |

#### Code mẫu Tuần 2:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nạp dữ liệu
courses = pd.read_csv('data/courses.csv')
assessments = pd.read_csv('data/assessments.csv')
student_info = pd.read_csv('data/studentInfo.csv')
student_reg = pd.read_csv('data/studentRegistration.csv')
student_assess = pd.read_csv('data/studentAssessment.csv')
student_vle = pd.read_csv('data/studentVle.csv')
vle = pd.read_csv('data/vle.csv')

# Kiểm tra tổng quan
for name, df in [('courses', courses), ('assessments', assessments),
                  ('studentInfo', student_info), ('studentReg', student_reg),
                  ('studentAssess', student_assess), ('studentVle', student_vle),
                  ('vle', vle)]:
    print(f"\n{'='*50}")
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Dtypes:\n{df.dtypes}")

# Phân bố kết quả
student_info['final_result'].value_counts().plot(kind='bar',
    color=['green','blue','red','orange'])
plt.title('Distribution of Final Results')
plt.ylabel('Count')
plt.show()
```

#### Deliverable Tuần 2:
- [ ] Notebook 01_data_loading.ipynb - nạp và kiểm tra 7 bảng
- [ ] Bảng thống kê missing values and data quality
- [ ] Sơ đồ ERD thực tế
- [ ] Nhận xét ban đầu về chất lượng dữ liệu

---

### TUẦN 3: Khám phá dữ liệu (EDA)

**Mục tiêu**: Phân tích mối tương quan giữa mức độ tương tác (số lần click, nộp bài) với kết quả cuối kỳ.

#### Công việc chi tiết:

| # | Công việc | Thời gian | Output |
|---|-----------|-----------|--------|
| 3.1 | Phân tích phân bố clicks theo final_result (boxplot, violin plot) | 1 ngày | Biểu đồ so sánh |
| 3.2 | Phân tích mối quan hệ giữa score assessment với final_result | 0.5 ngày | Scatter/box plots |
| 3.3 | Phân tích temporal patterns: xu hướng click theo thời gian (tuần) cho từng nhóm kết quả | 1 ngày | Time series plots |
| 3.4 | Phân tích clicks theo activity_type (quiz, forum, oucontent, resource...) | 1 ngày | Heatmap / grouped bar |
| 3.5 | Phân tích ảnh hưởng của demographics (gender, age, education, imd_band) | 0.5 ngày | Cross-tabulation |
| 3.6 | Tính correlation matrix và kiểm định thống kê (chi-square, t-test, ANOVA) | 1 ngày | Correlation heatmap |

#### Phân tích chính cần thực hiện:

**A. Engagement vs. Outcome**

```python
# Tổng clicks per student
total_clicks = student_vle.groupby(
    ['id_student', 'code_module', 'code_presentation']
)['sum_click'].sum().reset_index()
total_clicks.columns = ['id_student', 'code_module', 'code_presentation', 'total_clicks']

# Merge với kết quả
merged = total_clicks.merge(student_info,
    on=['id_student', 'code_module', 'code_presentation'])

# So sánh
sns.boxplot(x='final_result', y='total_clicks', data=merged,
            order=['Distinction', 'Pass', 'Fail', 'Withdrawn'])
plt.title('Total VLE Clicks by Final Result')
```

**B. Temporal Engagement Pattern**

```python
# Click theo tuần
student_vle_merged = student_vle.merge(vle[['id_site','activity_type']], on='id_site')
weekly = student_vle_merged.copy()
weekly['week'] = weekly['date'] // 7  # Chuyển ngày thành tuần

weekly_agg = weekly.groupby(
    ['id_student','code_module','code_presentation','week']
)['sum_click'].sum().reset_index()

weekly_final = weekly_agg.merge(
    student_info[['id_student','code_module','code_presentation','final_result']],
    on=['id_student','code_module','code_presentation'])

# Vẽ xu hướng
weekly_mean = weekly_final.groupby(['week','final_result'])['sum_click'].mean().reset_index()
sns.lineplot(x='week', y='sum_click', hue='final_result', data=weekly_mean)
plt.title('Average Weekly Clicks by Final Result')
```

**C. Activity Type Analysis**

```python
# Clicks theo loại activity
activity_clicks = student_vle.merge(vle[['id_site','activity_type']], on='id_site')
activity_agg = activity_clicks.groupby(
    ['id_student','code_module','code_presentation','activity_type']
)['sum_click'].sum().unstack(fill_value=0)
```

#### Deliverable Tuan 3:
- [ ] Notebook 02_eda.ipynb - phân tích khám phá đầy đủ
- [ ] Ít nhất 10 biểu đồ có ý nghĩa
- [ ] Bảng tóm tắt các phát hiện chính (key findings)
- [ ] Kết luận sơ bộ về mối quan hệ engagement-outcome

---

### TUẦN 4: Tiền xử lý dữ liệu

**Mục tiêu**: Xử lý missing values, chuẩn hóa, loại bỏ ngoại lai, chia tập Train/Val/Test.

#### Công việc chi tiết:

| # | Công việc | Thời gian | Output |
|---|-----------|-----------|--------|
| 4.1 | Xử lý missing values trong studentAssessment (điểm thiếu -> 0 hoặc impute) | 0.5 ngày | Pipeline xử lý |
| 4.2 | Xử lý date_submitted missing, imd_band missing | 0.5 ngày | Dữ liệu sạch |
| 4.3 | Xử lý sinh viên Withdrawn sớm (không có đủ dữ liệu tương tác) | 1 ngày | Chiến lược xử lý |
| 4.4 | Phát hiện và xử lý ngoại lai (outliers) - IQR method hoặc Z-score | 0.5 ngày | Biểu đồ before/after |
| 4.5 | Encoding biến categorical (One-Hot hoặc Label Encoding) | 0.5 ngày | DataFrame encoded |
| 4.6 | Chuẩn hóa dữ liệu số (StandardScaler / MinMaxScaler) | 0.5 ngày | Pipeline scaler |
| 4.7 | Chia tập Train/Val/Test (70/15/15) - **Stratified split** theo final_result | 0.5 ngày | 3 tập dữ liệu |
| 4.8 | Xử lý class imbalance (SMOTE / class_weight / undersampling) | 1 ngày | Chiến lược cân bằng |

#### Chiến lược xử lý missing values:

| Bảng | Cột | Chiến lược |
|------|-----|------------|
| studentInfo | imd_band | Mode imputation hoặc tạo category "Unknown" |
| studentAssessment | score | Nếu không nộp bài -> score = 0 |
| studentAssessment | date_submitted | Nếu missing -> dùng deadline (nộp đúng hạn) |
| studentRegistration | date_unregistration | NULL = hoàn thành khóa (giữ nguyên) |
| studentVle | - | Sinh viên không có click -> tổng click = 0 |

#### Chiến lược chia dữ liệu:

```python
from sklearn.model_selection import train_test_split

# Tạo biến nhị phân
df['at_risk'] = df['final_result'].isin(['Fail', 'Withdrawn']).astype(int)

# Chia 70/15/15 theo stratified
X = df.drop(['final_result', 'at_risk', 'id_student'], axis=1)
y = df['at_risk']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train at-risk ratio: {y_train.mean():.3f}")
```

#### Lưu ý quan trọng:

> **Data Leakage**: Không dùng thông tin từ tương lai để dự đoán. Nếu dự đoán ở tuần thứ 4, chỉ dùng dữ liệu từ tuần 1-4.

> **Group Split**: Cùng một sinh viên học nhiều môn -> cần đảm bảo cùng id_student nằm trong cùng một tập (train hoặc test) để tránh data leakage.

#### Deliverable Tuan 4:
- [ ] Notebook 03_preprocessing.ipynb
- [ ] Pipeline tiền xử lý hoàn chỉnh (reproducible)
- [ ] Dữ liệu sạch, encoded, chuẩn hóa
- [ ] Tập Train/Val/Test đã chia, lưu file

---

### TUẦN 5-6: Xử lý Feature Engineering

**Mục tiêu**: Trích xuất và xây dựng đặc trưng có ý nghĩa cho mô hình dự đoán.

#### Tuần 5: Đặc trưng cơ bản

| # | Feature Group | Đặc trưng | Nguồn |
|---|---------------|-----------|-------|
| 5.1 | **Click-based** | total_clicks - Tổng số lần click | studentVle |
| | | clicks_per_day - Trung bình click/ngày | studentVle |
| | | active_days - Số ngày hoạt động | studentVle |
| | | click_std - Độ lệch chuẩn click/ngày | studentVle |
| 5.2 | **Activity-based** | clicks_oucontent - Click xem nội dung | studentVle + vle |
| | | clicks_forumng - Click diễn đàn | |
| | | clicks_quiz - Click làm quiz | |
| | | clicks_resource - Click tài nguyên | |
| | | clicks_homepage - Click trang chủ | |
| | | clicks_subpage - Click trang con | |
| | | clicks_url - Click link ngoài | |
| | | clicks_glossary - Click thuật ngữ | |
| 5.3 | **Assessment-based** | avg_score - Điểm trung bình các bài | studentAssessment |
| | | num_submissions - Số bài đã nộp | |
| | | submission_rate - Tỷ lệ nộp bài | |
| | | avg_days_before_deadline - TB nộp sớm bao nhiêu ngày | |
| | | late_submissions - Số lần nộp trễ | |
| | | first_score - Điểm bài đầu tiên | |
| 5.4 | **Demographics** | gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability | studentInfo |

#### Tuần 6: Đặc trưng nâng cao (Early Warning)

| # | Feature Group | Đặc trưng | Mô tả |
|---|---------------|-----------|-------|
| 6.1 | **Temporal** | clicks_week_1_to_4 | Tổng click 4 tuần đầu |
| | | clicks_trend | Hệ số góc (slope) của xu hướng click qua tuần |
| | | engagement_decay | Click tuần gần / Click tuần đầu - đo sự giảm sút |
| | | days_since_last_activity | Số ngày không hoạt động gần nhất |
| | | consistency_score | Entropy / Std của phân bố click theo ngày |
| 6.2 | **Regularity** | login_regularity | Số tuần có hoạt động / Tổng số tuần |
| | | weekend_ratio | Tỷ lệ click vào cuối tuần |
| | | burst_count | Số đợt học tập "bùng nổ" (click > mean + 2*std) |
| 6.3 | **Early Warning** | Snapshot features tại **25%**, **50%**, **75%** thời lượng khóa học | Cho phép dự đoán sớm |
| 6.4 | **Registration** | days_before_start | Đăng ký sớm bao nhiêu ngày |
| | | is_late_registration | Đăng ký trễ? |

#### Code mẫu Feature Engineering:

```python
def create_features(student_vle, vle, student_assess, assessments,
                    student_info, courses, cutoff_pct=1.0):
    """
    Tạo features cho mỗi sinh viên.
    cutoff_pct: tỷ lệ thời gian khóa học (0.25, 0.5, 0.75, 1.0)
    """
    # Merge VLE với activity type
    vle_merged = student_vle.merge(vle[['id_site','activity_type']], on='id_site')

    # Lọc theo cutoff
    vle_merged = vle_merged.merge(courses, on=['code_module','code_presentation'])
    cutoff_day = vle_merged['module_presentation_length'] * cutoff_pct
    vle_merged = vle_merged[vle_merged['date'] <= cutoff_day]

    # Click features
    click_features = vle_merged.groupby(
        ['id_student','code_module','code_presentation']
    ).agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique'),
        click_std=('sum_click', 'std'),
        click_mean=('sum_click', 'mean'),
    ).reset_index()

    # Activity-type features (pivot)
    activity_pivot = vle_merged.groupby(
        ['id_student','code_module','code_presentation','activity_type']
    )['sum_click'].sum().unstack(fill_value=0).reset_index()
    activity_pivot.columns = [
        f'clicks_{c}' if c not in ['id_student','code_module','code_presentation']
        else c for c in activity_pivot.columns
    ]

    # Assessment features
    assess_merged = student_assess.merge(assessments, on='id_assessment')
    assess_merged['days_early'] = assess_merged['date'] - assess_merged['date_submitted']
    assess_merged['is_late'] = (assess_merged['days_early'] < 0).astype(int)

    assess_features = assess_merged.groupby(['id_student']).agg(
        avg_score=('score', 'mean'),
        num_submissions=('score', 'count'),
        avg_days_early=('days_early', 'mean'),
        late_submissions=('is_late', 'sum'),
        first_score=('score', 'first'),
    ).reset_index()

    # Merge all
    features = click_features.merge(
        activity_pivot,
        on=['id_student','code_module','code_presentation'], how='left')
    features = features.merge(assess_features, on='id_student', how='left')
    features = features.merge(
        student_info,
        on=['id_student','code_module','code_presentation'], how='left')

    return features
```

#### Deliverable Tuan 5-6:
- [ ] Notebook 04_feature_engineering.ipynb
- [ ] Bảng mô tả tất cả features (tên, mô tả, kiểu, nguồn)
- [ ] Feature matrix cho 4 snapshot: 25%, 50%, 75%, 100%
- [ ] Phân tích feature importance sơ bộ (correlation with target)

---

### TUẦN 7: Xây dựng mô hình Baseline

**Mục tiêu**: Xây dựng và đánh giá các mô hình cơ bản.

#### Mô hình Baseline cần thử:

| # | Mô hình | Thư viện | Ưu điểm |
|---|---------|----------|---------|
| 1 | **Logistic Regression** | sklearn | Đơn giản, interpretable, baseline tốt |
| 2 | **Decision Tree** | sklearn | Trực quan, dễ giải thích |
| 3 | **Random Forest** | sklearn | Mạnh, ít overfitting hơn DT |
| 4 | **Support Vector Machine** | sklearn | Hiệu quả cho high-dimensional |
| 5 | **K-Nearest Neighbors** | sklearn | Non-parametric, simple |

#### Metrics đánh giá:

| Metric | Ý nghĩa | Tại sao quan trọng? |
|--------|---------|---------------------|
| **Accuracy** | Tỷ lệ đúng tổng thể | Cái nhìn chung (nhưng misleading khi imbalanced) |
| **Precision** | TP / (TP + FP) | Trong các SV được cảnh báo, bao nhiêu thực sự at-risk? |
| **Recall** | TP / (TP + FN) | Trong tất cả SV at-risk, phát hiện được bao nhiêu? **QUAN TRỌNG NHẤT** |
| **F1-Score** | Harmonic mean P and R | Cân bằng Precision-Recall |
| **AUC-ROC** | Khả năng phân biệt tổng thể | Robust với threshold |
| **AUC-PR** | Quan trọng khi imbalanced | Tập trung vào positive class |

> **Ưu tiên Recall**: Trong cảnh báo sớm, bỏ sót sinh viên at-risk (False Negative) nghiêm trọng hơn cảnh báo nhầm (False Positive).

#### Code mẫu:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(
        class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', n_estimators=100, random_state=42),
    'SVM': SVC(class_weight='balanced', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

    results[name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc_roc': roc_auc_score(y_val, y_prob) if y_prob is not None else None,
    }
    print(f"\n{'='*50}\n{name}")
    print(classification_report(y_val, y_pred))
```

#### Deliverable Tuan 7:
- [ ] Notebook 05_baseline_models.ipynb
- [ ] Bảng so sánh hiệu suất tất cả baseline models
- [ ] Confusion matrix cho từng mô hình
- [ ] ROC curves so sánh
- [ ] Chọn 2-3 mô hình tốt nhất để tinh chỉnh

---

### TUẦN 8: Mô hình nâng cao và Tinh chỉnh

**Mục tiêu**: Áp dụng mô hình nâng cao, hyperparameter tuning, thử nghiệm early prediction.

#### Mô hình nâng cao:

| # | Mô hình | Thư viện | Mô tả |
|---|---------|----------|-------|
| 1 | **XGBoost** | xgboost | Gradient Boosting, SOTA cho tabular data |
| 2 | **LightGBM** | lightgbm | Nhanh hơn XGBoost, hiệu quả với dataset lớn |
| 3 | **CatBoost** | catboost | Tự xử lý categorical features |
| 4 | **Neural Network** | sklearn / pytorch | MLP cho comparison |
| 5 | **Ensemble / Stacking** | sklearn | Kết hợp nhiều mô hình |

#### Hyperparameter Tuning:

```python
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, ratio_neg_pos],
    'min_child_weight': [1, 3, 5],
}

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric='logloss', random_state=42)

search = RandomizedSearchCV(
    xgb_model, param_grid, n_iter=50, scoring='recall',
    cv=5, verbose=1, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best recall: {search.best_score_:.4f}")
```

#### Early Prediction Experiment:

```python
# Thử nghiệm dự đoán sớm ở các thời điểm khác nhau
time_points = [0.25, 0.50, 0.75, 1.00]
early_results = {}

for t in time_points:
    features_t = create_features(..., cutoff_pct=t)
    X_t = preprocess(features_t)
    model_t = best_model.fit(X_train_t, y_train_t)
    y_pred_t = model_t.predict(X_val_t)
    early_results[f'{int(t*100)}%'] = {
        'recall': recall_score(y_val_t, y_pred_t),
        'f1': f1_score(y_val_t, y_pred_t),
        'auc': roc_auc_score(y_val_t, model_t.predict_proba(X_val_t)[:,1]),
    }
```

#### Deliverable Tuan 8:
- [ ] Notebook 06_advanced_models.ipynb
- [ ] Bảng so sánh baseline vs. advanced models
- [ ] Kết quả early prediction (25%, 50%, 75%)
- [ ] Best model đã tuning + saved model file (.pkl)
- [ ] Đánh giá trên tập **Test** (chỉ chạy 1 lần cuối)

---

### TUẦN 9: Giải thích mô hình (Explainable AI)

**Mục tiêu**: Làm rõ tại sao mô hình đưa ra dự đoán - tăng tính tin cậy và ứng dụng thực tiễn.

#### Phương pháp giải thích:

| # | Phương pháp | Thư viện | Mức độ | Mô tả |
|---|-------------|----------|--------|-------|
| 1 | **Feature Importance** (built-in) | xgboost / sklearn | Global | Top features quan trọng nhất |
| 2 | **Permutation Importance** | sklearn | Global | Đo tác động khi xáo trộn từng feature |
| 3 | **SHAP** (SHapley Additive exPlanations) | shap | Global + Local | Giải thích chi tiết nhất |
| 4 | **LIME** (Local Interpretable Model-agnostic) | lime | Local | Giải thích từng dự đoán cụ thể |
| 5 | **Partial Dependence Plot** | sklearn | Global | Tác động biên của từng feature |

#### Code mẫu SHAP:

```python
import shap

# Tạo explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global: Feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)  # Beeswarm plot

# Local: Giải thích 1 sinh viên cụ thể (at-risk)
idx = 0  # Chọn sinh viên at-risk
shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx])

# Dependence plot cho top feature
shap.dependence_plot("total_clicks", shap_values, X_test)
```

#### Phân tích cần thực hiện:

1. **Top 10 features quan trọng nhất** - Kết hợp built-in importance + SHAP + Permutation
2. **Case studies**: Chọn 3-5 sinh viên cụ thể (2 at-risk đúng, 1 at-risk sai, 1 safe đúng, 1 safe sai) -> Giải thích bằng SHAP
3. **Partial Dependence**: Vẽ PDP cho top 3 features -> mô tả tác động phi tuyến
4. **Liên hệ với lý thuyết**: Kết quả feature importance phản ánh gì về engagement theo framework SEDL?

#### Deliverable Tuan 9:
- [ ] Notebook 07_model_explanation.ipynb
- [ ] SHAP summary plot (global)
- [ ] SHAP force plot (local) cho case studies
- [ ] Bảng Top 10 important features + giải thích ý nghĩa
- [ ] Kết luận về mối quan hệ giữa các chỉ số engagement và kết quả

---

### TUẦN 10: Xây dựng Demo

**Mục tiêu**: Tạo ứng dụng demo trực quan cho hệ thống cảnh báo sớm.

#### Lựa chọn công nghệ:

| Option | Công nghệ | Ưu điểm |
|--------|-----------|---------|
| **A (Khuyến nghị)** | **Streamlit** | Nhanh, dễ, Python-native, phù hợp ML demo |
| B | Gradio | Đơn giản hơn, tốt cho ML interface |
| C | Flask + HTML/JS | Linh hoạt nhưng mất thời gian hơn |

#### Tính năng Demo:

| # | Tính năng | Mô tả |
|---|-----------|-------|
| 1 | **Dashboard tổng quan** | Biểu đồ phân bố sinh viên theo risk level, thống kê tổng |
| 2 | **Bảng cảnh báo** | Danh sách sinh viên at-risk, sắp xếp theo risk score |
| 3 | **Dự đoán cá nhân** | Nhập thông tin sinh viên -> hiển thị risk score + SHAP explanation |
| 4 | **Phân tích xu hướng** | Biểu đồ engagement theo thời gian cho từng sinh viên/nhóm |
| 5 | **So sánh early vs. late** | Kết quả dự đoán ở 25%, 50%, 75% khóa học |

#### Code mẫu Streamlit:

```python
import streamlit as st
import pandas as pd
import joblib
import shap

# Load model
model = joblib.load('models/best_model.pkl')
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="LMS Early Warning System", layout="wide")
st.title("He thong Canh bao Som Sinh vien")

# Sidebar
st.sidebar.header("Bo loc")
module = st.sidebar.selectbox("Chon mon hoc", modules)
time_point = st.sidebar.slider("Thoi diem du doan (%)", 25, 100, 50, 25)

# Dashboard chinh
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tong SV", total_students)
col2.metric("At-Risk", at_risk_count, delta=f"{at_risk_pct:.1f}%")
col3.metric("Safe", safe_count)
col4.metric("Model Recall", f"{recall:.1%}")

# Bang canh bao
st.subheader("Danh sach Sinh vien Can Chu y")
st.dataframe(at_risk_df.style.background_gradient(
    subset=['risk_score'], cmap='Reds'))

# Chi tiet sinh vien
student_id = st.selectbox("Chon sinh vien", student_ids)
if student_id:
    student_data = get_student_features(student_id)
    risk_score = model.predict_proba(student_data)[:, 1][0]
    st.metric("Risk Score", f"{risk_score:.2%}")

    # SHAP explanation
    sv = explainer.shap_values(student_data)
    fig = shap.force_plot(
        explainer.expected_value, sv[0], student_data, matplotlib=True)
    st.pyplot(fig)
```

#### Deliverable Tuan 10:
- [ ] Ứng dụng Streamlit hoàn chỉnh (app.py)
- [ ] File requirements.txt
- [ ] Hướng dẫn chạy demo (README.md)
- [ ] Screenshots / Recording demo
- [ ] Model files saved (models/)

---

### TUẦN 11: Viết báo cáo

**Mục tiêu**: Viết báo cáo nghiên cứu hoàn chỉnh.

#### Cấu trúc báo cáo:

| Phần | Nội dung | Số trang |
|------|---------|----------|
| **Trang bìa** | Tên dự án, tác giả, ngày, GVHD | 1 |
| **Mục lục** | Tự động | 1 |
| **Tóm tắt (Abstract)** | Tóm tắt vấn đề, phương pháp, kết quả chính | 1 |
| **Chuong 1: Giới thiệu** | Bối cảnh, vấn đề, mục tiêu, câu hỏi NC, phạm vi | 3-4 |
| **Chuong 2: Cơ sở lý thuyết** | Student engagement (SEDL), Learning Analytics, Bài báo Bergdahl et al. (2024), Các nghiên cứu liên quan | 5-7 |
| **Chuong 3: Dataset va Phuong phap** | Mô tả OULAD, Tiền xử lý, Feature Engineering, Mô hình, Metrics | 5-7 |
| **Chuong 4: Kết quả va Thảo luận** | EDA findings, Kết quả baseline, Kết quả advanced, Early prediction, SHAP analysis, So sánh với literature | 8-10 |
| **Chuong 5: Demo** | Mô tả hệ thống, Screenshots, Kiến trúc | 3-4 |
| **Chuong 6: Kết luận** | Tóm tắt đóng góp, Hạn chế, Hướng phát triển | 2-3 |
| **Tài liệu tham khảo** | APA 7th format | 2-3 |
| **Phụ lục** | Code snippets, Bảng kết quả đầy đủ, Biểu đồ phụ | 3-5 |

#### Lưu ý viết báo cáo:

1. **Liên kết lý thuyết - thực nghiệm**: Luôn giải thích kết quả dưới góc nhìn framework SEDL
2. **Trích dẫn**: Sử dụng đúng APA 7, trích dẫn Bergdahl et al. (2024) cho phần lý thuyết engagement
3. **Biểu đồ**: Mỗi biểu đồ phải có caption rõ ràng, đánh số
4. **Reproducibility**: Cung cấp link GitHub repo (nếu có)

#### Deliverable Tuan 11:
- [ ] Bản thảo báo cáo hoàn chỉnh (Word/LaTeX)
- [ ] Tất cả biểu đồ/hình ảnh chất lượng cao
- [ ] Tài liệu tham khảo đầy đủ

---

### TUẦN 12: Thuyết trình

**Mục tiêu**: Chuẩn bị và thực hiện thuyết trình.

#### Cấu trúc slide (15-20 slide):

| Slide | Nội dung | Thời gian |
|-------|---------|-----------|
| 1 | Trang bìa | - |
| 2 | Mục lục | 0.5 phút |
| 3-4 | Bài toán va Motivation | 2 phút |
| 5-6 | Cơ sở lý thuyết (SEDL Framework) | 2 phút |
| 7-8 | Dataset OULAD va EDA highlights | 3 phút |
| 9-10 | Phương pháp (Pipeline tổng quan) | 2 phút |
| 11-13 | Kết quả chính (Bảng so sánh, Early prediction, SHAP) | 4 phút |
| 14-15 | Demo (Screenshots hoặc live demo) | 3 phút |
| 16 | Kết luận va Đóng góp | 1 phút |
| 17 | Hạn chế va Hướng phát triển | 1 phút |
| 18 | Q and A | 2 phút |

#### Deliverable Tuan 12:
- [ ] Slide thuyết trình (PowerPoint/Google Slides)
- [ ] Script thuyết trình (ghi chú cho mỗi slide)
- [ ] Demo sẵn sàng chạy
- [ ] Chuẩn bị trả lời câu hỏi (FAQ list)

---

## 5. CÔNG NGHỆ VÀ CÔNG CỤ

### 5.1. Ngôn ngữ va Environment

| Công cụ | Phiên bản | Mục đích |
|---------|-----------|---------|
| Python | 3.10+ | Ngôn ngữ chính |
| Jupyter Notebook | Latest | Phân tích tương tác |
| VS Code / PyCharm | Latest | IDE |
| Git + GitHub | Latest | Version control |

### 5.2. Thư viện Python

```
# requirements.txt
# Data manipulation
pandas>=2.0
numpy>=1.24

# Visualization
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15

# Machine Learning
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
imbalanced-learn>=0.11    # SMOTE

# Explainability
shap>=0.42
lime>=0.2

# Demo
streamlit>=1.28

# Utilities
joblib>=1.3
tqdm>=4.65
```

### 5.3. Cấu trúc thư mục dự án

```
Project_I/
|-- data/
|   |-- raw/                 # Dataset OULAD goc
|   |-- processed/           # Du lieu da xu ly
|-- notebooks/
|   |-- 01_data_loading.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_preprocessing.ipynb
|   |-- 04_feature_engineering.ipynb
|   |-- 05_baseline_models.ipynb
|   |-- 06_advanced_models.ipynb
|   |-- 07_model_explanation.ipynb
|-- src/
|   |-- data_loader.py       # Functions nap du lieu
|   |-- preprocessing.py     # Pipeline tien xu ly
|   |-- features.py          # Feature engineering
|   |-- models.py            # Training va evaluation
|   |-- explainer.py         # SHAP/LIME wrapper
|-- models/                  # Saved models (.pkl)
|-- demo/
|   |-- app.py               # Streamlit app
|   |-- assets/              # Images, CSS
|-- reports/
|   |-- figures/             # Bieu do xuat
|   |-- report.docx          # Bao cao
|   |-- slides.pptx          # Slide
|-- references/
|   |-- s41239-024-00493-y.pdf
|-- requirements.txt
|-- README.md
|-- .gitignore
```

---

## 6. ĐÁNH GIÁ VÀ TIÊU CHÍ THÀNH CÔNG

### 6.1. Tiêu chí kỹ thuật

| Tiêu chí | Mục tiêu | Ưu tiên |
|----------|---------|---------|
| AUC-ROC (trên Test set) | >= 0.80 | Cao |
| Recall (at-risk class) | >= 0.75 | **Rất cao** |
| F1-Score | >= 0.70 | Cao |
| Early prediction (50% khóa học) Recall | >= 0.65 | Trung bình |
| Số features có giải thích ý nghĩa | Top 10 | Cao |

### 6.2. Tiêu chí dự án

| Tiêu chí | Yêu cầu |
|----------|---------|
| Code quality | Có comment, modular, reproducible |
| Báo cáo | >= 30 trang, đúng format, trích dẫn đầy đủ |
| Demo | Chạy được, có giao diện trực quan |
| Thuyết trình | 15-20 phút, rõ ràng, trả lời được câu hỏi |

---

## 7. RỦI RO VÀ GIẢI PHÁP

| # | Rủi ro | Mức độ | Giải pháp |
|---|--------|--------|-----------|
| 1 | Class imbalance (Withdrawn + Fail chiếm ~49%) | Trung bình | SMOTE, class_weight, focal loss |
| 2 | Data leakage khi dùng dữ liệu tương lai | Cao | Cutoff nghiêm ngặt theo thời gian |
| 3 | Overfitting trên tập nhỏ | Trung bình | Cross-validation, regularization |
| 4 | Feature quá nhiều - curse of dimensionality | Thấp | Feature selection (chi2, mutual info, RFE) |
| 5 | Sinh viên Withdrawn sớm thiếu dữ liệu | Cao | Phân tích riêng hoặc loại bỏ withdrawn < tuần 2 |
| 6 | Demo không chạy được lúc thuyết trình | Trung bình | Chuẩn bị video backup, chạy thử trước |
| 7 | Chậm tiến độ | Trung bình | Buffer 1 tuần, ưu tiên core tasks |

---

## 8. TÀI LIỆU THAM KHẢO

### Bài báo chính:

1. **Bergdahl, N., Bond, M., Sjoden, J., and Dane, M. (2024)**. Engagements in digital learning: A systematic review of learning analytics research on student engagement. *International Journal of Educational Technology in Higher Education*, 21:63. https://doi.org/10.1186/s41239-024-00493-y

### Dataset:

2. **Kuzilek, J., Hlosta, M., and Zdrahal, Z. (2017)**. Open university learning analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

### Learning Analytics and At-risk Prediction:

3. Adnan, M., et al. (2021). Predicting at-risk students at different percentages of course length for early intervention using ML models. *IEEE Access*, 9, 7519-7539.
4. Hlosta, M., Zdrahal, Z., and Zendulka, J. (2017). Ouroboros: Early identification of at-risk students without models based on legacy data. *LAK*.
5. Jayaprakash, S.M., et al. (2014). Early alert of academically at-risk students: An open source analytics initiative. *JLAL*, 1(1), 6-47.

### Student Engagement:

6. Bond, M., and Bedenlier, S. (2019). Facilitating student engagement through educational technology: Towards a conceptual framework. *JIME*.
7. Fredricks, J. A., Blumenfeld, P. C., and Paris, A. H. (2004). School engagement: Potential of the concept, state of the evidence. *RER*, 74(1), 59-109.
8. Henrie, C. R., Halverson, L. R., and Graham, C. R. (2015). Measuring student engagement in technology-mediated learning: A review. *Computers and Education*, 90, 36-53.

### Machine Learning and XAI:

9. Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
10. Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.

---

> **Ghi chu**: Tai lieu nay la ke hoach tong the. Trong qua trinh thuc hien, co the dieu chinh linh hoat theo ket qua thuc nghiem va phan hoi tu GVHD.

---

*Ngay tao: 15/10/2026*
*Du an: LMS tien tien va Learning Analytics*
*Dataset: OULAD (Open University Learning Analytics Dataset)*
