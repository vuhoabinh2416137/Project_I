# BÁO CÁO KỸ THUẬT FEATURE ENGINEERING & LÁT CẮT THỜI GIAN CẢNH BÁO SỚM (TEMPORAL SNAPSHOTS)
## Tuần 5: Trích xuất Đặc trưng Cơ bản, Đa dạng Học liệu (SEDL Proxies) & Đối sánh Snapshot 25%/50%/75%/100%
## Đề tài: LMS Tiên tiến và Learning Analytics (Học máy Dự đoán Nguy cơ Học tập Sinh viên OULAD)

---

## 1. TỔNG QUAN TIẾN ĐỘ TUẦN 5

Toàn bộ các mục tiêu trọng tâm của Tuần 5 theo Kế hoạch Dự án LMS Learning Analytics đã được hoàn thành 100% thông qua lớp `OULADFeatureEngineer`, bộ kiểm thử tự động (**12/12 unit tests Tuần 5 PASSED**) tích hợp hoàn toàn trong Jupyter Notebook [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb), xuất bản **5 biểu đồ phân tích chuyên sâu** (`21_*.png` đến `25_*.png` tại `reports/figures/`), cùng bảng danh mục đặc trưng chuẩn hóa gồm **76 đặc trưng** tại [`data/processed/week5_feature_engineering/feature_catalog_week5.csv`](file:///c:/Users/mrore/Project_I/data/processed/week5_feature_engineering/feature_catalog_week5.csv).

| Mã CV | Nội dung công việc Tuần 5 | Trạng thái | Sản phẩm đầu ra |
|:---:|---|:---:|---|
| **T5.1** | **Click-based Features (Behavioural)**: `total_clicks`, `active_days`, `clicks_per_day`, `daily_click_mean`, `click_std`, `click_cv`, `active_days_ratio`, `pre_course_clicks`, `active_span_days` | Hoàn thành | Bảng đặc trưng & [21_feature_engineering_taxonomy_and_distributions.png](file:///c:/Users/mrore/Project_I/reports/figures/21_feature_engineering_taxonomy_and_distributions.png) |
| **T5.2** | **Activity-based Features (SEDL Proxies)**: 20 loại `activity_type` (gồm `clicks_glossary`), độ đa dạng học liệu `activity_diversity` và độ hỗn loạn Shannon `activity_entropy` | Hoàn thành | [22_activity_diversity_and_entropy_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/22_activity_diversity_and_entropy_analysis.png) |
| **T5.3** | **Assessment-based Features**: `weighted_avg_score`, `earned_weight_score`, `first_score`, `tma1_score`, `submission_rate` (Loại trừ tuyệt đối bài thi `Exam` để triệt tiêu Target Leakage) | Hoàn thành | [23_assessment_trajectory_and_weighted_scores.png](file:///c:/Users/mrore/Project_I/reports/figures/23_assessment_trajectory_and_weighted_scores.png) |
| **T5.4** | **Demographics & Context**: Tải học tập `studied_credits`, tiền sử học vụ `num_of_prev_attempts`, độ tuổi, học vấn, kinh tế vùng `imd_band`, cờ `credit_overload` | Hoàn thành | Danh mục `feature_catalog_week5.csv` |
| **T5.5** | **Temporal Snapshot Modeling (Zero Leakage)**: Trích xuất 4 lát cắt thời gian $25\%$, $50\%$, $75\%$, $100\%$ khóa học (`date <= cutoff_day` và `date_submitted <= cutoff_day`) | Hoàn thành | 4 file CSV snapshot ($32,593 \times 113$ cột Tuần 5–6) tại `data/processed/week5_6_snapshots/` |
| **T5.6** | **Cohort Management**: Quản lý và đối sánh Full Cohort ($N=32,593$), Pre-Course Active Cohort ($N=29,915$) và Dynamic Active Cohort tại từng mốc thời gian cắt | Hoàn thành | Bảng thống kê động học sinh viên rút môn |
| **T5.7** | **Feature Importance Ranking**: Đo lường Point-Biserial $r$, Spearman $\rho$, Cohen's $d$, Mutual Information và Random Forest Importance (Gini & Permutation) | Hoàn thành | [25_preliminary_feature_importance_ranking.png](file:///c:/Users/mrore/Project_I/reports/figures/25_preliminary_feature_importance_ranking.png) & `feature_importance_week5.json` |
| **T5.8** | **Benchmark Thực nghiệm Học máy**: Huấn luyện đối kháng 92 cấu hình mô hình (Random Forest, Logistic Regression, Decision Tree) qua các mốc snapshot và chiến lược chia tập | Hoàn thành | [24_snapshot_progression_25_50_75_100.png](file:///c:/Users/mrore/Project_I/reports/figures/24_snapshot_progression_25_50_75_100.png) & `week5_experiment_metrics.json` |

---

## 2. CHI TIẾT KỸ THUẬT 4 NHÓM ĐẶC TRƯNG TUẦN 5 (T5.1 – T5.4)

Dựa trên khung lý thuyết Gắn kết Học tập Số (Student Engagement in Digital Learning – SEDL; Bergdahl et al., 2024; Bond & Bedenlier, 2019), nghiên cứu đã mở rộng và chuẩn hóa toàn diện không gian đặc trưng từ **$p = 63$ đặc trưng cơ sở ban đầu** ở Tuần 4 (tương ứng **$64$ cột mã hóa baseline** khi bổ sung cờ chỉ báo nộp bài `submitted_tma1`) lên **$p = 76$ đặc trưng danh mục số học** (tương ứng **$98$ biến sau mã hóa One-Hot đa biến**) phân loại theo 4 nhóm kỹ thuật.

> **Ghi chú phương pháp luận về chiều Cảm xúc (Emotional / Affective Engagement) trong khung SEDL**:  
> Khung lý thuyết SEDL toàn diện bao gồm 4 chiều tương tác: Hành vi (Behavioural), Nhận thức (Cognitive), Cảm xúc (Emotional) và Xã hội (Social). Trong đề tài này, dữ liệu vết tương tác clickstream LMS khách quan của OULAD phản ánh trực tiếp 3 chiều: Hành vi (tần suất, cường độ, tính bền bỉ click và nộp bài), Nhận thức (tiến trình điểm số, học liệu lý thuyết/bài tập) và Xã hội (diễn đàn `forumng`, wiki `ouwiki`), bên cạnh Bối cảnh nhân khẩu học (Demographics & Context). Trạng thái cảm xúc tức thời của người học (như hứng thú, chán nản, âu lo môn học) không thể đo lường trực tiếp từ nhật ký máy chủ LMS nếu thiếu khảo sát tâm lý tự khai báo (self-reports) hoặc thiết bị đo sinh trắc thời gian thực — đây là một giới hạn tự nhiên đã được kiểm chứng và chấp nhận rộng rãi trong y văn Learning Analytics quốc tế (Bergdahl et al., 2024).

### 2.1. Nhóm 5.1: Đặc trưng Tương tác Hành vi (Click-based Behavioral Engagement)
Nhóm đặc trưng đo lường mức độ chuyên cần, thời lượng và cường độ học tập của sinh viên:
- `total_clicks`: Tổng số lượt nhấp chuột trên hệ sinh thái LMS tính đến thời điểm $T_{\text{cutoff}}$.
- `active_days`: Số ngày duy nhất có phát sinh tương tác ($\text{count}(\text{distinct date})$) với $\text{date} \le T_{\text{cutoff}}$.
- `clicks_per_day`: Cường độ tương tác trung bình trên toàn bộ thời lượng lịch khóa học đã trôi qua:
  $$\text{clicks\_per\_day} = \frac{\text{total\_clicks}}{\max(1.0, T_{\text{cutoff}})}$$
- `daily_click_mean` (`clicks_per_active_day`): Cường độ tương tác trung bình trong các ngày có học tập:
  $$\text{daily\_click\_mean} = \begin{cases} \frac{\text{total\_clicks}}{\text{active\_days}}, & \text{active\_days} > 0 \\ 0.0, & \text{active\_days} = 0 \end{cases}$$
- `click_std` (`daily_click_std`): Độ lệch chuẩn số click mỗi ngày trên các ngày có hoạt động ($\text{ddof}=1$), bằng $0.0$ nếu $\text{active\_days} \le 1$.
- `click_cv`: Hệ số biến thiên cường độ học tập ($\text{click\_std} / \text{daily\_click\_mean}$ nếu $\text{daily\_click\_mean} > 0$), đo tính ổn định hay đột biến bất thường.
- `active_days_ratio`: Tỷ lệ ngày có học tập trong kỳ học: $\text{active\_days\_in\_course} / \max(1.0, T_{\text{cutoff}})$, với $\text{active\_days\_in\_course}$ chỉ tính các ngày $0 \le \text{date} \le T_{\text{cutoff}}$.
- `pre_course_clicks` & `has_pre_course_clicks`: Lượt click và cờ tương tác trước ngày khai giảng ($\text{date} < 0$).
- `first_click_day`, `last_click_day`, `active_span_days`: Khoảng cách thời gian từ ngày học đầu tiên đến ngày học cuối cùng tính đến mốc $T_{\text{cutoff}}$ có xử lý điều kiện biên triệt để:
  $$\text{active\_span\_days} = \begin{cases} \text{last\_click\_day} - \text{first\_click\_day} + 1, & \text{active\_days} > 0 \\ 0.0, & \text{active\_days} = 0 \end{cases}$$

Biểu đồ minh họa: [reports/figures/21_feature_engineering_taxonomy_and_distributions.png](file:///c:/Users/mrore/Project_I/reports/figures/21_feature_engineering_taxonomy_and_distributions.png)

### 2.2. Nhóm 5.2: Đặc trưng Theo Loại Tài nguyên & Đại diện Khung SEDL (Activity-based & SEDL Proxies)
- **Bao phủ toàn diện 20 loại tài nguyên VLE**:
  Thay vì chỉ lấy 8 nhóm rút gọn như Tuần 3, pipeline Tuần 5 trích xuất đầy đủ 20 loại tài nguyên: `oucontent`, `forumng`, `quiz`, `resource`, `homepage`, `subpage`, `url`, `glossary`, `ouwiki`, `page`, `questionnaire`, `dataplus`, `externalquiz`, `oucollaborate`, `ouelluminate`, `dualpane`, `repeatactivity`, `htmlactivity`, `sharedsubpage`, `folder`.
- **Độ đa dạng học liệu (`activity_diversity`)**:
  Đếm số loại tài nguyên khác nhau mà sinh viên đã truy cập trên tổng số 20 loại ($0 \le \text{activity\_diversity} \le 20$). Sinh viên Not At-Risk có độ đa dạng đạt $8.03$ (mốc 25%) và $9.16$ (mốc 100%), cao vượt trội so với nhóm At-Risk ($5.54$ ở mốc 25% và $5.78$ ở mốc 100%; Cohen's $d = 1.05$).
- **Độ hỗn loạn Shannon Entropy (`activity_entropy`)**:
  Đo lường mức độ phân bổ đều đặn hay thiên lệch vào một loại học liệu duy nhất:
  $$H = \begin{cases} -\sum_{k=1}^{K} p_k \ln(p_k), & \text{total\_clicks} > 0 \quad (\text{với } p_k = \frac{\text{clicks}_k}{\text{total\_clicks}} > 0) \\ 0.0, & \text{total\_clicks} = 0 \end{cases}$$
  với $0 \le H \le \ln(20) \approx 2.996$. Nhóm Not At-Risk có entropy trung bình đạt $1.46 - 1.49$, phản ánh hành vi học tập cân bằng giữa đọc bài giảng (`oucontent`), làm quiz (`quiz`), thảo luận (`forumng`) và tra cứu (`resource`/`glossary`), trong khi nhóm At-Risk có entropy thấp ($1.10 - 1.11$) do chủ yếu chỉ truy cập trang chủ (`homepage`) rồi thoát.

Biểu đồ minh họa: [reports/figures/22_activity_diversity_and_entropy_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/22_activity_diversity_and_entropy_analysis.png)

### 2.3. Nhóm 5.3: Đặc trưng Bài kiểm tra & Điểm số (Assessment-based & Trajectory)
- **Quy tắc Vàng Loại bỏ Bài thi Cuối kỳ (`Exam Strictly Excluded`)**:
  Trong OULAD có 24 bài đánh giá dạng `Exam` diễn ra vào cuối khóa (ngày 240-261). Việc tính gộp `Exam` vào đặc trưng đầu vào sẽ gây ra hiện tượng rò rỉ nhãn (Target Leakage) và làm sai lệch chỉ số nộp bài. Pipeline Tuần 5 **chỉ tính toán trên các bài Coursework (TMA và CMA, tổng cộng 182 bài toàn bộ dataset)**.
- **Điểm trung bình có trọng số (`weighted_avg_score` - Cognitive Performance)**:
  $$\text{weighted\_avg\_score} = \begin{cases} \frac{\sum_{i \in \text{Graded}} \text{score}_i \times \text{weight}_i}{\sum_{i \in \text{Graded}} \text{weight}_i}, & \sum_{i \in \text{Graded}} \text{weight}_i > 0 \\ 0.0, & \sum_{i \in \text{Graded}} \text{weight}_i = 0 \end{cases}$$
  phản ánh năng lực học vụ chính xác hơn điểm trung bình cộng số học.
- **Điểm tích lũy thực tế (`earned_weight_score` - Cognitive Achievement)**:
  $$\text{earned\_weight\_score} = \frac{\sum_{i \in \text{Graded}} \text{score}_i \times \text{weight}_i}{100.0}$$
  Biến này đo lường tỷ lệ điểm khóa học mà sinh viên đã cầm chắc trong tay. Tại mốc 25%, sinh viên Not At-Risk đã tích lũy $15.76\%$ so với $6.64\%$ của nhóm At-Risk; tại mốc 100%, tỷ lệ tích lũy là $65.57\%$ so với $14.33\%$ (Cohen's $d = 2.20$).
- **Điểm bài kiểm tra đầu tiên thực tế (`first_score` - Cognitive Early Signal)**:
  Điểm của bài kiểm tra đầu tiên mà sinh viên thực tế đã nộp (bất kể là TMA hay CMA).
- **Tỷ lệ nộp bài theo hạn nộp trước mắt (`submission_rate` - Behavioural Task Completion)**:
  Được chuẩn hóa chuẩn xác theo số bài tập **thực tế đã đến hạn tính đến thời điểm $T_{\text{cutoff}}$** (`due_course_assessments`):
  $$\text{submission\_rate} = \min\left(1.0, \frac{\text{num\_graded\_submissions}}{\max(1, \text{due\_course\_assessments})}\right)$$
  đảm bảo $\text{submission\_rate} \in [0, 1]$ tuyệt đối, không còn hiện tượng vượt ngưỡng 1.0.
- **Tỷ lệ nộp bài theo đề cương toàn khóa (`course_progress_submission_rate` - Behavioural Task Completion)**:
  Được chuẩn hóa theo tổng số bài kiểm tra Coursework của toàn bộ học phần (`total_course_assessments`):
  $$\text{course\_progress\_submission\_rate} = \min\left(1.0, \frac{\text{num\_graded\_submissions}}{\max(1, \text{total\_course\_assessments})}\right)$$

> **Cơ chế đồng quy số học tại mốc Toàn khóa 100%**:  
> Ở mốc cuối khóa (100%), toàn bộ các bài kiểm tra Coursework trong đề cương môn học đều đã đến hạn ($\text{due\_course\_assessments} \equiv \text{total\_course\_assessments}$), do đó $\text{submission\_rate} \equiv \text{course\_progress\_submission\_rate}$ đối với mọi sinh viên. Điều này lý giải tại sao tại mốc 100%, hai đặc trưng này có chỉ số tương quan Point-Biserial ($r = -0.812$) và khoảng cách Cohen's $d$ ($2.784$) hoàn toàn trùng khớp. Tuy nhiên, tại các mốc sớm 25% và 50%, `submission_rate` đo lường mức độ tuân thủ hạn nộp trước mắt ($r = -0.531$, Cohen's $d = 1.255$), trong khi `course_progress_submission_rate` đo lường tiến độ hoàn thành đường dài so với toàn bộ chương trình môn học ($r = -0.480$, Cohen's $d = 1.097$). Cả hai bổ trợ thông tin quan trọng cho nhau trong mô hình dự báo sớm.

Biểu đồ minh họa: [reports/figures/23_assessment_trajectory_and_weighted_scores.png](file:///c:/Users/mrore/Project_I/reports/figures/23_assessment_trajectory_and_weighted_scores.png)

### 2.4. Nhóm 5.4: Bối cảnh Học tập (Context)
- `studied_credits`: Tải học tập (số tín chỉ sinh viên đăng ký học đồng thời).
- `credit_overload`: Biến cờ nhị phân ($= 1$ nếu $\text{studied\_credits} > 60$). Ngưỡng $60$ tín chỉ được chọn vì đây là mức tải chuẩn hệ bán thời gian (*part-time*) tại The Open University, đồng thời là mode ($51.4\%$, $n=16{,}751$) và trung vị ($Q_1 = \text{median} = 60$) của OULAD ($N=32{,}593$, phân tách $63.1\%$ học mức chuẩn $\le 60$ với $36.9\%$ vượt tải $> 60$); ngay khi vượt mốc $60$ tín chỉ, tỷ lệ `At-Risk` tăng nhảy vọt từ $48.2\%$ lên $60.7\%$ ($+12.5$ điểm \%), trong đó tỷ lệ rút môn (`Withdrawn`) tăng từ $25.4\%$ lên $41.1\%$.
- `num_of_prev_attempts` & `is_repeat_student`: Tiền sử học vụ và cờ học lại ($= 1$ nếu $\text{num\_of\_prev\_attempts} > 0$).

---

## 3. RÀNG BUỘC ZERO TEMPORAL LEAKAGE & QUẢN LÝ COHORT ĐỘNG (T5.5 & T5.6)

### 3.1. Ràng buộc Lát cắt Thời gian Tuyệt đối (Zero Temporal Leakage)
Tại mỗi tỷ lệ thời lượng $c \in \{0.25, 0.50, 0.75, 1.00\}$, ngày cắt của từng khóa học được xác định:
$$T_{\text{cutoff}} = \text{module\_presentation\_length} \times c$$
- **Tương tác VLE**: Tuyệt đối chỉ lấy các bản ghi có $\text{date} \le T_{\text{cutoff}}$. Bộ kiểm thử `test_02_zero_temporal_leakage_vle` xác nhận $100\%$ bản ghi có $\text{last\_click\_day} \le T_{\text{cutoff}}$ ($0$ vi phạm).
- **Bài nộp Assessment**: Chỉ tính các bài nộp có $\text{date\_submitted} \le T_{\text{cutoff}}$. Các bài tập có hạn nộp hoặc ngày nộp sau mốc $T_{\text{cutoff}}$ hoàn toàn vô hình đối với mô hình tại thời điểm dự đoán.

### 3.2. Quản lý 3 Định nghĩa Cohort Động qua các Mốc Snapshot
Nghiên cứu thiết lập và đối sánh 3 định nghĩa tập sinh viên:
1. **Full Cohort ($N = 32,593$, cố định)**: Toàn bộ sinh viên, phục vụ góc nhìn quản trị tuyển sinh và hoạch định nguồn lực chung.
2. **Pre-Course Active Cohort ($N = 29,915$, cố định)**: Loại bỏ $2,678$ sinh viên đã rút trước ngày khai giảng ($82.6\%$ có 0 click VLE).
3. **Dynamic Snapshot Active Cohort (Thu hẹp tự nhiên theo thời gian)**:
   Tại thời điểm dự đoán $T_{\text{cutoff}}$, nhà trường chỉ cần cảnh báo những sinh viên **còn đang theo học tại thời điểm đó**. Những sinh viên đã chính thức nộp đơn xin rút môn trước mốc $T_{\text{cutoff}}$ (`date_unregistration <= cutoff_day`) đã được hệ thống đào tạo ghi nhận và không cần mô hình AI dự báo nữa.

| Chỉ số Cohort & Thời điểm | Mốc 25% ($T \approx 60-67$ ngày) | Mốc 50% ($T \approx 120-134$ ngày) | Mốc 75% ($T \approx 180-201$ ngày) | Mốc 100% (Cuối khóa) |
|---|:---:|:---:|:---:|:---:|
| **Quy mô Dynamic Active Cohort ($N$)** | **$26,248$** | **$24,601$** | **$23,159$** | **$22,522$** |
| Số sinh viên đã chính thức rút trước mốc cắt | $6,345$ ($19.47\%$) | $7,992$ ($24.52\%$) | $9,434$ ($28.94\%$) | $10,071$ ($30.90\%$) |
| Tỷ lệ At-Risk thực tế trong Active Cohort | **$41.39\%$** | **$37.46\%$** | **$33.57\%$** | **$31.69\%$** |
| Số ca Fail tương lai cần phát hiện (*) | $7,044$ | $7,044$ | $7,043$ | $7,043$ |
| Số ca Withdrawn tiếp tục rút sau mốc cắt | $3,819$ | $2,172$ | $731$ | $94$ |
| Mean Total Clicks (Not At-Risk vs At-Risk) | $755.6$ vs $290.5$ | $1,175.5$ vs $366.4$ | $1,670.1$ vs $419.1$ | $2,068.2$ vs $452.4$ |
| Mean Earned Weight (Not At-Risk vs At-Risk) | $15.76\%$ vs $6.64\%$ | $36.89\%$ vs $11.04\%$ | $57.90\%$ vs $13.57\%$ | $65.57\%$ vs $14.33\%$ |

> (*) **Giải trình thực chứng hiện tượng dữ liệu**:  
> Số ca Fail trong Dynamic Active Cohort giảm đúng 1 ca từ $7,044$ (mốc 50%) xuống $7,043$ (mốc 75%). Phân tích đối soát dữ liệu gốc OULAD (`studentInfo.csv` và `studentRegistration.csv`) xác nhận: trong số 32,593 bản ghi có 9 sinh viên đã nộp đơn hủy môn (`date_unregistration` khác null) nhưng kết quả cuối kỳ vẫn được ghi nhận là `Fail` (thay vì `Withdrawn`). Trong đó, đúng 1 sinh viên (Mã SV `554243`, học phần `BBB_2013J`) rút môn vào ngày thứ $166$ — thời điểm nằm giữa mốc 50% ($T_{\text{cutoff}} = 134$ ngày) và 75% ($T_{\text{cutoff}} = 201$ ngày). Tại mốc 75%, sinh viên này đã chính thức rút môn nên theo đúng định nghĩa Dynamic Active Cohort (`date_unregistration > cutoff_day`), sinh viên này được lọc ra khỏi nhóm còn đang học, khiến số ca Fail giảm chính xác 1 trường hợp.

Biểu đồ minh họa: [reports/figures/24_snapshot_progression_25_50_75_100.png](file:///c:/Users/mrore/Project_I/reports/figures/24_snapshot_progression_25_50_75_100.png)

> ### Ý nghĩa và Nhận xét Biểu đồ Tiến triển Hiệu năng (Test ROC-AUC)
> Chỉ số **ROC-AUC** đo xác suất mô hình xếp hạng đúng sinh viên *At-Risk* có rủi ro cao hơn sinh viên *Not At-Risk* ($0.5$: ngẫu nhiên; $> 0.8$: tốt; $> 0.9$: xuất sắc). Biểu đồ trên cho thấy:
> - **Biểu đồ (A) -- Hiệu quả của Bộ đặc trưng Tuần 5**: Năng lực dự báo tăng mạnh theo lượng dữ liệu tích lũy từ mốc $25\%$ đến $100\%$ (tăng mạnh nhất ở giai đoạn $25\% \rightarrow 50\%$). Trên cả 4 mốc và cả 2 thuật toán, **Bộ mở rộng Tuần 5** luôn vượt trên **Bộ cơ sở Tuần 3--4**, khẳng định giá trị thực chất của các đặc trưng SEDL mới.
> - **Biểu đồ (B) -- Độ khó thực tế giữa 3 Cohort**: `Full Cohort` có AUC cao nhất do bao gồm nhiều sinh viên đã rút môn sớm ($0$ click, $0$ điểm -- rất dễ đoán). Khi lọc bỏ người đã rút môn để chỉ xét nhóm đang học thực tế cần giảng viên can thiệp (`Dynamic Snapshot Active`), bài toán khó hơn nhưng mô hình vẫn đạt hiệu năng cao ($\text{AUC} = 0.8411$ tại $25\%$ và tăng lên $0.9561$ tại $100\%$).

---

## 4. PHÂN TÍCH TẦM QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCE RANKING - T5.7)

Để đánh giá khách quan sức mạnh dự báo của từng đặc trưng mà không bị thiên lệch bởi giả định tuyến tính đơn thuần, nghiên cứu kết hợp hai thước đo trọng tâm bên cạnh tương quan Point-Biserial $r$ và cỡ hiệu ứng Cohen's $d$:
- **Thông tin tương hỗ (Mutual Information -- `Mutual Info`)**: Là thước đo lý thuyết thông tin định lượng mức độ giảm độ bất định (entropy) của nhãn mục tiêu $Y$ (`at_risk`) khi biết giá trị đặc trưng $X$:
  $$I(X; Y) = \iint p(x, y) \ln\left(\frac{p(x, y)}{p(x)p(y)}\right) dx\,dy \ge 0$$
  `Mutual Info` bằng $0$ khi $X$ và $Y$ hoàn toàn độc lập thống kê; giá trị càng lớn chứng tỏ đặc trưng càng chứa nhiều thông tin phân tách nhãn. Khác với hệ số tương quan tuyến tính $r$, `Mutual Info` nắm bắt được **mọi mối quan hệ phụ thuộc phi tuyến và đơn biến** mà không phụ thuộc vào mô hình học máy cụ thể.
- **Độ quan trọng Gini của Rừng ngẫu nhiên (Random Forest Gini Importance -- `RF Gini` / Mean Decrease in Impurity)**: Đo lường tổng mức giảm độ tạp chất Gini ($\Delta \text{Gini}$) có trọng số theo tỷ lệ mẫu tại mọi nút phân nhánh sử dụng đặc trưng $X$, tính trung bình trên toàn bộ các cây quyết định trong rừng và chuẩn hóa để tổng trọng số của tất cả đặc trưng bằng $1.0$. Chỉ số `RF Gini` phản ánh **quyền lực phân nhánh thực tế trong không gian đa biến**, giúp nhận diện các đặc trưng đóng vai trò trụ cột khi kết hợp tương tác chéo với các biến khác.

Biểu đồ minh họa: [reports/figures/25_preliminary_feature_importance_ranking.png](file:///c:/Users/mrore/Project_I/reports/figures/25_preliminary_feature_importance_ranking.png)

Kết hợp đa chỉ số thống kê (Tương quan Point-Biserial $r$, Khoảng cách Cohen's $d$, Thông tin tương hỗ Mutual Information và Tầm quan trọng Random Forest Gini/Permutation):

### 4.1. Top 10 Đặc trưng Hàng đầu tại Mốc 25% (Cảnh báo sớm Thời gian thực)

| Hạng | Tên Đặc trưng | Nhóm / SEDL | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini |
|---:|---|---|---:|---:|---:|---:|
| 1 | `weighted_avg_score` | 5.3 / Cognitive (Performance) | $-0.571$ | $1.392$ | $0.209$ | $0.0787$ |
| 2 | `max_score` | 5.3 / Cognitive (Performance) | $-0.566$ | $1.374$ | $0.194$ | $0.0718$ |
| 3 | `avg_score` | 5.3 / Cognitive (Performance) | $-0.565$ | $1.372$ | $0.201$ | $0.0635$ |
| 4 | `course_progress_submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.480$ | $1.097$ | $0.205$ | $0.0553$ |
| 5 | `min_score` | 5.3 / Cognitive (Performance) | $-0.536$ | $1.272$ | $0.199$ | $0.0461$ |
| 6 | `earned_weight_score` | 5.3 / Cognitive (Achievement) | $-0.460$ | $1.038$ | $0.162$ | $0.0437$ |
| 7 | `tma1_score` | 5.3 / Cognitive (Early Signal) | $-0.514$ | $1.201$ | $0.179$ | $0.0423$ |
| 8 | `submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.531$ | $1.255$ | $0.188$ | $0.0402$ |
| 9 | `active_span_days` | 5.1 / Behavioural | $-0.512$ | $1.195$ | $0.161$ | $0.0392$ |
| 10 | `first_score` | 5.3 / Cognitive (Early Signal) | $-0.523$ | $1.229$ | $0.182$ | $0.0326$ |

### 4.2. Top 10 Đặc trưng Hàng đầu tại Mốc 50% (Giữa khóa)

| Hạng | Tên Đặc trưng | Nhóm / SEDL | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini |
|---:|---|---|---:|---:|---:|---:|
| 1 | `active_span_days` | 5.1 / Behavioural | $-0.640$ | $1.666$ | $0.269$ | $0.0944$ |
| 2 | `earned_weight_score` | 5.3 / Cognitive (Achievement) | $-0.624$ | $1.598$ | $0.249$ | $0.0930$ |
| 3 | `submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.671$ | $1.811$ | $0.298$ | $0.0796$ |
| 4 | `course_progress_submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.639$ | $1.663$ | $0.305$ | $0.0712$ |
| 5 | `num_submissions_raw` | 5.3 / Behavioural (Task Completion) | $-0.581$ | $1.428$ | $0.253$ | $0.0569$ |
| 6 | `num_submissions` | 5.3 / Behavioural (Task Completion) | $-0.581$ | $1.431$ | $0.254$ | $0.0473$ |
| 7 | `weighted_avg_score` | 5.3 / Cognitive (Performance) | $-0.601$ | $1.506$ | $0.267$ | $0.0468$ |
| 8 | `score_std` | 5.3 / Cognitive (Consistency) | $-0.252$ | $0.522$ | $0.251$ | $0.0428$ |
| 9 | `avg_score` | 5.3 / Cognitive (Performance) | $-0.595$ | $1.485$ | $0.256$ | $0.0356$ |
| 10 | `max_score` | 5.3 / Cognitive (Performance) | $-0.599$ | $1.497$ | $0.240$ | $0.0337$ |

### 4.3. Top 10 Đặc trưng Hàng đầu tại Mốc 75% (Tiền cuối khóa)

| Hạng | Tên Đặc trưng | Nhóm / SEDL | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini |
|---:|---|---|---:|---:|---:|---:|
| 1 | `active_span_days` | 5.1 / Behavioural | $-0.725$ | $2.111$ | $0.371$ | $0.1150$ |
| 2 | `earned_weight_score` | 5.3 / Cognitive (Achievement) | $-0.705$ | $1.989$ | $0.330$ | $0.1116$ |
| 3 | `course_progress_submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.748$ | $2.255$ | $0.388$ | $0.1006$ |
| 4 | `submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.749$ | $2.264$ | $0.387$ | $0.0876$ |
| 5 | `num_submissions_raw` | 5.3 / Behavioural (Task Completion) | $-0.666$ | $1.789$ | $0.333$ | $0.0677$ |
| 6 | `num_submissions` | 5.3 / Behavioural (Task Completion) | $-0.667$ | $1.792$ | $0.335$ | $0.0551$ |
| 7 | `active_days_ratio` | 5.1 / Behavioural | $-0.604$ | $1.516$ | $0.284$ | $0.0397$ |
| 8 | `weighted_avg_score` | 5.3 / Cognitive (Performance) | $-0.605$ | $1.522$ | $0.295$ | $0.0356$ |
| 9 | `active_days_in_course` | 5.1 / Behavioural | $-0.606$ | $1.527$ | $0.270$ | $0.0301$ |
| 10 | `score_std` | 5.3 / Cognitive (Consistency) | $-0.273$ | $0.569$ | $0.305$ | $0.0280$ |

### 4.4. Top 10 Đặc trưng Hàng đầu tại Mốc 100% (Toàn khóa)

| Hạng | Tên Đặc trưng | Nhóm / SEDL | Point-Bis. $r$ | Cohen's $d$ | Mutual Info | RF Gini |
|---:|---|---|---:|---:|---:|---:|
| 1 | `active_span_days` | 5.1 / Behavioural | $-0.777$ | $2.470$ | $0.424$ | $0.1334$ |
| 2 | `course_progress_submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.812$ | $2.784$ | $0.448$ | $0.1314$ |
| 3 | `earned_weight_score` | 5.3 / Cognitive (Achievement) | $-0.739$ | $2.197$ | $0.373$ | $0.1122$ |
| 4 | `submission_rate` | 5.3 / Behavioural (Task Completion) | $-0.812$ | $2.784$ | $0.446$ | $0.0801$ |
| 5 | `num_submissions_raw` | 5.3 / Behavioural (Task Completion) | $-0.700$ | $1.961$ | $0.365$ | $0.0664$ |
| 6 | `num_submissions` | 5.3 / Behavioural (Task Completion) | $-0.700$ | $1.965$ | $0.365$ | $0.0582$ |
| 7 | `active_days_ratio` | 5.1 / Behavioural | $-0.633$ | $1.636$ | $0.313$ | $0.0422$ |
| 8 | `active_days_in_course` | 5.1 / Behavioural | $-0.635$ | $1.648$ | $0.303$ | $0.0333$ |
| 9 | `max_score` | 5.3 / Cognitive (Performance) | $-0.613$ | $1.555$ | $0.264$ | $0.0278$ |
| 10 | `weighted_avg_score` | 5.3 / Cognitive (Performance) | $-0.605$ | $1.522$ | $0.302$ | $0.0262$ |

### 4.5. Phân bổ Tầm quan trọng theo 4 Nhóm Đặc trưng qua 4 Lát cắt
| Mốc Snapshot | 5.1 Click-based | 5.2 Activity-based | 5.3 Assessment-based | 5.4 Demographics & Context |
|:---:|---:|---:|---:|---:|
| **25%** | $16.2\%$ | $15.5\%$ | **$60.5\%$** | $5.2\%$ |
| **50%** | $20.6\%$ | $11.6\%$ | **$63.4\%$** | $2.8\%$ |
| **75%** | $25.2\%$ | $9.5\%$ | **$62.5\%$** | $1.9\%$ |
| **100%** | $27.9\%$ | $8.9\%$ | **$61.0\%$** | $1.5\%$ |

*(Lưu ý phương pháp luận: Dữ liệu log máy chủ LMS không đo lường trực tiếp trạng thái Cảm xúc Emotional/Affective do thiếu vắng khảo sát tự khai báo thời gian thực)*.

---

## 5. ĐỐI SÁNH THỰC NGHIỆM ĐỊNH LƯỢNG MÔ HÌNH HỌC MÁY (T5.8)

Nghiên cứu đã thực thi **92 cấu hình thực nghiệm độc lập** trên tập Test, so sánh trực tiếp giữa:
- **Bộ đặc trưng cơ sở Tuần 3-4** ($64$ biến mã hóa) vs **Bộ đặc trưng mở rộng Tuần 5** ($76$ đặc trưng danh mục, $98$ biến mã hóa sau One-Hot bao gồm `activity_diversity`, `activity_entropy`, `earned_weight_score`, `weighted_avg_score`, `click_cv`, `active_span_days`...).
- **Chiến lược chia tập**: `Stratified Random Split` vs `Group-Stratified Split` theo `id_student` (Triệt tiêu 100% rò rỉ $3,538$ sinh viên đa môn).

### 5.1. Kết quả trên Full Cohort ($N = 32,593$) - Group-Stratified Split:

| Mốc Snapshot | Thuật toán | Bộ Đặc trưng | Test Accuracy | Bal-Acc | Test Precision | Test Recall (At-Risk) | Test F1-Score | Test ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **25%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.8018$ | $0.8043$ | $0.8493$ | $0.7592$ | $0.8017$ | $0.8875$ |
| **25%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.8151$** | **$0.8175$** | **$0.8621$** | **$0.7734$** | **$0.8154$** | **$0.8938$** |
| **25%** | Random Forest | Tuần 3-4 Cơ sở | $0.8106$ | $0.8140$ | $0.8717$ | $0.7518$ | $0.8073$ | $0.8887$ |
| **25%** | Random Forest | **Tuần 5 Mở rộng** | **$0.8146$** | **$0.8182$** | **$0.8771$** | **$0.7547$** | **$0.8113$** | **$0.8914$** |
| **50%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.8690$ | $0.8709$ | $0.9085$ | $0.8360$ | $0.8708$ | $0.9394$ |
| **50%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.8671$** | **$0.8688$** | **$0.9035$** | **$0.8377$** | **$0.8693$** | **$0.9432$** |
| **50%** | Random Forest | Tuần 3-4 Cơ sở | $0.8681$ | $0.8706$ | $0.9153$ | $0.8267$ | $0.8687$ | $0.9368$ |
| **50%** | Random Forest | **Tuần 5 Mở rộng** | **$0.8776$** | **$0.8799$** | **$0.9233$** | **$0.8377$** | **$0.8784$** | **$0.9436$** |
| **75%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.9038$ | $0.9054$ | $0.9370$ | $0.8767$ | $0.9058$ | $0.9658$ |
| **75%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.9149$** | **$0.9163$** | **$0.9440$** | **$0.8918$** | **$0.9172$** | **$0.9701$** |
| **75%** | Random Forest | Tuần 3-4 Cơ sở | $0.9053$ | $0.9073$ | $0.9453$ | $0.8710$ | $0.9066$ | $0.9672$ |
| **75%** | Random Forest | **Tuần 5 Mở rộng** | **$0.9184$** | **$0.9204$** | **$0.9581$** | **$0.8841$** | **$0.9196$** | **$0.9718$** |
| **100%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.9227$ | $0.9242$ | $0.9533$ | $0.8975$ | $0.9246$ | $0.9768$ |
| **100%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.9336$** | **$0.9349$** | **$0.9602$** | **$0.9121$** | **$0.9355$** | **$0.9809$** |
| **100%** | Random Forest | Tuần 3-4 Cơ sở | $0.9246$ | $0.9266$ | $0.9631$ | $0.8914$ | $0.9258$ | $0.9780$ |
| **100%** | Random Forest | **Tuần 5 Mở rộng** | **$0.9366$** | **$0.9386$** | **$0.9754$** | **$0.9028$** | **$0.9377$** | **$0.9823$** |

### 5.2. Kết quả trên Dynamic Snapshot Active Cohort - Group-Stratified Split:

Khi loại bỏ các sinh viên đã chính thức hủy môn trước thời điểm cắt (chỉ đánh giá trên nhóm sinh viên còn đang thực tế theo học tại thời điểm quan sát):

| Mốc Snapshot | Thuật toán | Bộ Đặc trưng | Test Accuracy | Bal-Acc | Test Precision | Test Recall (At-Risk) | Test F1-Score | Test ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **25%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.7728$ | $0.7511$ | $0.7823$ | $0.6250$ | $0.6948$ | $0.8368$ |
| **25%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.7827$** | **$0.7620$** | **$0.7936$** | **$0.6418$** | **$0.7097$** | **$0.8449$** |
| **25%** | Random Forest | Tuần 3-4 Cơ sở | $0.7731$ | $0.7505$ | $0.7871$ | $0.6192$ | $0.6931$ | $0.8394$ |
| **25%** | Random Forest | **Tuần 5 Mở rộng** | **$0.7717$** | **$0.7485$** | **$0.7881$** | **$0.6134$** | **$0.6899$** | **$0.8411$** |
| **50%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.8270$ | $0.7992$ | $0.8214$ | $0.6879$ | $0.7488$ | $0.8904$ |
| **50%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.8353$** | **$0.8082$** | **$0.8336$** | **$0.7001$** | **$0.7610$** | **$0.9006$** |
| **50%** | Random Forest | Tuần 3-4 Cơ sở | $0.8245$ | $0.7939$ | $0.8271$ | $0.6720$ | $0.7415$ | $0.8877$ |
| **50%** | Random Forest | **Tuần 5 Mở rộng** | **$0.8356$** | **$0.8052$** | **$0.8476$** | **$0.6841$** | **$0.7571$** | **$0.8947$** |
| **75%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.8728$ | $0.8341$ | $0.8825$ | $0.7165$ | $0.7909$ | $0.9312$ |
| **75%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.8818$** | **$0.8485$** | **$0.8830$** | **$0.7471$** | **$0.8094$** | **$0.9399$** |
| **75%** | Random Forest | Tuần 3-4 Cơ sở | $0.8713$ | $0.8301$ | $0.8888$ | $0.7048$ | $0.7861$ | $0.9297$ |
| **75%** | Random Forest | **Tuần 5 Mở rộng** | **$0.8864$** | **$0.8481$** | **$0.9125$** | **$0.7318$** | **$0.8122$** | **$0.9372$** |
| **100%** | Logistic Regression | Tuần 3-4 Cơ sở | $0.8946$ | $0.8567$ | $0.8982$ | $0.7529$ | $0.8192$ | $0.9479$ |
| **100%** | Logistic Regression | **Tuần 5 Mở rộng** | **$0.9061$** | **$0.8696$** | **$0.9214$** | **$0.7696$** | **$0.8387$** | **$0.9547$** |
| **100%** | Random Forest | Tuần 3-4 Cơ sở | $0.8940$ | $0.8489$ | $0.9238$ | $0.7255$ | $0.8127$ | $0.9481$ |
| **100%** | Random Forest | **Tuần 5 Mở rộng** | **$0.9052$** | **$0.8636$** | **$0.9387$** | **$0.7500$** | **$0.8338$** | **$0.9561$** |

> ### Đánh giá Khách quan Tiêu chuẩn Nghiệm thu Câu hỏi Nghiên cứu RQ2
> Đối chiếu kết quả thực nghiệm giữa hai góc nhìn quản trị đào tạo và nghiệp vụ lớp học trực tiếp:
> 1. **Trên Full Cohort ($N = 32,593$)**: Với bài toán giám sát vĩ mô của nhà trường (tính cả người học bỏ cuộc sớm trước ngày 60), tại mốc Cảnh báo sớm 25% thời lượng, mô hình với bộ đặc trưng Tuần 5 đạt **Test Recall = 75.47%** (Random Forest) và **Test Recall = 77.34%** (Logistic Regression), cùng **Test ROC-AUC $\approx$ 0.891 - 0.894**. Trên phương diện giám sát toàn thể, kết quả này **đáp ứng vững chắc tiêu chuẩn thành công $\text{Recall} \ge 75\%$** của **RQ2**.
> 2. **Trên Dynamic Snapshot Active Cohort ($N = 26,248$ tại 25%)**: Với nghiệp vụ sư phạm trực tiếp (chỉ theo dõi sinh viên còn đang học tại thời điểm cảnh báo để can thiệp), bài toán có độ khó học thuật cao hơn rõ rệt. Nghiên cứu ghi nhận tiến trình Recall tăng dần theo lượng thông tin tích lũy: từ **$61.34\%$** (mốc 25%) $\to$ **$68.41\%$** (mốc 50%) $\to$ **$73.18\%$** (mốc 75%) $\to$ **$75.00\%$** (mốc 100%) với Random Forest (và $64.18\% \to 70.01\% \to 74.71\% \to 76.96\%$ với Logistic Regression).
> 3. **Bản chất Sư phạm & Định hướng Tuần 6**: Tại mốc 25% khóa học, các sinh viên đang theo học có tần suất tương tác click ban đầu tương đối đồng nhất và mới chỉ hoàn thành 0–1 bài TMA đầu tiên, khiến tín hiệu suy giảm động lực của nhóm rớt môn về sau còn tiềm ẩn. Đây là thách thức nghiên cứu thực tiễn, tạo động lực cốt lõi để Tuần 6 tập trung khai phá **Đặc trưng Chuỗi thời gian (độ dốc suy giảm click, số ngày không hoạt động liên tiếp)** và **Đặc trưng Tính Quy luật học tập** nhằm bứt phá Recall của nhóm Active tại mốc 25% vượt ngưỡng $\ge 75\%$.

---

## 6. KẾT QUẢ BỘ KIỂM THỬ ĐỘC LẬP (TEST SUITE VERIFICATION)
Tích hợp tại Mục 5 trong [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb)  
Toàn bộ **18/18 unit tests** (12 tests nền Tuần 5 + 6 tests mở rộng Tuần 6) đã chạy thành công tuyệt đối ($100\%$ Passed):

```text
Ran 18 tests in 19.464s
OK
```

1. `test_01_snapshot_files_and_row_counts`: Xác nhận 4 file snapshot đều có đủ $32,593$ dòng và $28,785$ sinh viên duy nhất. (PASSED)
2. `test_02_zero_temporal_leakage_vle`: Xác nhận $0$ bản ghi có $\text{last\_click\_day} > T_{\text{cutoff}}$. (PASSED)
3. `test_03_zero_temporal_leakage_assessments`: Xác nhận chỉ có bài nộp $\le T_{\text{cutoff}}$ được tính. (PASSED)
4. `test_04_exam_strictly_excluded`: Xác nhận 24 bài thi cuối kỳ `Exam` bị loại bỏ $100\%$ khỏi feature đầu vào. (PASSED)
5. `test_05_monotonic_cumulative_features`: Xác nhận tính đơn điệu tăng không giảm ($25\% \le 50\% \le 75\% \le 100\%$) của `total_clicks`, `active_days`, `num_submissions`, `earned_weight_score`. (PASSED)
6. `test_06_activity_types_and_ratios_sum_to_one`: Xác nhận đủ 20 loại activity_type và tổng tỷ trọng bằng $1.0$ khi $\text{total\_clicks} > 0$. (PASSED)
7. `test_07_shannon_entropy_and_diversity_bounds`: Kiểm tra $0 \le \text{activity\_diversity} \le 20$ và $0 \le \text{activity\_entropy} \le \ln(20)$. (PASSED)
8. `test_08_submission_rate_and_weighted_score_invariants`: Kiểm tra $\text{submission\_rate} \in [0, 1]$ và $\min \le \text{weighted\_avg} \le \max$. (PASSED)
9. `test_09_cohort_filtering_definitions`: Xác nhận quy mô 3 định nghĩa cohort khớp số liệu toán học. (PASSED)
10. `test_10_ml_matrix_zero_nan_inf_and_leakage_free`: Xác nhận ma trận học máy sạch $100\%$ NaN/Inf và không rò rỉ thống kê. (PASSED)
11. `test_11_group_stratified_split_zero_overlap`: Xác nhận $0$ sinh viên trùng lặp giữa Train và Test khi chia nhóm. (PASSED)
12. `test_12_catalog_and_metrics_artifacts_exist`: Xác nhận tính toàn vẹn của catalog $76$ đặc trưng và file metrics JSON. (PASSED)

---

## 7. DANH MỤC SẢN PHẨM BÀN GIAO TUẦN 5 (DELIVERABLES)

- [x] **Jupyter Notebook Thực nghiệm Tự chủ Toàn diện (bao gồm lớp `OULADFeatureEngineer`, benchmark & 18/18 Unit Tests PASSED)**: [`notebooks/04_feature_engineering.ipynb`](file:///c:/Users/mrore/Project_I/notebooks/04_feature_engineering.ipynb).
- [x] **4 Ma trận Đặc trưng Snapshot Đã Lưu**:
  - `data/processed/week5_6_snapshots/features_snapshot_25pct.csv` ($32,593 \times 113$ cột đầy đủ Tuần 5–6; chuẩn hóa 76 đặc trưng Tuần 5)
  - `data/processed/week5_6_snapshots/features_snapshot_50pct.csv` ($32,593 \times 113$ cột)
  - `data/processed/week5_6_snapshots/features_snapshot_75pct.csv` ($32,593 \times 113$ cột)
  - `data/processed/week5_6_snapshots/features_snapshot_100pct.csv` ($32,593 \times 113$ cột)
- [x] **Từ điển Danh mục & Độ quan trọng Đặc trưng**:
  - `data/processed/week5_feature_engineering/feature_catalog_week5.csv` (Mô tả chi tiết 76 đặc trưng kèm công thức và chỉ số thống kê).
  - [`data/processed/week5_feature_engineering/feature_importance_week5.json`](file:///c:/Users/mrore/Project_I/data/processed/week5_feature_engineering/feature_importance_week5.json) & [`data/processed/week5_feature_engineering/week5_experiment_metrics.json`](file:///c:/Users/mrore/Project_I/data/processed/week5_feature_engineering/week5_experiment_metrics.json) (92 thực nghiệm mô hình).
- [x] **5 Biểu đồ Báo cáo Độ nét cao (300 DPI)**:
  - `reports/figures/21_feature_engineering_taxonomy_and_distributions.png`
  - `reports/figures/22_activity_diversity_and_entropy_analysis.png`
  - `reports/figures/23_assessment_trajectory_and_weighted_scores.png`
  - `reports/figures/24_snapshot_progression_25_50_75_100.png`
  - `reports/figures/25_preliminary_feature_importance_ranking.png`
- [x] **Tài liệu Báo cáo Khoa học (LaTeX / PDF)**: [`báo_cáo/ProjectI_W5_Report.tex`](file:///c:/Users/mrore/Project_I/báo_cáo/ProjectI_W5_Report.tex) và [`báo_cáo/ProjectI_W5_Report.pdf`](file:///c:/Users/mrore/Project_I/báo_cáo/ProjectI_W5_Report.pdf).
