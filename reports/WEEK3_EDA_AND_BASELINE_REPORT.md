# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & ĐỐI CHUẨN MÔ HÌNH BASELINE
## Tuần 3: Khám phá Dữ liệu (EDA), Kiểm định Thống kê & Chuyển đổi Mô hình 4-Sample sang 2-Sample
## Đề tài: LMS Tiên tiến và Learning Analytics

---

## 1. TỔNG QUAN TIẾN ĐỘ TUẦN 3

Toàn bộ 7 hạng mục công việc của Tuần 3 (từ T3.1 đến T3.7) đã được hoàn thành và đóng gói tự chủ hoàn toàn trong Jupyter Notebook [notebooks/02_eda.ipynb](file:///c:/Users/mrore/Project_I/notebooks/02_eda.ipynb), xuất bản **11 biểu đồ phân tích chuyên sâu** (`04_*.png` đến `14_*.png` tại `reports/figures/`).

| Mã CV | Nội dung công việc Tuần 3 | Trạng thái | Sản phẩm đầu ra |
|:---:|---|:---:|---|
| **T3.1** | Phân tích phân bố clicks và ngày hoạt động theo `final_result` (4 lớp & 2 lớp) | Hoàn thành | [04_clicks_by_final_result.png](file:///c:/Users/mrore/Project_I/reports/figures/04_clicks_by_final_result.png) |
| **T3.2** | Phân tích chuỗi thời gian (Temporal Trends) theo tuần học, tách riêng `Fail` và `Withdrawn` | Hoàn thành | [07_temporal_weekly_clicks_trend.png](file:///c:/Users/mrore/Project_I/reports/figures/07_temporal_weekly_clicks_trend.png) |
| **T3.3** | Phân tích điểm đánh giá (`avg_score`), điểm bài đầu tiên (`TMA1`), tính kịp thời & học trước khai giảng | Hoàn thành | [05_assessment_score_vs_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/05_assessment_score_vs_outcome.png), [06_submission_timeliness_analysis.png](file:///c:/Users/mrore/Project_I/reports/figures/06_submission_timeliness_analysis.png) |
| **T3.4** | Phân tích clicks theo `activity_type` và đối chiếu 3 chiều Khung lý thuyết SEDL | Hoàn thành | [08_activity_type_by_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/08_activity_type_by_outcome.png), [09_sedl_dimensions_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/09_sedl_dimensions_comparison.png) |
| **T3.5** | Phân tích ảnh hưởng của nhân khẩu học (Demographics Cross-tabulation) | Hoàn thành | [10_demographics_vs_at_risk.png](file:///c:/Users/mrore/Project_I/reports/figures/10_demographics_vs_at_risk.png) |
| **T3.6** | Kiểm tra bản ghi trùng 6 cột `studentVle`, ma trận tương quan & kiểm định thống kê (t-test, ANOVA, $\chi^2$) | Hoàn thành | [11_correlation_matrix_heatmap.png](file:///c:/Users/mrore/Project_I/reports/figures/11_correlation_matrix_heatmap.png), [12_statistical_tests_effect_size.png](file:///c:/Users/mrore/Project_I/reports/figures/12_statistical_tests_effect_size.png) |
| **T3.7** | Chạy thử mô hình bài báo dự đoán cũ, chuyển cấu hình từ **4 sample** sang **2 sample** | Hoàn thành | [13_baseline_4class_vs_2class_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/13_baseline_4class_vs_2class_comparison.png), [14_baseline_confusion_matrices_and_roc.png](file:///c:/Users/mrore/Project_I/reports/figures/14_baseline_confusion_matrices_and_roc.png) |

---

## 2. KIỂM ĐỊNH MỞ RỘNG BẢNG `studentVle` & KHÓA NGOẠI TỔ HỢP (T3.6)

Tiếp nối khuyến nghị từ Báo cáo Tuần 2, nghiên cứu đã kiểm tra chi tiết bản chất trùng lặp của `studentVle` (10,655,280 dòng) và đối chiếu khóa ngoại tổ hợp:

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

## 3. GIẢI THÍCH CÁC ĐỊNH NGHĨA VÀ ĐẠI LƯỢNG KIỂM ĐỊNH THỐNG KÊ

Nhằm giúp người đọc và hội đồng thẩm định hiểu rõ toàn bộ nền tảng toán học, công thức tính toán và ý nghĩa thực tiễn của các đại lượng thống kê được sử dụng xuyên suốt đề tài, phần này tổng hợp tường minh các định nghĩa cốt lõi:

### 3.1. Thống kê mô tả cho phân phối lệch phải (Skewed Distribution Metrics)
- **Trung bình số học (Mean $\bar{x}$)**: 
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$
  *Ý nghĩa & Hạn chế*: Phản ánh mức kỳ vọng tổng thể nhưng cực kỳ nhạy cảm với các giá trị ngoại lai (outliers). Trong OULAD, giá trị trung bình lượt click thường lớn hơn trung vị từ 1.5 đến 3.5 lần.
- **Trung vị (Median $\tilde{x}$) và Khoảng tứ phân vị (IQR)**:
  $$\text{IQR} = Q_3 - Q_1$$
  với $Q_1$ là phân vị 25% và $Q_3$ là phân vị 75%. Đây là các số đo thống kê bền vững (Robust Statistics), mô tả chính xác xu hướng trung tâm của đại đa số người học mà không bị méo mó bởi nhóm sinh viên siêu tích cực.
- **Hệ số bất đối xứng Skewness ($\gamma_1$) & Độ nhọn Kurtosis ($\beta_2$)**:
  $$\text{Skewness} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{\left[\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right]^{3/2}}, \quad \text{Kurtosis} = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^4}{\left[\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right]^2}$$
  Skewness của `total_clicks` đạt $+3.03$ khẳng định phân phối lệch phải cực đoan; Kurtosis đạt $14.95$ cho thấy đuôi rất nặng (Leptokurtic), minh chứng sự cần thiết phải áp dụng kỹ thuật Capping kết hợp biến đổi $\ln(1+x)$.

### 3.2. Kiểm định tham số so sánh trung bình: Welch's $t$-test & One-Way ANOVA
- **Welch's $t$-test**: So sánh 2 nhóm độc lập (`Not At-Risk` vs `At-Risk`) với giả định phương sai không đồng nhất ($\sigma_1^2 \ne \sigma_2^2$):
  $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}, \quad \nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$
  Do nhóm an toàn có phương sai click lớn gấp nhiều lần nhóm nguy cơ ($s_1^2 \gg s_2^2$) và cỡ mẫu hai nhóm chênh lệch ($n_1 = 15,385 \ne n_2 = 17,208$), Welch's $t$-test là lựa chọn chuẩn xác duy nhất về mặt toán học thay thế Student's $t$-test truyền thống.
- **One-Way ANOVA ($F$-test)**: So sánh giá trị trung bình trên 4 nhóm kết quả gốc (`Distinction`, `Pass`, `Fail`, `Withdrawn`):
  $$F = \frac{\text{MSB}}{\text{MSW}} = \frac{\sum_{i=1}^k n_i (\bar{x}_i - \bar{x})^2 / (k - 1)}{\sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 / (N - k)}$$
  trong đó $\text{MSB}$ là phương sai giữa các nhóm (Mean Square Between) và $\text{MSW}$ là phương sai nội bộ các nhóm (Mean Square Within).

### 3.3. Độ lớn hiệu ứng (Effect Size) & Nghịch lý Mẫu lớn (Big Data Paradox)
- **Nghịch lý giá trị $p$ ($p$-value Paradox)**: Khi kích thước mẫu lớn ($N = 32,593$), sai số chuẩn của ước lượng $\text{SE} = \frac{s}{\sqrt{N}}$ thu nhỏ tiệm cận $0$. Khi đó giá trị thống kê $t = \frac{\bar{x}_1 - \bar{x}_2}{\text{SE}}$ bùng nổ, khiến hầu như mọi sai khác số học dù vụn vặt và không mang ý nghĩa sư phạm nào cũng đều cho $p < 0.0001$. Do đó, $p$-value chỉ khẳng định "có khác biệt", còn **Effect Size** mới xác định "khác biệt đó có ý nghĩa thực tế hay không".
- **Hệ số Cohen's $d$**: Lượng hóa độ lớn hiệu ứng cho 2 nhóm độc lập:
  $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
  Quy chuẩn Cohen (1988): $|d| \approx 0.2$ (nhỏ), $|d| \approx 0.5$ (trung bình), $|d| \ge 0.8$ (lớn). Trong dự án, `active_days` đạt $d = 1.608$ (hiệu ứng cực lớn), phân hóa vượt trội hoàn toàn.

### 3.4. Kiểm định tính độc lập Chi-Square ($\chi^2$) & Hệ số Cramér's $V$
- **Kiểm định Chi-Square ($\chi^2$)**: Đánh giá sự độc lập giữa biến phân loại (nhân khẩu học) và nhãn nguy cơ:
  $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \times C_j}{N}, \quad df = (r-1)(c-1)$$
- **Hệ số liên kết Cramér's $V$**: Đo lường cường độ liên kết chuẩn hóa về thang $[0, 1]$:
  $$V = \sqrt{\frac{\chi^2}{N \times \min(r - 1, c - 1)}}$$
  Ngưỡng diễn giải: $V < 0.1$ (không đáng kể), $0.1 \le V < 0.3$ (liên kết yếu đến trung bình), $V \ge 0.3$ (liên kết mạnh).

### 3.5. Quy đổi chuẩn hóa Cohen's $d$ sang Hệ số tương quan Point-Biserial $r$
Toán học cho phép so sánh trực tiếp biến hành vi định lượng ($d$) với biến nhân khẩu học định tính ($V$) thông qua công thức quy đổi sang độ lớn hệ số tương quan Point-Biserial $|r| \in [0, 1]$ (với $r \in [-1, 1]$):
$$|r| = \frac{|d|}{\sqrt{d^2 + \frac{1}{p(1 - p)}}}$$
Với tỷ lệ lớp Not At-Risk $p = 15,385 / 32,593 \approx 0.472 \Rightarrow \frac{1}{p(1-p)} \approx 4.0125$. Nhờ đó, $d = 1.608$ của `active_days` cho độ lớn tương quan $|r| \approx 0.626$ (tức $r \approx -0.626$ khi xét tương quan nghịch với nhãn nguy cơ `At-Risk = 1`), cao gấp hơn 4 lần biến nhân khẩu học có liên kết mạnh nhất là `highest_education` ($V = 0.1504$).

### 3.6. Kiểm định phi tham số & Hiệu chỉnh kiểm định bội
- **Kiểm định Mann-Whitney $U$**: Dựa trên phép xếp hạng thứ tự toàn bộ mẫu gộp ($N = n_1 + n_2$), không đòi hỏi giả định phân phối chuẩn:
  $$U_1 = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1, \quad U_2 = n_1 n_2 - U_1, \quad U = \min(U_1, U_2)$$
  Xấp xỉ phân phối chuẩn với cỡ mẫu lớn: $z = \frac{U - \mu_U}{\sigma_U}$ với $\mu_U = \frac{n_1 n_2}{2}$ và $\sigma_U = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}$.
- **Kiểm định Kruskal-Wallis $H$**: Phiên bản phi tham số của ANOVA kiểm định sự đồng nhất phân bố giữa 4 nhóm kết quả học tập gốc:
  $$H = \frac{12}{N(N + 1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N + 1), \quad df = k - 1 = 3$$
- **Hiệu chỉnh Bonferroni & Holm Step-down**: Kiểm soát sai lầm loại I (tích tụ dương tính giả) khi kiểm định đồng thời $m = 11$ đặc trưng với $\alpha' = 0.05 / 11 \approx 0.0045$. Toàn bộ 11 biến đều đạt $p < 10^{-4} \ll \alpha'$. Thủ tục tuần tự Holm Step-down: sắp xếp $p_{(1)} \le \dots \le p_{(m)}$, tại mỗi bước $k$ kiểm tra $p_{(k)} \le \frac{\alpha}{m - k + 1}$.

### 3.7. Các chỉ số hành vi và nhận thức học tập đặc thù LMS (Khung lý thuyết SEDL)
Nhằm chuyển hóa dữ liệu nhật ký vết click thô (Clickstream logs) và bản ghi học vụ thành các thông số định lượng phản ánh tiến trình nhận thức và thái độ học tập theo Khung lý thuyết SEDL (Student Engagement in Digital Learning), dự án định nghĩa và chuẩn hóa toán học 5 chỉ số cốt lõi:
- **Tỷ lệ hoàn thành nhiệm vụ đánh giá (`submission_rate`)**:
  $$\text{submission\_rate} = \frac{\sum_{i=1}^{M_{\text{due}}} \mathbb{I}(\text{assessment}_i \text{ is submitted})}{M_{\text{due}}}$$
  *Ý nghĩa sư phạm*: Thước đo trực tiếp phản ánh sự tuân thủ quy chế đào tạo (Academic Compliance) và cam kết học tập (Commitment). Việc bỏ nộp dù chỉ một bài kiểm tra đầu tiên (như `TMA1`) là chỉ dấu cảnh báo sớm rõ nét nhất của nguy cơ bỏ học giữa chừng.
- **Mức độ chủ động thời gian / Nộp bài sớm (`avg_days_early`)**:
  $$\text{avg\_days\_early} = \frac{1}{|A_{\text{sub}}|} \sum_{i \in A_{\text{sub}}} (\text{date\_cutoff}_i - \text{date\_submission}_i)$$
  *Ý nghĩa*: Đại diện cho năng lực Tự điều chỉnh học tập (Self-Regulated Learning - SRL). Giá trị dương ($\text{avg\_days\_early} > 0$) thể hiện tính chủ động và thói quen lập kế hoạch học tập chu đáo; giá trị âm phản ánh xu hướng trì hoãn học tập (Academic Procrastination) hoặc sự quá tải học thuật.
- **Số ngày tích cực tương tác (`active_days`)**:
  $$\text{active\_days} = \sum_{t=1}^{T} \mathbb{I}\left( \sum_{a \in \mathcal{A}} \text{clicks}(t, a) > 0 \right)$$
  *Ý nghĩa*: Đo lường sự bền bỉ, kiên trì (Grit) và thói quen phân bổ việc học đều đặn (Distributed Practice) thay vì học dồn cục bộ. Trong thực nghiệm OULAD, đây là đặc trưng có độ phân tách mạnh mẽ nhất ($d = 1.608$).
- **Cường độ tương tác trong ngày hoạt động (`clicks_per_active_day`)**:
  $$\text{clicks\_per\_active\_day} = \frac{\text{total\_clicks}}{\max(\text{active\_days}, 1)}$$
  *Ý nghĩa*: Phân biệt giữa người học đều đặn có nhịp độ học tập ổn định (Steady Learners) và người học dồn ứ (Crammers - dồn hàng nghìn lượt click vào 1-2 ngày ngay trước hạn nộp bài để đối phó).
- **Tương tác chuẩn bị trước thềm khai giảng (`pre_course_clicks`)**:
  $$\text{pre\_course\_clicks} = \sum_{t < 0} \sum_{a \in \mathcal{A}} \text{clicks}(t, a)$$
  với $t < 0$ là các mốc thời gian trước ngày khai giảng chính thức ($t = 0$).
  *Ý nghĩa*: Đo lường mức độ sẵn sàng học tập (Academic Readiness), động lực nội tại và sự chủ động nghiên cứu đề cương, tài nguyên nhập môn trước khi môn học bắt đầu.

### 3.8. Các độ đo đánh giá mô hình học máy & Kiểm định McNemar
- **Ma trận nhầm lẫn (Confusion Matrix)**: Gồm TP, FP, TN, FN. Lớp dương tính ($1$) là học viên có nguy cơ (`At-Risk`), lớp âm tính ($0$) là học viên an toàn (`Not At-Risk`).
- **Accuracy, Precision, Recall và $F_1$-Score**:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  $$\text{Recall (Độ bao phủ nguy cơ)} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Lý giải Sư phạm về Recall - "Chỉ số sinh mệnh" trong Cảnh báo sớm**:
  Trong giáo dục trực tuyến, chi phí sai số mang tính bất đối xứng nghiêm trọng (*Asymmetric Misclassification Costs*):
  - *Bỏ sót sinh viên nguy cơ (False Negative - FN)*: Đây là tổn thất sư phạm tồi tệ nhất. Sinh viên thực sự gặp khó khăn trong học tập hoặc có ý định bỏ học bị mô hình đánh giá nhầm là an toàn, dẫn đến việc không nhận được bất kỳ sự trợ giúp, nhắc nhở hay can thiệp kịp thời nào từ giảng viên và cố vấn học tập. Hậu quả trực tiếp là sinh viên trượt môn hoặc bỏ học vĩnh viễn, đánh mất thời gian, chi phí học phí và cơ hội học tập.
  - *Cảnh báo nhầm sinh viên an toàn (False Positive - FP)*: Mô hình dự báo một sinh viên có năng lực tốt vào diện có nguy cơ. Hậu quả thực tế chỉ là sinh viên đó nhận được một email nhắc nhở động viên từ hệ thống LMS hoặc một lời mời tư vấn ngắn từ cố vấn học tập. Sinh viên chỉ cần vài giây để xác nhận tình trạng tốt của mình, chi phí can thiệp gần như bằng $0$ và hoàn toàn không gây bất kỳ tác hại nào cho lộ trình học tập.
  
  Vì vậy, **Recall được xác định là "chỉ số sinh mệnh" tối thượng**: một hệ thống cảnh báo sớm bắt buộc phải tối đa hóa Recall (hạn chế tối đa FN) trong khi duy trì Precision ở mức chấp nhận được.
- **Diện tích dưới đường cong ROC (ROC-AUC)**:
  $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt = P(S_{\text{pos}} > S_{\text{neg}})$$
  Năng lực xếp hạng dự báo tổng quát không phụ thuộc vào ngưỡng phân loại $\tau \in [0, 1]$.
- **Kiểm định McNemar (So sánh 2 mô hình phân loại trên cùng tập Test)**:
  Lập bảng tiếp liên $2 \times 2$ của các trường hợp bất đồng giữa mô hình A và B trên cùng tập Test:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad df = 1$$
  trong đó $b$ là số mẫu mô hình A đoán đúng nhưng B đoán sai, $c$ là số mẫu A đoán sai nhưng B đoán đúng (có hiệu chỉnh Yates). Kết quả thực nghiệm cho $p > 0.40$, khẳng định tính tương đương toán học vững chắc giữa mô hình gộp xác suất $4 \rightarrow 2$ và mô hình nhị phân $2$-class trực tiếp.

---

## 4. CÁC PHÁT HIỆN CHÍNH TỪ PHÂN TÍCH KHÁM PHÁ (KEY EDA FINDINGS)

### 4.1. Chiều Hành vi (Behavioural Engagement) - Kiểm định Giả thuyết H1a
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

### 4.2. Chiều Nhận thức (Cognitive Engagement), Điểm `TMA1` & Tính Kịp thời - Kiểm định Giả thuyết H1b (T3.3)
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

### 4.3. Phân tích Chuỗi thời gian (Temporal Trends) & Sự Phân hóa `Fail` vs `Withdrawn` (T3.2)
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

### 4.4. Chiều Xã hội (Social Engagement - `forumng`) & Đa chiều SEDL - Kiểm định Giả thuyết H1c
Biểu đồ minh họa: [reports/figures/08_activity_type_by_outcome.png](file:///c:/Users/mrore/Project_I/reports/figures/08_activity_type_by_outcome.png) và [reports/figures/09_sedl_dimensions_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/09_sedl_dimensions_comparison.png)

| Mức độ tương tác Diễn đàn (`forumng_clicks`) | Distinction (%) | Pass (%) | Fail (%) | Withdrawn (%) | Tổng At-Risk (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **0 clicks (Cô lập xã hội hoàn toàn)** | 0.87% | 6.04% | 22.15% | **70.94%** | **93.09%** |
| **1 - 20 clicks (Tương tác thấp)** | 3.64% | 24.80% | 33.37% | 38.19% | **71.56%** |
| **21 - 100 clicks (Tương tác trung bình)** | 6.24% | 38.75% | 27.35% | 27.66% | **55.01%** |
| **> 100 clicks (Tương tác tích cực)** | 16.60% | **56.69%** | 14.62% | **12.09%** | **26.71%** |

> **Kết luận H1c (ĐƯỢC XÁC NHẬN)**: Sự cô lập xã hội trên môi trường học tập số (`forumng_clicks = 0`) gắn liền với tỷ lệ bỏ học giữa chừng lên tới **70.94%** (tổng nguy cơ `At-Risk` là **93.09%**). Ngược lại, sinh viên tham gia diễn đàn trên 100 clicks có tỷ lệ đỗ (`Pass + Distinction`) đạt **73.29%** và tỷ lệ bỏ học giảm gần 6 lần xuống chỉ còn **12.09%** ($F = 1,238.15, p < 0.0001$).

---

### 4.5. Ảnh hưởng của Nhân khẩu học & Kiểm định Thống kê Tổng hợp (T3.5 & T3.6)
Biểu đồ minh họa: [reports/figures/10_demographics_vs_at_risk.png](file:///c:/Users/mrore/Project_I/reports/figures/10_demographics_vs_at_risk.png), [reports/figures/11_correlation_matrix_heatmap.png](file:///c:/Users/mrore/Project_I/reports/figures/11_correlation_matrix_heatmap.png), [reports/figures/12_statistical_tests_effect_size.png](file:///c:/Users/mrore/Project_I/reports/figures/12_statistical_tests_effect_size.png)

- **Các nhóm nhân khẩu học có rủi ro cao nhất**:
  1. **Trình độ học vấn đầu vào (`highest_education`)**: Nhóm `No Formal quals` có tỷ lệ At-Risk **70.32%** và `Lower Than A Level` là **61.15%**, cao gấp đôi so với `Post Graduate Qualification` (**34.50%**, $\chi^2 = 737.25, p < 0.0001$, Cramer's $V = 0.1504$).
  2. **Chỉ số kinh tế xã hội vùng (`imd_band`)**: Giảm đơn điệu đều đặn từ vùng nghèo nhất `0-10%` (**64.84% At-Risk**) xuống vùng khá giả nhất `90-100%` (**42.47% At-Risk**, $\chi^2 = 721.97, p < 0.0001$, Cramer's $V = 0.1488$).
  3. **Tiền sử học lại (`num_of_prev_attempts`)**: Sinh viên học lần đầu có tỷ lệ At-Risk **50.72%**, tăng lên **65.44%** nếu đã trượt 1 lần và **72.51%** nếu trượt $\ge 2$ lần.
  4. **Áp lực tín chỉ (`studied_credits`)**: Sinh viên đăng ký quá tải (`> 120 tín chỉ`) có tỷ lệ At-Risk **69.64%**, so với **48.20%** ở nhóm học vừa sức (`<= 60 tín chỉ`).
  5. **Khuyết tật (`disability`)**: Nhóm có khuyết tật (`Y`) có tỷ lệ At-Risk **61.85%** so với **51.82%** ở nhóm không khuyết tật ($\chi^2 = 114.90, p < 0.0001$).
- **So sánh Độ lớn Hiệu ứng (Effect Size)**: Mặc dù mọi biến nhân khẩu học đều có $p < 0.0001$ (do kích thước mẫu lớn $N = 32,593$), hệ số tương quan và độ lớn hiệu ứng của nhân khẩu học ($|r| \in [0.10, 0.15]$, Cramer's $V \le 0.15$) **nhỏ hơn rất nhiều** so với các đặc trưng tương tác LMS theo khung SEDL (`submission_rate`: $d = 2.829$, `active_days`: $d = 1.608$, `avg_score`: $d = 1.507$, `tma1_score`: $d = 1.219$, `total_clicks`: $d = 1.086$, `clicks_oucontent`: $d = 0.823$). Điều này chứng minh **hành vi nỗ lực thực tế trong khóa học quyết định kết quả nhiều hơn xuất phát điểm nhân khẩu học**.

---

## 5. THỰC NGHIỆM MÔ HÌNH BÀI BÁO CŨ: CHUYỂN CẤU HÌNH 4-SAMPLE SANG 2-SAMPLE (NHIỆM VỤ T3.7)

Biểu đồ minh họa: [reports/figures/13_baseline_4class_vs_2class_comparison.png](file:///c:/Users/mrore/Project_I/reports/figures/13_baseline_4class_vs_2class_comparison.png) và [reports/figures/14_baseline_confusion_matrices_and_roc.png](file:///c:/Users/mrore/Project_I/reports/figures/14_baseline_confusion_matrices_and_roc.png)

### 5.1. Thiết kế thực nghiệm đối chuẩn
Nghiên cứu tái hiện bộ mô hình phân loại cơ sở từ các bài báo gốc trên OULAD (Kuzilek et al., 2017; Hlosta et al., 2017; Jayaprakash et al., 2014) gồm 4 thuật toán (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`) trên cùng phép chia Stratified Train/Test (80% Train = 26,074 mẫu; 20% Test = 6,519 mẫu) để so sánh 3 cấu hình nhãn:
1. **Cấu hình 4-Sample Gốc (4-Class Multiclass)**: Dự đoán 4 lớp `Distinction`, `Pass`, `Fail`, `Withdrawn`.
2. **Cấu hình 4-sang-2 Hậu nghiệm (Post-hoc Probability Aggregation)**: Huấn luyện mô hình 4 lớp nhưng gộp xác suất dự đoán $P(\text{At-Risk}) = P(\text{Fail}) + P(\text{Withdrawn})$.
3. **Cấu hình 2-Sample Trực tiếp (Direct Binary Target Adaptation)**: Chuyển đổi trực tiếp mục tiêu bài toán sang 2 lớp `0: Not At-Risk` (`Pass + Distinction`) và `1: At-Risk` (`Fail + Withdrawn`).

### 5.2. Bảng đối sánh kết quả giữa Cấu hình 4-Sample và 2-Sample (Toàn khóa)

| Mô hình Baseline | 4-Sample Accuracy | 4-Sample Macro-F1 | 4-Sample F1 (`Fail`) | 4$\rightarrow$2 Mapped F1 | **2-Sample Accuracy** | **2-Sample Precision** | **2-Sample Recall** | **2-Sample F1-Score** | **2-Sample ROC-AUC** | **Mức tăng F1 (+%)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 70.82% | 63.39% | 41.89% | 90.46% | **90.12%** | 92.27% | 88.73% | **90.46%** | 0.9600 | **+27.07%** |
| **Decision Tree** | 70.32% | 62.26% | 39.98% | 90.87% | **91.07%** | 95.98% | 86.72% | **91.12%** | 0.9669 | **+28.86%** |
| **Random Forest** | 71.38% | 62.37% | 40.78% | 91.40% | **91.64%** | 95.45% | 88.38% | **91.78%** | 0.9720 | **+29.41%** |
| **Gradient Boosting** | 72.82% | 66.22% | 47.53% | 92.84% | **92.28%** | 95.04% | **90.09%** | **92.50%** | **0.9776** | **+26.28%** |

### 5.3. Tại sao chuyển từ 4-Sample sang 2-Sample mang lại bước nhảy vọt hiệu năng (+26% đến +29% F1)?
Phân tích Ma trận nhầm lẫn (Confusion Matrix) của mô hình `Random Forest` trên tập Test ($N = 6,519$):
- **Trong cấu hình 4-Sample**:
  - Giữa `Distinction` và `Pass`: Trong 605 sinh viên `Distinction` thực tế, có tới **381 sinh viên (63.0%) bị đoán nhầm thành `Pass`** do cả hai nhóm đều có hành vi tương tác và nộp bài đầy đủ suốt khóa học.
  - Giữa `Fail` và `Withdrawn`: Trong 1,411 sinh viên `Fail` thực tế, có **539 sinh viên (38.2%) bị đoán nhầm thành `Withdrawn`** và **431 sinh viên (30.5%) bị đoán nhầm thành `Pass`**, khiến F1-Score của lớp `Fail` chỉ đạt **40.78%**.
- **Khi chuyển sang cấu hình 2-Sample (`Not At-Risk` vs `At-Risk`)**:
  - Mọi sự nhầm lẫn nội bộ giữa `Distinction <-> Pass` (cùng thuộc nhóm An toàn `0`) và giữa `Fail <-> Withdrawn` (cùng thuộc nhóm Nguy cơ cần can thiệp `1`) được triệt tiêu hợp lý về mặt nghiệp vụ sư phạm!
  - Ma trận nhầm lẫn 2-Sample của `Random Forest` đạt độ phân tách sắc nét: nhận diện đúng **2,932 / 3,077 (95.3%)** sinh viên `Not At-Risk` và phát hiện đúng **3,042 / 3,442 (88.4%)** sinh viên `At-Risk`.

### 5.4. Kết quả đối chuẩn Cảnh báo sớm (Early Warning Snapshots: 25%, 50%, 100%)

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

## 6. DANH MỤC SẢN PHẨM BÀN GIAO TUẦN 3 (DELIVERABLES CHECKLIST)

- [x] Notebook phân tích khám phá, kiểm định thống kê & đối chuẩn baseline đầy đủ (tự chủ 100%, nhúng sẵn bảng và 11 biểu đồ): [notebooks/02_eda.ipynb](file:///c:/Users/mrore/Project_I/notebooks/02_eda.ipynb)
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
- [x] Bảng dữ liệu đặc trưng tổng hợp cấp độ Sinh viên – Khóa học ($32,593 \times 51$): [`student_course_eda_features.csv`](file:///c:/Users/mrore/Project_I/data/processed/week3_eda/student_course_eda_features.csv) (các kết quả kiểm định và đối chuẩn baseline lưu trực tiếp trong kết quả thực thi của `notebooks/02_eda.ipynb`)
- [x] Báo cáo tổng hợp Tuần 3 (Markdown & LaTeX): [WEEK3_EDA_AND_BASELINE_REPORT.md](file:///c:/Users/mrore/Project_I/reports/WEEK3_EDA_AND_BASELINE_REPORT.md) & [ProjectI_W3_Report.tex](file:///c:/Users/mrore/Project_I/báo_cáo/ProjectI_W3_Report.tex) (và [ProjectI_W3_Report.pdf](file:///c:/Users/mrore/Project_I/báo_cáo/ProjectI_W3_Report.pdf))
