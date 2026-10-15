# TỔNG QUAN TÀI LIỆU NGHIÊN CỨU (LITERATURE REVIEW)
## Đề tài: LMS Tiên tiến và Learning Analytics - Dự đoán Nguy cơ Học tập của Sinh viên

---

## 1. BÀI BÁO NỀN TẢNG: Bergdahl et al. (2024)

- **Trích dẫn**: Bergdahl, N., Bond, M., Sjöden, J., & Dane, M. (2024). *Engagements in digital learning: A systematic review of learning analytics research on student engagement*. International Journal of Educational Technology in Higher Education, 21(1), 63. DOI: 10.1186/s41239-024-00493-y.
- **Mục tiêu nghiên cứu**: Tổng quan hệ thống (Systematic Literature Review) về cách các nghiên cứu Learning Analytics (LA) định nghĩa, đo lường và vận hành hóa khái niệm "Student Engagement" trong môi trường học tập số.

### 1.1. Framework SEDL (Student Engagement in Digital Learning)
Bài báo phân loại sự tham gia của người học (Engagement) và sự giảm sút tương tác (Disengagement) thành 4 chiều cốt lõi:

| Chiều Engagement | Khái niệm & Ý nghĩa | Chỉ số đo lường trong LA (Engagement Indicators) | Dấu hiệu Disengagement tương ứng |
|---|---|---|---|
| **Behavioural Engagement** (Hành vi) | Mức độ nỗ lực, tham gia thực tế, chuyên cần và tuân thủ các hoạt động học tập. | - Số lượt click (clicks count)<br>- Thời gian trên hệ thống (time-on-task)<br>- Tần suất đăng nhập (login frequency)<br>- Tỷ lệ hoàn thành bài tập (assignment completion rate)<br>- Điểm danh / tham gia buổi học (attendance) | - Không hoàn thành bài tập (Task incompletion)<br>- Vắng mặt kéo dài (Absence)<br>- Đăng nhập thụ động/không tương tác (Lurking)<br>- Rút lui / Bỏ học (Drop out / Withdrawn) |
| **Cognitive Engagement** (Nhận thức) | Mức độ đầu tư trí tuệ, tự điều chỉnh việc học (self-regulation), tư duy phản biện và chiến lược học sâu. | - Tỷ lệ đọc tài liệu vs làm bài kiểm tra<br>- Tần suất xem lại bài giải / tài liệu bổ trợ<br>- Khoảng thời gian nộp bài trước hạn chót (procrastination / submission timeliness)<br>- Tính đều đặn trong chu kỳ học theo tuần | - Hoang mang, lúng túng (Confusion)<br>- Thiếu tập trung, lơ là (Distracted)<br>- Học vẹt, học bề nổi (Superficial processing)<br>- Học dồn vào phút chót (Cramming) |
| **Emotional Engagement** (Cảm xúc) | Thái độ, hứng thú, sự hài lòng, trạng thái tâm lý tích cực đối với khóa học và giảng viên. | - Tương tác tự nguyện (ngoài yêu cầu bắt buộc)<br>- Đánh giá phản hồi khóa học (course feedback/surveys)<br>- Cảm xúc văn bản trong bài viết diễn đàn (Sentiment analysis) | - Chán nản (Boredom)<br>- Thất vọng, bức xúc (Frustration)<br>- Lo lắng quá mức (Anxiety)<br>- Lãnh đạm, thờ ơ (Apathy) |
| **Social Engagement** (Xã hội) | Tương tác giữa người học với bạn học và giảng viên, tinh thần hợp tác, cảm giác gắn kết cộng đồng. | - Số bài viết / phản hồi trên diễn đàn (Forum posts / replies)<br>- Mạng lưới tương tác xã hội (Social Network Analysis - SNA)<br>- Tương tác trong các hoạt động nhóm (Collaborative wiki / peer review) | - Cô lập xã hội (Social isolation)<br>- Bỏ qua ý kiến của bạn học (Ignoring others)<br>- Không tìm kiếm sự trợ giúp khi gặp bế tắc (Giving up on help-seeking) |

### 1.2. Phát hiện quan trọng từ bài báo (Key Findings)
1. **Sự mất cân đối nghiêm trọng**: Có tới **95% nghiên cứu LA** chỉ tập trung đo lường **Behavioural Engagement** thông qua log thô (clicks, logins, hits). Các chiều Cognitive, Emotional và Social bị bỏ qua hoặc chỉ được đo lường gián tiếp.
2. **Thiếu tính định lượng đa chiều**: Hầu hết các mô hình cảnh báo sớm (Early Alert Systems) quy đổi "tương tác nhiều = học tốt", dẫn đến sai số lớn khi sinh viên "click ảo" hoặc đăng nhập nhưng không học (passive lurking).
3. **Cơ hội nghiên cứu cho đề tài**: Tận dụng bộ dữ liệu OULAD để trích xuất các đặc trưng đa chiều:
   - *Behavioural*: Tổng số click, số ngày active, số bài nộp đúng hạn.
   - *Cognitive*: Chỉ số nộp bài trước hạn (early submission days), tỷ lệ click tài nguyên kiến thức (`oucontent`, `subpage`) so với tài nguyên đánh giá (`quiz`, `url`).
   - *Social*: Số lần truy cập diễn đàn (`forumng`), hoạt động thảo luận wiki (`ouwiki`).

---

## 2. BỐN BÀI BÁO LIÊN QUAN TRỌNG TÂM

### 2.1. Paper 1: Giới thiệu Dataset OULAD
- **Tác giả**: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017).
- **Tên bài báo**: *Open University Learning Analytics Dataset*. Scientific Data (Nature), 4:170171. DOI: 10.1038/sdata.2017.171.
- **Tóm tắt cốt lõi**:
  - Công bố bộ dữ liệu mở OULAD bao gồm dữ liệu từ The Open University (Anh Quốc) với 32,593 sinh viên, 22 khóa học qua 4 kỳ (2013B, 2013J, 2014B, 2014J).
  - Cung cấp 7 bảng quan hệ: dữ liệu nhân khẩu học, đăng ký, bài kiểm tra, điểm số và hơn 10.6 triệu bản ghi log tương tác VLE theo từng ngày.
  - Phân loại kết quả học tập thành 4 nhóm: Distinction, Pass, Fail, Withdrawn.
- **Ý nghĩa với dự án**: Cung cấp cấu trúc chuẩn, định dạng dữ liệu và metadata để chuẩn bị pipeline xử lý dữ liệu.

### 2.2. Paper 2: Dự đoán sớm sinh viên có nguy cơ (Ouroboros)
- **Tác giả**: Hlosta, M., Zdrahal, Z., & Zendulka, J. (2017).
- **Tên bài báo**: *Ouroboros: Early identification of at-risk students without models based on legacy data*. Proceedings of the 7th International Learning Analytics & Knowledge Conference (LAK '17), pp. 256–265.
- **Tóm tắt cốt lõi**:
  - Đề xuất hệ thống Ouroboros để phát hiện sinh viên có nguy cơ bỏ học/trượt môn ngay từ những tuần đầu tiên mà không cần dữ liệu lịch sử của khóa học đó.
  - Sử dụng đặc trưng tích lũy theo thời gian (running cumulative clicks, running assessment scores).
  - Kết luận: Dự đoán ở mốc 25% - 30% thời lượng khóa học có thể đạt Recall trên 75% cho nhóm sinh viên có nguy cơ, tạo cơ hội vàng để cố vấn học tập can thiệp.
- **Ý nghĩa với dự án**: Định hình bài toán **Early Prediction** tại các mốc thời gian 25%, 50%, và 75% của khóa học.

### 2.3. Paper 3: Khai phá quy trình học tập (Process Mining)
- **Tác giả**: Umer, R., Susnjak, T., Mathrani, A., & Suriadi, S. (2017).
- **Tên bài báo**: *On predicting academic performance with process mining in learning analytics*. Journal of Research in Innovative Teaching & Learning, 10(2), 160-176.
- **Tóm tắt cốt lõi**:
  - Không chỉ đếm số lượng click, nghiên cứu xem xét **chuỗi hành vi** (sequences of learning activities) và luồng tương tác giữa các loại tài nguyên.
  - Sinh viên đạt kết quả cao thường có lộ trình học bài bản: xem tài liệu lý thuyết -> làm quiz tự luyện -> tham gia diễn đàn -> nộp bài đánh giá.
  - Sinh viên trượt/bỏ cuộc thường chỉ nhảy thẳng vào bài kiểm tra mà không đọc tài liệu, hoặc có chu kỳ ngắt quãng dài ngày.
- **Ý nghĩa với dự án**: Gợi ý việc tạo đặc trưng tỷ lệ tương tác theo loại tài nguyên (`oucontent`, `forumng`, `quiz`) và độ biến động hoạt động (standard deviation của clicks theo tuần).

### 2.4. Paper 4: Hệ thống cảnh báo sớm trong giáo dục đại học
- **Tác giả**: Jayaprakash, S. M., Moody, E. W., Lauría, E. J., Regan, J. R., & Baron, J. D. (2014).
- **Tên bài báo**: *Early alert of academically at-risk students: An open source analytics initiative*. Journal of Learning Analytics, 1(1), 6-47.
- **Tóm tắt cốt lõi**:
  - Triển khai hệ thống cảnh báo sớm mã nguồn mở (Open Academic Analytics Initiative - OAAI).
  - So sánh các mô hình học máy (Logistic Regression, Decision Tree, Naive Bayes, SVM) trong việc phân loại sinh viên at-risk.
  - Nhấn mạnh sự đánh đổi giữa Precision và Recall: trong bài toán giáo dục, **Recall (độ nhạy)** quan trọng hơn Precision vì bỏ sót một sinh viên cần trợ giúp gây hậu quả nghiêm trọng hơn việc cảnh báo nhầm (false alarm).
- **Ý nghĩa với dự án**: Lựa chọn metric đánh giá ưu tiên Recall và F1-Score (hoặc PR-AUC) cho lớp At-Risk, thay vì chỉ dựa vào Accuracy do sự mất cân bằng lớp.

---

## 3. BẢNG SO SÁNH TỔNG HỢP CÁC NGHIÊN CỨU

| Tiêu chí | Bergdahl et al. (2024) | Kuzilek et al. (2017) | Hlosta et al. (2017) | Umer et al. (2017) | Jayaprakash et al. (2014) | **Dự án của chúng ta** |
|---|---|---|---|---|---|---|
| **Loại hình** | Systematic Review (131 papers) | Dataset Paper & Benchmark | Research Paper (Early Prediction) | Research Paper (Process Mining) | Empirical Study (OAAI System) | Nghiên cứu ứng dụng + Demo XAI |
| **Dataset** | Nhiều nguồn khác nhau | OULAD (~32.6k SV, 22 courses) | OULAD (các module cụ thể) | Dữ liệu Moodle / OULAD | Dữ liệu Sakai LMS (Marist College) | **OULAD (Toàn bộ 7 bảng CSV)** |
| **Mục tiêu chính** | Tổng hợp định nghĩa & chỉ số engagement | Công bố dữ liệu mở cho cộng đồng LA | Phát hiện sớm sinh viên at-risk | Phân tích quy trình học tập | Xây dựng hệ thống cảnh báo sớm | **Phân tích hành vi + Dự đoán sớm + Giải thích mô hình (XAI)** |
| **Nhóm đặc trưng** | 4 chiều: Hành vi, Nhận thức, Cảm xúc, Xã hội | Clicks, nhân khẩu học, điểm số, đăng ký | Clicks tích lũy, điểm bài tập theo tuần | Chuỗi sự kiện, thời gian chuyển đổi trạng thái | Clicks, điểm số, nhân khẩu học | **Đa chiều SEDL (30+ đặc trưng tương tác + nhân khẩu học)** |
| **Phương pháp / Mô hình** | Định tính & Phân loại định lượng | Thống kê mô tả | Moving average, Time-series classifiers | Process Mining, Decision Trees | Logistic Regression, Decision Tree, SVM | **Baseline (LogReg, RF) + XGBoost + LightGBM + SHAP** |
| **Mốc thời gian dự đoán** | Không áp dụng | Cuối khóa | Tuần 2 đến tuần 10 | Toàn bộ khóa | Sau 2-4 tuần học | **3 mốc: 25%, 50%, 75% thời lượng khóa học** |
| **Chỉ số đánh giá** | Số lượng trích dẫn, xu hướng | Phân bố, missing data | Recall, Precision, AUC | F-measure, Accuracy | Recall, Sensitivity, Specificity | **Recall, F1-Score, PR-AUC, ROC-AUC** |
| **Giải thích mô hình (XAI)** | Không | Không | Gián tiếp qua feature importance cơ bản | Biểu đồ luồng quy trình | Cây quyết định trực quan | **SHAP values (Global + Local explanation cho từng SV)** |

---

## 4. KHOẢNG TRỐNG NGHIÊN CỨU VÀ ĐÓNG GÓP CỦA DỰ ÁN

1. **Vận hành hóa đa chiều**: Đa phần các nghiên cứu OULAD trước đây chỉ cộng dồn tổng số click (`sum_click`). Dự án sẽ phân rã tương tác theo từng loại `activity_type` và thời điểm để phản ánh chiều nhận thức (Cognitive) và xã hội (Social) theo Bergdahl et al. (2024).
2. **Chống Data Leakage thực sự**: Áp dụng phân chia dữ liệu theo khóa học/kỳ học (Group Split / Time-based Split), loại bỏ toàn bộ thông tin tương tác sau mốc cutoff thời gian ($T_{25\%}, T_{50\%}, T_{75\%}$).
3. **Minh bạch hóa quyết định (Explainability)**: Tích hợp SHAP để giải thích lý do tại sao một sinh viên bị gán nhãn "Nguy cơ", hỗ trợ giảng viên đưa ra hành động can thiệp trúng đích thay vì một kết quả "hộp đen".
