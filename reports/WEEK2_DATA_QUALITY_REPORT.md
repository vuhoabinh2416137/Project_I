# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU OULAD (DATA QUALITY REPORT)
## Tuần 2: Thu thập và Tìm hiểu Dữ liệu
## Đề tài: LMS Tiên tiến và Learning Analytics

---

## 1. TỔNG QUAN NẠP DỮ LIỆU

Bộ dữ liệu **Open University Learning Analytics Dataset (OULAD)** đã được tải chính thức từ Figshare và giải nén hoàn chỉnh vào thư mục `data/raw/` của dự án với tổng dung lượng giải nén khoảng **442.5 MB**.

### Danh mục 7 file dữ liệu CSV:
| Tên file | Kích thước trên đĩa | Số dòng (Records) | Số cột (Features) | Nội dung nghiệp vụ |
|---|:---:|:---:|:---:|---|
| **`courses.csv`** | 0.001 MB | 22 | 3 | Danh mục 22 khóa học (7 modules x 4 kỳ học) |
| **`assessments.csv`** | 0.01 MB | 206 | 6 | Thông tin cấu trúc bài kiểm tra (TMA, CMA, Exam) |
| **`studentInfo.csv`** | 3.30 MB | 32,593 | 12 | Thông tin nhân khẩu học và kết quả cuối kỳ |
| **`studentRegistration.csv`** | 1.06 MB | 32,593 | 5 | Nhật ký ngày đăng ký và ngày hủy môn |
| **`studentAssessment.csv`** | 5.43 MB | 173,912 | 5 | Điểm số và ngày nộp bài của sinh viên |
| **`vle.csv`** | 0.25 MB | 6,364 | 6 | Danh mục tài nguyên trực tuyến trên VLE Moodle |
| **`studentVle.csv`** | 432.81 MB | 10,655,280 | 6 | Log chi tiết tương tác (click) theo từng ngày |

---

## 2. BẢNG KIỂM TRA CHẤT LƯỢNG VÀ MISSING VALUES (DATA AUDIT)

Qua kiểm tra tự động trong [notebooks/01_data_loading.ipynb](file:///c:/Users/mrore/Project_I/notebooks/01_data_loading.ipynb):

| Bảng dữ liệu | Số dòng | Số cột | Số ô bị Missing | Tỷ lệ Missing (%) | Trùng lặp (Duplicates) | Bộ nhớ RAM (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `courses` | 22 | 3 | 0 | 0.00% | 0 | 0.00 MB |
| `assessments` | 206 | 6 | 11 | 0.89% | 0 | 0.01 MB |
| `studentInfo` | 32,593 | 12 | 1,111 | 0.28% | 0 | 4.85 MB |
| `studentRegistration` | 32,593 | 5 | 22,566 | 13.85% | 0 | 1.49 MB |
| `studentAssessment` | 173,912 | 5 | 173 | 0.02% | 0 | 6.63 MB |
| `vle` | 6,364 | 6 | 10,486 | 27.46% | 0 | 0.38 MB |
| `studentVle` | 10,655,280 | 6 | 0 | 0.00% | 787,170* | 569.05 MB |

> *\*Ghi chú về trùng lặp trong `studentVle`*: Trong bảng log `studentVle`, có các dòng ghi nhận nhiều phiên click trong cùng một ngày cho cùng một tài nguyên. Khi tổng hợp (aggregation) theo sinh viên và ngày, các dòng này sẽ được cộng dồn tự nhiên (`sum_click`).

### Chi tiết các cột có giá trị khuyết thiếu (Missing Values Analysis):
1. **`assessments.date`** (11 dòng missing ~ 5.3%):
   - *Bản chất*: Toàn bộ 11 bài bị thiếu ngày hạn nộp đều là các bài thi kết thúc môn (`assessment_type == 'Exam'`), do ngày thi cuối kỳ được trường công bố riêng tùy theo lịch thi tập trung.
2. **`studentInfo.imd_band`** (1,111 dòng missing ~ 3.4%):
   - *Bản chất*: Chỉ số điều kiện kinh tế xã hội vùng bị thiếu do sinh viên không khai báo địa chỉ hoặc ở ngoài vùng điều tra nước Anh. Cần xử lý bằng cách gán nhãn `Missing` hoặc mode.
3. **`studentRegistration.date_unregistration`** (22,521 dòng missing ~ 69.1%):
   - *Bản chất*: Đây là **Missing có ý nghĩa nghiệp vụ (Structural Missing)**. Sinh viên không rút môn sẽ không có ngày rút. Giá trị không missing tương ứng với sinh viên có `final_result == 'Withdrawn'`.
   - *CẢNH BÁO NGUY HIỂM*: Không được dùng cột này làm đặc trưng dự đoán để tránh Data Leakage!
4. **`studentAssessment.score`** (173 dòng missing ~ 0.1%):
   - *Bản chất*: Sinh viên nộp bài nhưng chưa được chấm hoặc có sự cố chấm bài. Có thể điền 0 hoặc trung bình điểm của môn.
5. **`vle.week_from` & `vle.week_to`** (5,243 dòng missing ~ 82.4%):
   - *Bản chất*: Tài nguyên mở tự do trong suốt khóa học thay vì bị giới hạn trong một tuần học cụ thể.

---

## 3. KIỂM TRA TÍNH TOÀN VẸN QUAN HỆ (REFERENTIAL INTEGRITY)

Kết quả kiểm tra liên kết khóa ngoại giữa 7 bảng:
- Sinh viên trong `studentVle` không có trong `studentInfo`: **0 (100% khớp)**
- Sinh viên trong `studentAssessment` không có trong `studentInfo`: **0 (100% khớp)**
- Mã bài thi trong `studentAssessment` không có trong `assessments`: **0 (100% khớp)**
- Mã tài nguyên `id_site` trong `studentVle` không có trong `vle`: **0 (100% khớp)**

=> **Kết luận**: Dữ liệu có tính toàn vẹn quan hệ hoàn hảo, không có bản ghi mồ côi (orphan records).

---

## 4. PHÂN TÍCH PHÂN BỐ BIẾN MỤC TIÊU (FINAL_RESULT)

Biểu đồ đã được xuất bản tại: [reports/figures/01_target_distribution.png](file:///c:/Users/mrore/Project_I/reports/figures/01_target_distribution.png)

```
Phân bố 4 nhóm gốc:
- Pass:        12,361 (37.9%)
- Withdrawn:   10,156 (31.2%)
- Fail:         7,052 (21.6%)
- Distinction:  3,024 ( 9.3%)

Ánh xạ bài toán Nhị phân:
- 0: Not At-Risk (Pass + Distinction) = 15,385 SV (47.2%)
- 1: At-Risk     (Fail + Withdrawn)   = 17,208 SV (52.8%)
```
> **Nhận định**: Phân loại nhị phân đạt tỷ lệ cân bằng lý tưởng (~47% vs ~53%), không bị mất cân bằng trầm trọng như các bài toán gian lận hay phát hiện bệnh hiếm. Tuy nhiên, hành vi giữa nhóm `Withdrawn` (dừng tương tác sớm) và `Fail` (học hết nhưng thi trượt) có sự phân hóa lớn, sẽ được khai thác sâu ở Tuần 3 (EDA).

---

## 5. PHÂN TÍCH TƯƠNG TÁC TÀI NGUYÊN LMS (VLE ACTIVITIES)

Biểu đồ đã được xuất bản tại: [reports/figures/03_activity_type_clicks.png](file:///c:/Users/mrore/Project_I/reports/figures/03_activity_type_clicks.png)

Top các tài nguyên có lượt click cao nhất:
1. **`oucontent`**: Chiếm tỷ trọng lớn nhất (~50% tổng số click) - Tài liệu bài giảng chính thống.
2. **`forumng`**: Chiếm ~25% tổng số click - Diễn đàn trao đổi thảo luận sinh viên.
3. **`homepage`**: Chiếm ~12% - Trang chủ truy cập hàng ngày.
4. **`quiz`**: Chiếm ~8% - Bài tập trắc nghiệm tự luyện.
5. **`subpage`, `resource`, `url`**: Chiếm phần còn lại.

=> Phát hiện này hoàn toàn khớp với Khung SEDL (Bergdahl et al., 2024): ta có thể tách riêng tỷ lệ click `oucontent` (Cognitive), `forumng` (Social) và `quiz` (Cognitive Practice) để làm giàu bộ đặc trưng.

---

## 6. DANH MỤC SẢN PHẨM BÀN GIAO TUẦN 2 (DELIVERABLES CHECKLIST)

- [x] Lưu trữ 7 file CSV OULAD tại `data/raw/`
- [x] Tạo Jupyter Notebook nạp và kiểm tra dữ liệu toàn diện: [01_data_loading.ipynb](file:///c:/Users/mrore/Project_I/notebooks/01_data_loading.ipynb)
- [x] Bảng thống kê chi tiết chất lượng và missing data trong [01_data_loading.ipynb](file:///c:/Users/mrore/Project_I/notebooks/01_data_loading.ipynb)
- [x] Xuất bản 3 biểu đồ phân bố và nhân khẩu học:
  - [01_target_distribution.png](file:///c:/Users/mrore/Project_I/reports/figures/01_target_distribution.png)
  - [02_demographics_overview.png](file:///c:/Users/mrore/Project_I/reports/figures/02_demographics_overview.png)
  - [03_activity_type_clicks.png](file:///c:/Users/mrore/Project_I/reports/figures/03_activity_type_clicks.png)
- [x] Báo cáo đánh giá chất lượng dữ liệu: [WEEK2_DATA_QUALITY_REPORT.md](file:///c:/Users/mrore/Project_I/reports/WEEK2_DATA_QUALITY_REPORT.md)

---

## 7. KẾ HOẠCH BỔ SUNG CHO TUẦN 3
- Triển khai phân tích khám phá dữ liệu (EDA) theo khung lý thuyết SEDL.
- **Nhiệm vụ trọng tâm bổ sung**: Chạy thử mô hình của bài báo dự đoán cũ (baseline model), thực hiện chuyển đổi cấu hình nhãn từ **4 sample** (`Distinction`, `Pass`, `Fail`, `Withdrawn`) sang **2 sample** (`0: Not At-Risk` vs `1: At-Risk`) đối với bài toán cảnh báo sớm hiện tại để thiết lập ngưỡng chuẩn (benchmark) ban đầu.

