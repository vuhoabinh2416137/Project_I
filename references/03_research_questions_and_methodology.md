# CÂU HỎI NGHIÊN CỨU VÀ PHƯƠNG PHÁP LUẬN (METHODOLOGY)
## Đề tài: LMS Tiên tiến và Learning Analytics - Dự đoán Nguy cơ Học tập của Sinh viên

---

## 1. HỆ THỐNG CÂU HỎI NGHIÊN CỨU VÀ GIẢ THUYẾT

### Câu hỏi nghiên cứu 1 (RQ1): Mối tương quan giữa tương tác LMS và kết quả học tập
> **RQ1**: *Các chỉ số tương tác nào trên LMS (số lượt click, tính chuyên cần, hành vi nộp bài) có mối tương quan và khả năng phân biệt mạnh nhất đối với kết quả học tập cuối kỳ của sinh viên?*

- **Giả thuyết H1a (Hành vi)**: Sinh viên đạt kết quả cao (Pass/Distinction) có tổng số lượt click (`total_clicks`) và số ngày hoạt động (`active_days`) cao hơn có ý nghĩa thống kê ($p < 0.001$) so với nhóm có nguy cơ (Fail/Withdrawn).
- **Giả thuyết H1b (Nhận thức)**: Sinh viên chủ động chuẩn bị bài trước khai giảng (`pre_course_clicks > 0`) và nộp bài trước hạn (`submission_delay < 0`) có xác suất vượt qua môn học cao hơn đáng kể.
- **Giả thuyết H1c (Xã hội)**: Mức độ tương tác trên diễn đàn (`forumng_clicks`) tỷ lệ thuận với khả năng duy trì khóa học, giúp giảm thiểu tỷ lệ rút lui (Withdrawn).

---

### Câu hỏi nghiên cứu 2 (RQ2): Khả năng cảnh báo sớm (Early At-Risk Prediction)
> **RQ2**: *Mô hình học máy có thể phát hiện sinh viên có nguy cơ gặp khó khăn sớm tới mức độ nào (trước 25%, 50%, 75% thời lượng khóa học) mà vẫn đảm bảo độ tin cậy thực tiễn ($\text{Recall} \ge 75\%$, $\text{ROC-AUC} \ge 0.80$, $\text{F1-Score} \ge 0.70$)?*

- **Cơ sở thiết lập 3 tiêu chuẩn nghiệm thu RQ2**:
  - **$\text{ROC-AUC} \ge 0.80$**: Kế thừa trực tiếp ngưỡng phân tách mức *tốt/xuất sắc* (*excellent discrimination*: $0.80 \le \text{AUC} < 0.90$) theo Hosmer, Lemeshow & Sturdivant (2013, tr. 177).
  - **$\text{Recall} \ge 75\%$**: Chỉ tiêu thực tiễn do đề tài xác lập, neo theo vùng hiệu năng độ nhạy cảnh báo sớm $74\%$–$80\%$ tại mốc $25\%$ học kỳ của hệ thống OAAI (Jayaprakash et al., 2014) và đặt cao hơn rõ rệt mức đối chuẩn sớm trên OULAD của Adnan et al. (2021) ($\text{Recall} \approx 59\%$ ở mốc $20\%$).
  - **$\text{F1-Score} \ge 0.70$**: Ràng buộc cân bằng của đề tài nhằm ngăn mô hình hạ ngưỡng quá mức gây quá tải cảnh báo giả (*alert fatigue*); về mặt toán học, tại $\text{Recall} = 0.75$, điều kiện $F_1 \ge 0.70$ đòi hỏi $\text{Precision} \ge 0.656$, đồng thời vượt mức $F_1 \approx 0.59$ của Adnan et al. (2021).

- **Giả thuyết H2a**: Tại mốc 25% thời lượng khóa học (~tuần 6-8), các mô hình dựa trên gradient boosting (XGBoost / LightGBM) có thể phát hiện ít nhất 70% sinh viên có nguy cơ bỏ cuộc sớm (Withdrawn).
- **Giả thuyết H2b**: Khi bổ sung thêm dữ liệu điểm số bài tập đầu tiên tại mốc 50% thời lượng, hiệu năng dự đoán (F1-score và PR-AUC) sẽ tăng vọt so với chỉ sử dụng dữ liệu click thuần túy.

---

### Câu hỏi nghiên cứu 3 (RQ3): Giải thích mô hình và các yếu tố quyết định (Explainability & XAI)
> **RQ3**: *Những đặc trưng nào đóng vai trò then chốt nhất trong việc dẫn dắt dự đoán của mô hình, và SHAP (SHapley Additive exPlanations) có thể cung cấp gợi ý can thiệp sư phạm khả thi cho từng sinh viên cụ thể như thế nào?*

- **Giả thuyết H3a**: Đặc trưng nỗ lực tương tác theo thời gian gần nhất (recency of interaction và inactivity gap) có giá trị SHAP vượt trội hơn các yếu tố nhân khẩu học tĩnh (tuổi, giới tính, vùng miền).
- **Giả thuyết H3b**: Phân tích SHAP waterfall plot cho từng cá nhân có thể chỉ ra chính xác điểm nghẽn của từng sinh viên (ví dụ: điểm bài 1 thấp kết hợp ngừng truy cập diễn đàn 2 tuần liền).

---

## 2. VẬN HÀNH HÓA ĐẶC TRƯNG THEO KHUNG SEDL (Bergdahl et al., 2024)

Để giải quyết khoảng trống 95% nghiên cứu chỉ đo lường hành vi đơn giản, dự án phân tách dữ liệu thô OULAD thành 4 nhóm đặc trưng:

```
                      ┌────────────────────────────────────────┐
                      │     DỮ LIỆU GỐC OULAD (7 CSVs)        │
                      └──────────────────┬─────────────────────┘
                                         │
        ┌──────────────────┬─────────────┴───────┬──────────────────┐
        ▼                  ▼                     ▼                  ▼
┌──────────────┐   ┌──────────────┐      ┌──────────────┐   ┌──────────────┐
│  BEHAVIOURAL │   │  COGNITIVE   │      │    SOCIAL    │   │ DEMOGRAPHIC  │
│  ENGAGEMENT  │   │  ENGAGEMENT  │      │  ENGAGEMENT  │   │  & CONTEXT   │
├──────────────┤   ├──────────────┤      ├──────────────┤   ├──────────────┤
│- Total clicks│   │- Early submit│      │- Forum clicks│   │- Highest edu │
│- Active days │   │- Content ratio│     │- Forum days  │   │- IMD band    │
│- Clicks/day  │   │- Quiz clicks │      │- Wiki clicks │   │- Age band    │
│- Pre-clicks  │   │- Regularity  │      │- Collab score│   │- Prev attempt│
│- Last active │   │- Mean score  │      │              │   │- Credits     │
└──────────────┘   └──────────────┘      └──────────────┘   └──────────────┘
```

### Bảng định nghĩa đặc trưng chi tiết:

| Nhóm đặc trưng | Tên đặc trưng | Cách tính toán từ OULAD | Ý nghĩa sư phạm |
|---|---|---|---|
| **Behavioural** | `total_clicks` | $\sum \text{sum\_click}$ trong khoảng thời gian quan sát | Tổng khối lượng nỗ lực học tập |
| **Behavioural** | `active_days` | Số ngày duy nhất có tương tác ($\text{count}(\text{distinct date})$) | Độ kiên trì và thói quen học tập |
| **Behavioural** | `clicks_per_active_day` | $\text{total\_clicks} / \text{active\_days}$ | Cường độ tương tác trong mỗi phiên học |
| **Behavioural** | `pre_course_clicks` | Clicks với $\text{date} < 0$ | Sự sẵn sàng và tính chủ động trước khóa học |
| **Behavioural** | `days_since_last_activity` | Ngày cutoff trừ đi ngày tương tác cuối cùng | Nguy cơ "rơi rụng" (disengagement drop-off) |
| **Cognitive** | `avg_submission_delay` | Trung bình $(\text{date\_submitted} - \text{assessment.date})$ | Thói quen nộp bài sớm hay trì hoãn (procrastination) |
| **Cognitive** | `oucontent_ratio` | $\text{clicks}_{\text{oucontent}} / \text{total\_clicks}$ | Mức độ chú trọng đọc tài liệu lý thuyết cốt lõi |
| **Cognitive** | `quiz_attempt_ratio` | $\text{clicks}_{\text{quiz}} / \text{total\_clicks}$ | Nỗ lực tự đánh giá kiến thức |
| **Cognitive** | `study_regularity_std` | Độ lệch chuẩn số click giữa các tuần | Tính ổn định hay học dồn (cramming) |
| **Cognitive** | `avg_assessment_score` | Trung bình trọng số điểm các bài đã nộp trước cutoff | Năng lực học tập hiện tại |
| **Social** | `forum_clicks` | Clicks vào tài nguyên có `activity_type == 'forumng'` | Tương tác trao đổi trên cộng đồng môn học |
| **Social** | `forum_ratio` | $\text{clicks}_{\text{forumng}} / \text{total\_clicks}$ | Tỷ trọng giao tiếp xã hội trong hành vi học |
| **Social** | `wiki_clicks` | Clicks vào tài nguyên có `activity_type == 'ouwiki'` | Sự tham gia vào các hoạt động học tập hợp tác |
| **Demographic** | `studied_credits` | Giá trị nguyên từ `studentInfo` | Áp lực tải học tập của sinh viên |
| **Demographic** | `num_of_prev_attempts`| Số lần học lại môn từ `studentInfo` | Tiền sử gặp khó khăn học thuật |
| **Demographic** | `highest_education_enc`| Mã hóa thứ bậc trình độ học vấn | Nền tảng học vấn đầu vào |
| **Demographic** | `imd_band_enc` | Mã hóa mức độ điều kiện kinh tế xã hội vùng | Yếu tố ngoại cảnh tác động |

---

## 3. THIẾT KẾ THỰC NGHIỆM VÀ PHÒNG CHỐNG DATA LEAKAGE

### 3.1. Thiết lập các mốc thời gian Early Prediction
Đối với mỗi khóa học có độ dài $L$ ngày (từ `courses.module_presentation_length`):
- **Mốc 25% ($T_{25\%} = 0.25 \times L$)**: Thường rơi vào khoảng ngày 60. Tại mốc này sinh viên thường chỉ mới nộp 0 hoặc 1 bài TMA đầu tiên.
- **Mốc 50% ($T_{50\%} = 0.50 \times L$)**: Khoảng ngày 120. Đã có 2-3 bài kiểm tra, xu hướng tương tác đã bộc lộ rõ.
- **Mốc 75% ($T_{75\%} = 0.75 \times L$)**: Khoảng ngày 180. Chuẩn bị bước vào giai đoạn ôn thi cuối kỳ.

> **Quy tắc vàng chống Data Leakage**: Khi huấn luyện và đánh giá mô hình tại mốc $T$:
> 1. Toàn bộ bản ghi log trong `studentVle` có $\text{date} > T$ **bắt buộc phải bị loại bỏ hoàn toàn**.
> 2. Toàn bộ bài nộp trong `studentAssessment` có $\text{date\_submitted} > T$ **bắt buộc không được đưa vào tính điểm**.
> 3. Tuyệt đối không dùng thông tin `date_unregistration` làm đặc trưng đầu vào (vì nếu sinh viên có ngày hủy môn trước cuối khóa thì 100% là Withdrawn, gây rò rỉ nhãn nghiêm trọng).

### 3.2. Chiến lược chia dữ liệu (Validation Strategy)
- Không dùng `Random Split` thông thường trên toàn bộ sinh viên, vì sinh viên có thể học cùng một khóa dẫn đến tương quan chéo.
- Áp dụng **Group Split** theo kỳ học hoặc module:
  - Huấn luyện trên các kỳ học trước (ví dụ: `2013B`, `2013J`, `2014B`).
  - Đánh giá kiểm thử ngoại suy (Out-of-Time Test Set) trên kỳ cuối cùng (`2014J`).
  - Cách chia này phản ánh chính xác bài toán thực tế: Dùng mô hình học từ các năm học cũ để cảnh báo sinh viên cho năm học mới.

---

## 4. BỘ CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Trong bài toán giáo dục cảnh báo nguy cơ:
- $TP$ (True Positive): Sinh viên at-risk và được mô hình dự đoán đúng là at-risk.
- $FN$ (False Negative): Sinh viên at-risk nhưng mô hình bỏ sót (nguy cơ bị trượt hoặc bỏ học mà không được trợ giúp).
- $FP$ (False Positive): Sinh viên bình thường nhưng bị cảnh báo nhầm (giảng viên tốn công liên hệ nhưng không gây hại lớn).
- $TN$ (True Negative): Sinh viên bình thường và mô hình dự đoán bình thường.

Do tính chất can thiệp giáo dục, chúng ta ưu tiên **Recall** của lớp At-Risk:

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Ngoài ra, sử dụng **PR-AUC (Precision-Recall Area Under Curve)** và **ROC-AUC** để đánh giá toàn diện khả năng xếp hạng rủi ro (risk ranking) của mô hình độc lập với ngưỡng phân loại (threshold).
