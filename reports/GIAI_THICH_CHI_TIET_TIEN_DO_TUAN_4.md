# CẨM NANG TOÀN DIỆN VỀ TIỀN XỬ LÝ DỮ LIỆU TUẦN 4
## Giải thích chi tiết bản chất kỹ thuật theo góc nhìn trực quan, dễ hiểu
**Dự án**: LMS Tiên tiến và Learning Analytics (Phân tích dữ liệu học tập OULAD)  
**Tác giả**: Antigravity Assistant & Vũ Hoà Bình  
**Đối tượng phục vụ**: Bất kỳ ai muốn hiểu cội nguồn bản chất của dữ liệu học máy mà không cần có sẵn kiến thức chuyên sâu về Data Engineering.

---

## 🧭 Lời mở đầu: Tiền xử lý dữ liệu thực chất là gì?

Hãy tưởng tượng bạn là một **đầu bếp chuẩn bị nấu một bữa tiệc lớn** (huấn luyện mô hình Trí tuệ Nhân tạo - AI):
- Dữ liệu thô thu thập từ hệ thống LMS giống như **nguyên liệu vừa mới mua từ chợ về**: có củ khoai còn dính bùn đất (dữ liệu thiếu/lỗi), có mớ rau lẫn sỏi đá (giá trị ngoại lai bất thường), và có những bao bì nhãn mác bằng tiếng nước ngoài mà máy móc không đọc được (dữ liệu chữ/danh mục).
- Nếu bạn cứ thế ném toàn bộ nguyên liệu bẩn vào nồi nấu ngay, món ăn (mô hình AI) chắc chắn sẽ bị hỏng, đưa ra những dự đoán sai lệch hoặc "học vẹt".
- **Tiền xử lý dữ liệu (Data Preprocessing)** chính là toàn bộ công đoạn nhặt rau, rửa sạch, gọt vỏ, thái hạt lựu đồng đều và chia khẩu phần sẵn sàng trước khi bật bếp.

Trong Tuần 4, chúng ta đã giải quyết **7 bài toán kỹ thuật lớn**. Dưới đây là câu chuyện chi tiết đằng sau từng bước:

---

## 1. Xử lý dữ liệu bị khuyết thiếu (Missing Values)

### ❓ Vấn đề thực tế:
Khi nhìn vào bảng dữ liệu gồm **$32,593$ dòng**, chúng ta thấy rất nhiều ô trống không có dữ liệu (gọi là `NaN` hoặc `Null`):
- Có cột thiếu tới $69\%$ dữ liệu (`date_unregistration` - ngày hủy môn).
- Có cột thiếu $25\%$ (`tma1_score` - điểm bài tập đầu tiên).
- Có cột thiếu điểm trung bình (`avg_score`).

### 💡 Tại sao không đơn giản là "xóa hết các dòng bị thiếu"?
Một phản xạ tự nhiên của người mới làm dữ liệu là: *"Cứ dòng nào có ô trống thì xóa béng đi cho sạch!"*.
**Đó là một sai lầm chết người!**
- Trong giáo dục, sinh viên **không có điểm kiểm tra** thường chính là những bạn **bỏ thi, lười học hoặc đã bỏ cuộc giữa chừng**.
- Mục tiêu của dự án là **phát hiện sinh viên có nguy cơ rớt môn**. Nếu bạn xóa hết những bạn không nộp bài đi, bạn đã vô tình xóa sạch nhóm người quan trọng nhất mà mô hình cần học để bảo vệ!

### 🛠️ Chúng ta đã xử lý thế nào?
1. **Điểm số bài tập (`avg_score`, `tma1_score`)**: 
   - *Cách làm*: Điền giá trị **$0.0$ điểm**. 
   - *Lý do*: Trong thực tế học đường, không đi thi hay không nộp bài đồng nghĩa với việc bạn nhận điểm 0. Đồng thời tạo thêm một cột đánh dấu `has_no_submissions = 1` (đã bỏ thi) để AI chú ý.
2. **Số ngày nộp bài trước hạn (`avg_days_early`)**:
   - *Cách làm*: Điền giá trị phạt danh nghĩa là **$-10.0$ ngày**.
   - *Lý do*: Những bạn nộp đúng hạn sẽ có số ngày dương (ví dụ $+5$ ngày là nộp sớm 5 ngày). Người không nộp bài xem như trễ hạn rất nặng, nên ta gán số âm $-10$ để biểu thị mức phạt.
3. **Mức độ nghèo khó của vùng sinh sống (`imd_band`)**:
   - *Cách làm*: Lấy giá trị xuất hiện nhiều nhất trong lớp học (giá trị `Mode` - ở đây là nhóm `20-30%`) điền vào, kết hợp gắn cờ `imd_was_missing = 1`.
   - *Lý do*: Thông tin nhân khẩu học này không làm thay đổi hoàn toàn năng lực học tập, điền giá trị phổ biến giúp giữ lại dòng dữ liệu mà không làm sai lệch phân phối.

---

## 2. Bẫy "Rò rỉ Dữ liệu Tương lai" (Target Leakage)

### ❓ Vấn đề: Cột `date_unregistration` thiếu tới $69.10\%$, tại sao tuyệt đối KHÔNG ĐƯỢC dùng nó để dự đoán?
- Cột `date_unregistration` ghi lại ngày mà một sinh viên lên văn phòng làm đơn xin rút khỏi môn học.
- Nếu một sinh viên **không rút môn** (học đến cuối kỳ), cột này sẽ bị bỏ trống (`NaN`).

### ⚠️ Nguy cơ "Học vẹt" (Target Leakage):
- Giả sử bạn đi thi đại học, nhưng giám thị lại vô tình để lộ đáp án của bài thi ngay bên cạnh đề bài. Bạn sẽ được điểm 10 tuyệt đối dễ dàng, nhưng khi bước ra đời thật gặp bài toán mới, bạn sẽ không biết làm gì cả.
- Nếu chúng ta đưa cột `date_unregistration` vào ma trận dự đoán, mô hình AI sẽ nhanh chóng phát hiện ra mẹo lười biếng: *"Cứ ai có ngày tháng ở cột này thì đích thị là đã bỏ học, ai không có ngày tháng thì là học xong!"*.
- Khi đó, AI đạt độ chính xác giả tạo $100\%$. Nhưng khi đưa vào thực tế (đầu kỳ học, sinh viên chưa rút môn), mô hình sẽ hoàn toàn mù tịt vì thông tin tương lai này chưa hề xảy ra!
- **Giải pháp**: Chúng ta **loại bỏ hoàn toàn** cột `date_unregistration` khỏi danh sách đặc trưng đầu vào để mô hình buộc phải học từ hành vi thực tế (số lần click, số bài nộp).

---

## 3. Câu chuyện về những sinh viên "Rút lui sớm" (Early Withdrawn)

### ❓ Phát hiện kỳ lạ từ dữ liệu:
Khi lọc dữ liệu, ta thấy có **$2,678$ lượt đăng ký** xin rút môn trước ngày khai giảng chính thức (`date_unregistration < 0`):
- Trong đó có tới **$2,213$ bạn hoàn toàn có 0 lượt click** trên hệ thống LMS.
- Tức là họ đăng ký môn học xong, nhưng trước khi lớp bắt đầu thì họ đã đổi ý, hủy môn và chưa từng mở trang web khóa học ra xem dù chỉ một lần.

### 💡 Tại sao phải phân tách nhóm này?
- Nếu bạn muốn làm bài toán **Dự báo học tập thời gian thực (Learning Analytics)**: Mục đích là giúp đỡ những sinh viên *đang ngồi học trong lớp* nhưng có dấu hiệu đuối sức. Việc giữ lại $2,213$ bạn "chưa từng bước chân vào cửa lớp" sẽ khiến mô hình bị nhiễu.
- Chúng ta đã xây dựng cơ chế **linh hoạt 2 chế độ**:
  - *Chế độ 1 (Full Cohort - Mặc định $32,593$ sinh viên)*: Phục vụ quản trị tuyển sinh toàn trường.
  - *Chế độ 2 (Active Cohort - $30,380$ sinh viên)*: Loại bỏ những trường hợp hủy môn trước ngày 0 để mô hình học máy tập trung phân tích hành vi của người học thực thụ.

---

## 4. Xử lý Ngoại lai (Outliers): Đừng giết nhầm "Học sinh giỏi"!

### ❓ Vấn đề thực tế:
- Đa số sinh viên trong kỳ học click khoảng $500$ đến $1,500$ lần.
- Tuy nhiên, có những sinh viên "quái vật" click tới **$24,139$ lần**, hoặc có người đăng ký tới **$655$ tín chỉ** (gấp 5 lần bình thường!).
- Đây gọi là các **giá trị ngoại lai (outliers)**.

### 💡 Tại sao không xóa những dòng này đi?
Trong khoa học dữ liệu truyền thống (ví dụ đo nhiệt độ sensor bị hỏng, bắn ra số $10,000^\circ\text{C}$), ta xóa đi vì đó là lỗi cảm biến.
Nhưng trong trường học:
- Một sinh viên click $20,000$ lần không phải là do máy hỏng, mà đó là **học sinh siêu chăm chỉ (Super-active learners)**.
- Khi đối chiếu với kết quả cuối khóa, hầu hết các bạn này đều đạt loại **Xuất sắc (Distinction)**.
- Nếu bạn xóa họ đi, mô hình của bạn sẽ không bao giờ học được hình mẫu của một người đạt điểm Giỏi!

### 🛠️ Kỹ thuật chúng ta đã dùng: Chặn trần (Capping) & Biến đổi hàm Log
1. **Chặn trần (Winsorization Capping)**: Thay vì để bạn ấy ở mức $24,139$ clicks (quá xa so với các bạn khác, làm méo mó thước đo), ta tính ngưỡng trần hợp lý dựa trên công thức $Q_3 + 3 \times \text{IQR} \approx 5,200$ clicks. Bất kỳ ai click trên $5,200$ lần đều được "ghìm" lại ở mức $5,200$. Bạn ấy vẫn giữ được danh hiệu người click nhiều nhất, nhưng không làm lệch thước đo của cả lớp.
2. **Biến đổi hàm Log ($x' = \ln(1 + x)$)**: Giúp nén khoảng cách giữa người $10$ clicks và người $5,000$ clicks lại theo cấp số nhân, biến một biểu đồ bị "lệch hẳn về một bên" trở thành hình quả chuông cân đối (phân phối chuẩn), giúp AI tính toán nhẹ nhàng hơn rất nhiều.

---

## 5. Mã hóa dữ liệu (Encoding): Biến chữ thành số

### ❓ Tại sao phải mã hóa?
Máy tính và các phương trình toán học trong AI chỉ có thể cộng, trừ, nhân, chia các **con số**. Chúng không thể hiểu được chữ `"Nam"`, `"Nữ"`, `"Scotland"` hay `"Bằng Cử nhân"`.

### 🛠️ Hai cách biến chữ thành số:
1. **Mã hóa có thứ bậc (Ordinal Encoding)**:
   - Dùng cho những thông tin có trật tự cao thấp rõ ràng.
   - Ví dụ trình độ học vấn: *Chưa có bằng cấp ($0$) $\rightarrow$ Dưới cấp ba ($1$) $\rightarrow$ Cấp ba ($2$) $\rightarrow$ Đại học ($3$) $\rightarrow$ Sau đại học ($4$)*. Số càng to thể hiện trình độ càng cao.
2. **Mã hóa One-Hot (One-Hot Encoding)**:
   - Dùng cho những thông tin không có thứ bậc hơn kém (như giới tính, vùng miền, mã môn học).
   - Nếu bạn gán *Hà Nội = 1, Đà Nẵng = 2, TP.HCM = 3*, máy tính sẽ lầm tưởng rằng *TP.HCM gấp 3 lần Hà Nội*, hoặc *Đà Nẵng nằm ở giữa Hà Nội và TP.HCM về mặt giá trị!*
   - Để tránh điều này, ta tách ra thành các cột đèn bật/tắt (0 hoặc 1): cột `region_London` (1 nếu ở London, 0 nếu không phải), cột `gender_M` (1 nếu là Nam, 0 nếu là Nữ).

---

## 6. Chuẩn hóa dữ liệu số (Scaling): Đưa về cùng một hệ quy chiếu

### ❓ Thước đo không đồng nhất:
- Cột `total_clicks` có giá trị từ $0$ đến $5,000$.
- Cột `avg_score` có giá trị từ $0$ đến $100$.
- Cột `num_of_prev_attempts` (số lần học lại) chỉ từ $0$ đến $3$.

Nếu bạn đưa thẳng các con số này vào mô hình AI:
- Thuật toán sẽ nghĩ rằng: *"Ôi, con số $5,000$ to quá, chắc nó quan trọng gấp hàng nghìn lần con số 3!"*.
- Trong khi thực tế, việc một sinh viên đã từng **học lại 3 lần** có thể là dấu hiệu cảnh báo nguy hiểm hơn rất nhiều so với việc click ít đi vài cái.

### 🛠️ Giải pháp: `RobustScaler`
Chúng ta dùng công thức trừ đi Trung vị (Median) và chia cho Khoảng tứ phân vị (IQR):
$$z = \frac{x - \text{Median}}{\text{IQR}}$$
Tất cả các biến (dù to hàng nghìn hay nhỏ từ 0 đến 1) đều được co giãn về một khoảng số học tương đồng (quanh mức 0). `RobustScaler` được chọn vì nó sử dụng Trung vị nên **không bị ảnh hưởng bởi những sinh viên cá biệt click quá nhiều**.

### 🔒 Quy tắc thép: Zero Data Leakage trong Scaling
- Khi đi thi, học sinh không được biết trước đề thi.
- Thước đo (Median, IQR, Mean) **CHỈ ĐƯỢC PHÉP TÍNH TRÊN TẬP HỌC (Train set)**.
- Sau đó, lấy nguyên cây thước đó mang sang đo tập Validation và tập Test. Tuyệt đối không được gộp chung cả tập thi vào để tính thước đo!

---

## 7. Chia tập dữ liệu (70/15/15) & Rò rỉ Nhóm sinh viên (Group Leakage)

### ❓ Chia dữ liệu làm gì?
- **$70\%$ Tập Train (Học)**: Đưa cho AI học quy luật ngày đêm.
- **$15\%$ Tập Validation (Chữa bài)**: Dùng để tinh chỉnh các nút vặn, thuật toán của AI.
- **$15\%$ Tập Test (Thi tốt nghiệp)**: Đem giấu đi hoàn toàn. Đến phút cuối mới mang ra chấm điểm xem AI học thật hay học tủ.

### ⚠️ Nguy cơ Rò rỉ Nhóm (Group Leakage) tinh vi:
- Trong OULAD có **$3,538$ sinh viên đăng ký học nhiều hơn 1 môn** (ví dụ bạn Bình học cả môn Triết học và môn Lập trình).
- Nếu ta bốc thăm ngẫu nhiên thông thường (Random Split):
  - Bản ghi học môn Triết của bạn Bình rơi vào tập Train (Học).
  - Bản ghi học môn Lập trình của bạn Bình rơi vào tập Test (Thi).
- Vì bạn Bình có giới tính, tuổi tác, vùng quê và hoàn cảnh nghèo khó hoàn toàn giống nhau ở cả 2 môn, mô hình AI khi đi thi sẽ **"nhận ra người quen"** thay vì học quy luật tổng quát!
- **Giải pháp**: Chúng ta áp dụng phương pháp **Stratified Group Split**. Nếu bạn Bình học 3 môn, thì toàn bộ 3 môn của bạn Bình đều phải cùng nằm ở phòng học (Train) hoặc cùng nằm ở phòng thi (Test). Trùng lặp sinh viên giữa 2 phòng đạt **chính xác bằng 0**!

---

## 8. Xử lý Mất cân bằng lớp (Class Imbalance - SMOTE)

### ❓ Vấn đề: "Bài toán đoán mò"
Hãy thử nghĩ: Trong một trường học có $100$ học sinh, chỉ có $9$ bạn đạt loại Xuất sắc (`Distinction`), còn lại $91$ bạn là Trung bình và Khá.
- Nếu một học sinh lười biếng chỉ cần đoán bừa: *"Chẳng ai đạt loại Xuất sắc cả đâu!"*, thì cậu ấy đã đoán đúng tới $91\%$!
- Mô hình AI cũng vậy. Nếu dữ liệu quá chênh lệch, AI sẽ có xu hướng bỏ qua nhóm thiểu số.

### 🛠️ Kỹ thuật SMOTE (Synthetic Minority Over-sampling Technique):
- SMOTE không đơn giản là copy paste dòng dữ liệu cũ (sẽ gây trùng lặp).
- SMOTE tìm các sinh viên có hoàn cảnh gần giống nhau trong nhóm thiểu số, rồi **tạo ra một "sinh viên tổng hợp mới" nằm ở giữa hai người đó**.
- Nhờ vậy, tập Train được cân bằng hoàn hảo tỷ lệ **$50:50$** giữa nhóm An toàn và nhóm Nguy cơ, giúp AI chú ý đồng đều cả hai phía.

---

## 📊 Bảng tổng kết Kết quả đạt được sau Tuần 4

Nhờ quy trình tiền xử lý chuẩn mực trên, khi đưa dữ liệu sạch vào các mô hình học máy để thi thử trên tập Test độc lập ($4,889$ người chưa từng xuất hiện), kết quả đạt được vô cùng ấn tượng:

| Mô hình học máy | Độ chính xác tổng thể (Accuracy) | Tỷ lệ nhận diện đúng nguy cơ (Precision) | Độ bao phủ nhóm nguy cơ (Recall) | Điểm tổng hợp ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | $92.96\%$ | $96.94\%$ | $89.50\%$ | $0.9779$ |
| **Decision Tree** | $92.39\%$ | $95.93\%$ | $89.38\%$ | $0.9728$ |
| **Random Forest (Rừng ngẫu nhiên)** | **$93.07\%$** | **$97.42\%$** | **$89.42\%$** | **$0.9812$** |

> 🌟 **Kết luận quan trọng**: Mô hình Random Forest có thể phát hiện chính xác tới **$97.42\%$** những sinh viên thực sự có nguy cơ gặp khó khăn để nhà trường kịp thời can thiệp, trong khi chỉ số đánh giá tổng thể ROC-AUC đạt **$0.9812$** (gần mức hoàn hảo 1.0).

---

## 🗂️ Tra cứu nhanh các sản phẩm đã hoàn thiện trong dự án

Tất cả những gì được giải thích ở trên đã được đóng gói thành các sản phẩm thực tế:
1. 📓 **Jupyter Notebook tự chạy từ A đến Z**: [notebooks/03_preprocessing.ipynb](file:///c:/Users/mrore/Project_I/notebooks/03_preprocessing.ipynb)
2. 💻 **Mã nguồn thư viện tái sử dụng**: [src/preprocessing.py](file:///c:/Users/mrore/Project_I/src/preprocessing.py)
3. 🧪 **Bộ kiểm thử tự động (9/9 passed)**: [src/tests/test_preprocessing.py](file:///c:/Users/mrore/Project_I/src/tests/test_preprocessing.py)
4. 📈 **6 Biểu đồ trực quan chất lượng cao**:
   - [Hình 15: Missing Values & Điền khuyết](file:///c:/Users/mrore/Project_I/reports/figures/15_missing_values_heatmap_and_imputation.png)
   - [Hình 16: Phân tích sinh viên rút môn sớm](file:///c:/Users/mrore/Project_I/reports/figures/16_early_unregistration_impact.png)
   - [Hình 17: Xử lý ngoại lai và chặn trần Log](file:///c:/Users/mrore/Project_I/reports/figures/17_outliers_boxplots_and_capping.png)
   - [Hình 18: Phân phối chuẩn hóa sau co giãn](file:///c:/Users/mrore/Project_I/reports/figures/18_feature_scaling_distributions.png)
   - [Hình 19: Tính phân tầng và chống trùng lặp sinh viên](file:///c:/Users/mrore/Project_I/reports/figures/19_train_val_test_stratification.png)
   - [Hình 20: Cân bằng lớp SMOTE](file:///c:/Users/mrore/Project_I/reports/figures/20_class_imbalance_and_resampling.png)
5. 📑 **Tài liệu Báo cáo Khoa học (LaTeX / PDF)**: [báo_cáo_tuần_4.tex](file:///c:/Users/mrore/Project_I/báo_cáo_tuần_4.tex) và [báo_cáo_tuần_4.pdf](file:///c:/Users/mrore/Project_I/báo_cáo_tuần_4.pdf)
