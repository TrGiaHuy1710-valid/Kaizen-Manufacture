# Kịch bản thuyết trình — AI Kaizen Copilot · Motion Analyzer

Deck: `AI_Kaizen_Copilot_Motion_Analyzer.pptx` · 15 slide · thời lượng mục tiêu ≈ 14–15 phút + hỏi đáp.
Kịch bản dưới đây trùng với speaker notes trong file PowerPoint (sinh tự động bởi `build_slides.py`).

Gợi ý phân vai: nhóm trưởng Trịnh Gia Huy mở đầu (slide 1–3) và kết thúc (slide 14–15); các phần giữa chia theo mảng mỗi thành viên phụ trách — đội tự điều chỉnh.

Chuẩn bị demo: `python -m kaizen.core.projects prepare-demo --trials U001_V001,U001_V002` (in bảng READY), rồi `KAIZEN_DEMO=1 streamlit run app/ui.py`; mở thẳng dự án: `?page=project&id=U001_V001`.

## Slide 1 — Bìa

[~45 giây — mở đầu]
Kính chào ban giám khảo. Chúng em là đội AI Kaizen Copilot, gồm Trịnh Gia Huy (nhóm trưởng), Vũ Văn Mạnh, Nguyễn Trần Quang và Trịnh Hoài Nam.
Đề tài của đội là "Motion Analyzer": biến video thao tác lắp ráp thành các chỉ số Lean Muda – Mura – Muri và đề xuất Kaizen có kiểm chứng.
Đề tài thuộc danh mục A2: tích lũy know-how của người thợ giỏi thành dữ liệu để AI học và để đào tạo người mới.
Trong khoảng 15 phút, chúng em sẽ trình bày bài toán, giải pháp, kết quả đo thật trên dataset HA4M và kế hoạch pilot tại DENSO.

## Slide 2 — Tóm tắt đề xuất

[~1 phút — tóm tắt cho giám khảo]
Slide này trả lời nhanh 6 câu hỏi của ban tổ chức.
Điểm mới: chúng em đo cả 3M chứ không chỉ bấm giờ, và tách bạch "con số do thuật toán tính – LLM chỉ giải thích", có validator tự động chặn số liệu hay trích dẫn bịa.
Vấn đề: bấm giờ thủ công tốn 2 đến 4 giờ một trạm và know-how của người thợ giỏi không được lưu lại.
Kết quả đã đo: sai số cycle time 0,6 giây trên chu kỳ 77,9 giây, F1 phân đoạn bước 0,75, và cổ tay phải khớp cảm biến Azure Kinect ở 95% khung hình.
Thông điệp chính: nếu được chọn, đội sẽ pilot 4 tuần tại một trạm, đo trước và sau bằng cùng một phương pháp.

## Slide 3 — Giới thiệu team

[~45 giây — giới thiệu đội]
Đội gồm 4 thành viên: Trịnh Gia Huy là nhóm trưởng, cùng Vũ Văn Mạnh, Nguyễn Trần Quang và Trịnh Hoài Nam.
Trong giai đoạn hackathon, đội đã xây dựng đủ 4 mảng của một hệ thống chạy thật:
thứ nhất là thị giác máy tính để trích khung xương và sự kiện thao tác;
thứ hai là lớp chỉ số Kaizen – phân đoạn bước SOP và tính Muda, Mura, Muri;
thứ ba là RAG tài liệu và Kaizen Copilot dùng LLM có kiểm chứng;
và cuối cùng là giao diện theo dự án video cùng bộ kiểm thử tự động 117 test.

## Slide 4 — Bài toán tại line

[~1 phút — bài toán]
Ở dây chuyền lắp ráp, cái làm nên năng suất của người thợ giỏi là thứ tự động tác, nhịp làm việc và cách phối hợp hai tay. Đây là tri thức ngầm: không nằm trong tài liệu, chỉ truyền qua kèm cặp, và mất khi người thợ nghỉ.
Muốn đo thì hiện nay kỹ sư IE phải bấm giờ thủ công, 2 đến 4 giờ cho một trạm, mỗi người đo một kiểu và chỉ lấy được vài chu kỳ.
Hệ quả là ba loại lãng phí theo ngôn ngữ Lean – Muda, Mura, Muri – không được theo dõi liên tục, và đề xuất Kaizen khó chứng minh bằng số liệu trước sau.
Hướng giải của đội: dùng camera đo mọi chu kỳ, dùng LLM giải thích theo tri thức Kaizen nhưng có kiểm chứng, và kỹ sư là người quyết định.

## Slide 5 — Luồng giải pháp

[~1 phút — giải pháp]
Giải pháp là một luồng khép kín bốn khâu.
OBSERVE: camera thường quay trạm, MediaPipe trích khung xương ngay trên CPU, chỉ lưu tọa độ khớp nên không cần giữ khuôn mặt.
MEASURE: hệ thống tách 12 bước SOP dựa trên vùng khay linh kiện và quy tắc, rồi tính cycle time, tỷ lệ giá trị gia tăng, thời gian chờ, số lần lấy linh kiện, độ biến động và góc cổ tay.
EXPLAIN: 8 quy tắc tìm vấn đề, xếp hạng theo ảnh hưởng — trong đó có quy tắc so thời lượng từng bước với thời gian chuẩn đọc thẳng từ work instruction; LLM giải thích và trích dẫn đúng tài liệu, đúng trang.
VERIFY: kỹ sư xem dashboard và video bằng chứng, duyệt hay bác bỏ, rồi đo lại bằng đúng phương pháp đó.

## Slide 6 — Kiến trúc luồng

[~1 phút 15 — kiến trúc luồng]
Đây là kiến trúc sau khi đội tách lõi thành package kaizen gồm bốn phần: skeleton cho thị giác máy tính, analyzer cho chỉ số Kaizen, rag cho tài liệu và llm cho Copilot.
Hàng trên là 6 bước của một lần chạy, cũng chính là 6 bước tiến độ người dùng thấy trên giao diện: trích keypoint, sinh sự kiện thao tác, tách bước SOP, tính 3M và chạy 8 quy tắc, LLM giải thích, rồi xuất báo cáo và video overlay.
Hàng giữa là làn tài liệu: work instruction hay guideline được chia theo mục và trang, nhúng vector và lưu cục bộ. Từ đó hệ thống đọc ra thời gian chuẩn từng bước và ngưỡng 30% để chạy quy tắc R8, đưa đoạn tài liệu cho LLM trích dẫn, và ghi bảng tham chiếu trong báo cáo — mọi con số chuẩn đều chỉ được về đúng tài liệu, đúng trang.
Hàng dưới là các ứng dụng dùng chung lõi: giao diện theo dự án video, job chạy nền, so sánh 3M nhiều video và API. Không ứng dụng nào tự tính lại số.

## Slide 7 — Dữ liệu & độ tin cậy keypoint

[~1 phút — dữ liệu]
Để phát triển và đánh giá khách quan, đội dùng HA4M, một dataset công khai: 41 người lắp bộ bánh răng hành tinh gồm 12 bước, có nhãn chuyên gia theo từng khung hình. Trong slide này là hai chu kỳ lắp của người U001.
Hình bên trái là một khung hình từ video demo do hệ thống sinh ra: vùng khay được tô theo màu bước, vệt cổ tay trong khoảng 1 giây và trạng thái hai tay.
Độ tin cậy keypoint: hơn 95% khung hình phát hiện được người. So với skeleton của cảm biến Azure Kinect, cổ tay phải lệch trung vị 24,5 pixel và 95% khung hình dưới 50 pixel.
Tay trái hay bị thân che nên kém tin cậy hơn; hệ thống gắn cờ điều này thay vì che giấu.

## Slide 8 — Phân đoạn bước: CV vs nhãn

[~1 phút 15 — kết quả phân đoạn]
Đây là kết quả quan trọng nhất về độ chính xác. Mỗi biểu đồ có hàng trên là bước theo nhãn chuyên gia, hàng dưới là bước do hệ thống tự tách, cùng hoạt động của hai tay.
Với chu kỳ V001, sai số cycle time chỉ 0,6 giây trên 77,9 giây, tức khoảng 0,8%; F1 phân đoạn bước 0,75; sai số thời lượng mỗi bước trung bình 0,62 giây.
Chu kỳ V002 khó hơn: F1 0,71, sai số cycle time 2,27 giây, vì hệ thống chưa bắt được hai lần lấy linh kiện.
Đây mới là baseline dựa trên quy tắc, chưa huấn luyện model nào. Điểm quan trọng là hệ thống trung thực về giới hạn: phần không chắc chắn được tô xám và báo thiếu bước, không tự đoán.

## Slide 9 — Muda · Mura · Muri

[~1 phút 15 — 3M]
Từ các bước đã tách, hệ thống tính đủ ba chiều Lean.
Muda – lãng phí: biểu đồ Yamazumi bên trái chồng thời lượng 12 bước cho mỗi chu kỳ; tỷ lệ giá trị gia tăng theo nhãn là 82 và 85%; mỗi chu kỳ có 15 và 14 lần lấy linh kiện.
Mura – biến động: cùng một người nhưng hai lần lắp chênh nhau gần 6,5 giây; bước ghép Block 2 lên Block 1 và bước siết vít biến động nhiều nhất – đó là nơi nên chuẩn hóa thao tác trước, và cũng chính là hai bước bị quy tắc R8 bắt vì chậm hơn thời gian chuẩn.
Muri – quá tải: hệ thống sàng lọc góc cổ tay 2D; tay phải lệch quá 30 độ khoảng 12 đến 15% thời gian. Với tay trái, độ phủ chỉ 27% nên hệ thống ghi rõ là chỉ mang tính sàng lọc.
Phần 3M tổng hợp nhiều video được tách sang trang So sánh 3M: người đánh giá chọn các video rồi mới tính; khi có từ 3 video trở lên, quy tắc bước chậm bất thường sẽ tự kích hoạt.

## Slide 10 — Findings & Copilot có truy vết

[~1 phút 15 — findings & Copilot có truy vết]
Đây là kết quả thật trên hai lần lắp của người U001, với work instruction WI-EGT-001 đã được nạp vào hệ thống.
Hệ thống tự đọc bảng thời gian chuẩn của 12 bước trong tài liệu và ngưỡng "chênh lệch trên 30% cần ghi nhận". Từ đó quy tắc R8 bắt được F01: ở lần lắp V002, bước 9 – ghép Block 2 lên Block 1 – mất 14,97 giây, chậm hơn chuẩn 8,5 giây tới 76%. Ảnh bên trái là bằng chứng tự động đúng thời điểm đó, thấy rõ thẻ số 9 trên bàn.
Phần Copilot: Gemini 2.5 Flash viết quan sát, giả thuyết, đề xuất theo ECRS và cách đo lại; mọi câu đều trích dẫn về đúng tài liệu và trang – ở đây là trang 1 của WI-EGT-001.
Validator vẫn là chốt chặn: mọi con số phải có trong dữ liệu, mọi trích dẫn phải tồn tại trong kho tài liệu, độ dài từng trường phải đúng quy định; sai thì LLM phải sửa.

## Slide 11 — UI dự án video

[~1 phút — demo giao diện; có thể chuyển sang demo trực tiếp]
Giao diện được thiết kế lại theo đúng cách kỹ sư làm việc: mỗi video là một dự án.
Trang Dự án: thêm video – từ dataset, file mp4 tự quay hoặc thư mục ảnh. Trang Chi tiết: bấm Chạy full luồng, hệ thống hiện tiến độ 6 bước, xong thì có video overlay theo nhãn chuẩn như hình trái, và Kaizen Brain ở giữa – bảng từng bước so với thời gian chuẩn, kèm nguồn là trang của work instruction, cùng các finding.
Trang Tài liệu để nạp work instruction và thử hỏi; trang So sánh 3M để người đánh giá chọn nhiều video rồi mới tính biến động, như Yamazumi bên phải.
Khi demo, chế độ demo chỉ phát lại kết quả đã tính sẵn và luôn có nhãn báo rõ; video mới thì hệ thống luôn xử lý thật, không bao giờ bịa số.

## Slide 12 — Hiệu quả mang lại

[~1 phút — hiệu quả]
Tóm lại các con số đã đo được – tất cả đo thật, không phải mục tiêu: sai số cycle time 0,6 giây, F1 phân đoạn bước 0,75, 95% khung hình cổ tay khớp cảm biến công nghiệp dưới 50 pixel, và phân tích hai chu kỳ chỉ mất khoảng 9 giây sau khi có keypoint.
Về kinh doanh: chi phí phần cứng gần như bằng 0 vì chỉ cần webcam và laptop; đo được 100% chu kỳ thay vì vài mẫu; và golden cycle của người thợ giỏi được lưu lại để đào tạo người mới.
Chúng em nói rõ: hiệu quả vận hành tại DENSO như thời gian time study dưới 30 phút là ước tính và cần pilot để xác nhận, theo đúng ba bước xác minh bên phải.

## Slide 13 — Kế hoạch triển khai

[~1 phút — kế hoạch]
Giai đoạn hackathon đã xong: pipeline chạy end-to-end, có đánh giá độ chính xác, validator LLM và dashboard.
Bốn tuần đầu: pilot tại một trạm thật của DENSO – hiệu chỉnh camera, vẽ vùng khay theo layout thật, đổi đơn vị pixel sang centimet và đối chiếu với kỹ sư IE.
Tháng 2 đến 3: mở rộng 3 đến 5 trạm, xây thư viện golden cycle, nạp SOP nội bộ và đo tỷ lệ đề xuất được kỹ sư chấp nhận.
Tháng 4 đến 6: dùng dữ liệu tích lũy để huấn luyện model phân đoạn học sâu, hỗ trợ nhiều camera và đưa vào đào tạo công nhân mới.
Về quyền riêng tư, hệ thống chỉ cần lưu khung xương, không cần lưu khuôn mặt.

## Slide 14 — Kết luận

[~45 giây — kết luận]
Tóm lại, AI Kaizen Copilot không chỉ bấm giờ: nó biến mỗi thao tác thành tri thức đo được.
MEASURE: đo bước SOP, cycle time và 3M từ camera thường.
EXPLAIN: Copilot giải thích có trích dẫn và không được phép tạo số liệu.
VERIFY: kỹ sư là người quyết định, và cải tiến được đo lại bằng đúng phương pháp đó.
Đề xuất của đội: pilot 4 tuần tại một trạm DENSO.

## Slide 15 — Cảm ơn — Hỏi & đáp

[Hỏi & đáp]
Cảm ơn ban giám khảo đã lắng nghe. Đội sẵn sàng demo trực tiếp dashboard và trả lời câu hỏi.
Câu hỏi có thể gặp: (1) Khác gì bấm giờ bằng video thường? → đo tự động 100% chu kỳ, đủ 3M, có bằng chứng và đánh giá với nhãn chuyên gia.
(2) LLM có bịa không? → mọi số do code tính; validator chặn số/trích dẫn không có thật, có retry và báo tỷ lệ đạt.
(3) Áp dụng trạm khác thế nào? → chỉ cần vẽ zone và khai báo SOP cho trạm; mỗi lần chạy là một run để so sánh.
(4) Quyền riêng tư? → chỉ cần lưu khung xương; mục tiêu cải tiến quy trình, không giám sát cá nhân.
