# Kịch bản thuyết trình — AI Kaizen Copilot · Motion Analyzer

Deck: `AI_Kaizen_Copilot_Motion_Analyzer.pptx` · 15 slide · thời lượng mục tiêu ≈ 14–15 phút + hỏi đáp.
Kịch bản dưới đây trùng với speaker notes trong file PowerPoint (sinh tự động bởi `build_slides.py`).

Gợi ý phân vai: nhóm trưởng Trịnh Gia Huy mở đầu (slide 1–3) và kết thúc (slide 14–15); các phần giữa chia theo mảng mỗi thành viên phụ trách — đội tự điều chỉnh.

Chuẩn bị demo: `streamlit run app/ui.py`, mở run `20261013-003627_pred_mediapipe_dry_u001-2trials` (link `?run=...`), tab *So sánh video*.

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
thứ ba là Kaizen Copilot dùng LLM có kiểm chứng;
và cuối cùng là dashboard cùng bộ kiểm thử tự động 30 test.

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
EXPLAIN: 7 quy tắc tìm vấn đề, xếp hạng theo ảnh hưởng; LLM giải thích có trích dẫn knowledge base và SOP.
VERIFY: kỹ sư xem dashboard và video bằng chứng, duyệt hay bác bỏ, rồi đo lại bằng đúng phương pháp đó.

## Slide 6 — Kiến trúc pipeline

[~1 phút — kiến trúc]
Về kỹ thuật, pipeline gồm 9 bước, mỗi bước là một script độc lập đọc file và ghi file, nên chạy lại hay thay một khâu rất dễ.
Năm bước đầu là thị giác máy tính: chốt cấu trúc dữ liệu thật, trích keypoint, lọc nhiễu, sinh sự kiện thao tác và tách bước.
Bước 5 và 6 là lớp Kaizen: tính chỉ số 3M và chạy quy tắc phát hiện. Bước 7 là Copilot với LLM, bước 8 sinh báo cáo.
Có ba nguyên tắc quan trọng: thứ nhất, mọi con số do code tính, LLM chỉ giải thích và bị validator kiểm tra.
Thứ hai, chế độ oracle dùng nhãn chuyên gia để kiểm tra nửa sau độc lập với sai số CV.
Thứ ba, mỗi lần chạy là một run có manifest, giúp tái lập và so sánh trước – sau.

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
Mura – biến động: cùng một người nhưng hai lần lắp chênh nhau gần 6,5 giây; bước đặt nắp và bước siết vít biến động nhiều nhất – đó là nơi nên chuẩn hóa thao tác trước.
Muri – quá tải: hệ thống sàng lọc góc cổ tay 2D; tay phải lệch quá 30 độ khoảng 12 đến 15% thời gian. Với tay trái, độ phủ chỉ 27% nên hệ thống ghi rõ là chỉ mang tính sàng lọc.
Khi nạp thêm dữ liệu từ 3 chu kỳ trở lên, quy tắc phát hiện bước chậm bất thường sẽ tự kích hoạt.

## Slide 10 — Findings & Kaizen Copilot

[~1 phút 15 — findings & Copilot]
Từ các chỉ số, 7 quy tắc tìm ra vấn đề và xếp hạng theo ảnh hưởng nhân với độ tin cậy.
Finding đứng đầu là F01: ở chu kỳ V002, người thợ lấy linh kiện ở khay vít và lục giác 4 lần trong khi SOP chỉ cần 3. Ảnh bên trái là bằng chứng tự động: đúng thời điểm 62,6 giây, bước 11, tay phải đang ở khay vít.
Phần Copilot: LLM nhận finding và knowledge base Kaizen, viết quan sát, giả thuyết và đề xuất có trích dẫn. Đây là câu trả lời thật của Gemini 3.8 Flash, trích dẫn KB-13 về tư thế cổ tay và KB-01 về 3M.
Điểm mấu chốt là validator: mọi con số LLM viết phải có trong dữ liệu, mọi trích dẫn phải tồn tại. Trong thử nghiệm, validator đã chặn số bịa và yêu cầu LLM sửa – đúng nỗi lo "AI bịa" trong môi trường nhà máy.

## Slide 11 — Dashboard demo

[~1 phút — demo dashboard; có thể chuyển sang demo trực tiếp]
Toàn bộ kết quả được đưa lên dashboard Streamlit gồm 8 tab, chỉ đọc kết quả của run nên không bao giờ tự tính lại hay sai lệch con số.
Bên trái là tab So sánh video: hai lần lắp V001 và V002 phát song song, bấm "Nhảy tới bước" thì cả hai video cùng về đầu một bước SOP để kỹ sư so cách thao tác.
Bên phải là timeline: hàng bước theo nhãn và hàng bước hệ thống dự đoán xếp chồng nhau để thấy ngay chỗ lệch.
Ngoài ra còn các tab Muda, Mura và Muri, Findings có trích dẫn bấm xem nguyên văn, so sánh run và nút chạy pipeline ngay từ giao diện.

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
