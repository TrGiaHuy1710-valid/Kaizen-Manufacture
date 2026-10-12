# Overview: AI Kaizen Copilot cho nhà máy DENSO

Mình đề xuất kết hợp A1 – tri thức và kinh nghiệm chuyên gia với A3 – phân tích video, tư thế và chuyển động công nhân thành một hệ thống thống nhất: AI Kaizen Copilot – từ quan sát thao tác đến đề xuất cải tiến có bằng chứng.

Ý tưởng quan trọng nhất là không xây dựng hai module AI rời rạc. A3 cung cấp dữ liệu định lượng về những gì đang diễn ra trên dây chuyền; A1 sử dụng kiến thức sản xuất, tiêu chuẩn thao tác và kinh nghiệm chuyên gia để giải thích vấn đề, đề xuất phương án cải tiến và tạo báo cáo để kỹ sư xác nhận.

[How Automated Assembly Processes Boost Manufacturing Quality](https://images.openai.com/static-rsc-4/Sz4XWY_bAuH2WHV5t6p6LHfYA68T3zZN4YFPcBzr73hj_PpSt6GAfEkPisN55a1msw2Y5rzwdlXmOhoO4BBrPm47PULJdScvQk2CksqQUHjrhN3uOKvcP3HXxJde3Kytl0xlohr0P-q8VH1tGlKIVPPFM-pqprPAmlICv08ETMs?purpose=inline)

[kinexon.com](https://kinexon.com/resources/blog/how-to-boost-manufacturing-quality-and-accelerate-assembly-processes-with-kinexon-assembly-tool-control)

Bối cảnh mục tiêu: một trạm lắp ráp linh kiện với camera cố định hoặc video quay bằng điện thoại.

## 1. Phân chia vai trò A1 và A3

## A1 — Expert Knowledge Copilot

Tri thức chuyên gia · SOP · QC · Kaizen rules

- Tra cứu SOP, tiêu chuẩn thao tác và tài liệu hướng dẫn.
- Tiếp nhận vấn đề do kỹ sư mô tả hoặc phát hiện từ video.
- Giải thích nguyên nhân khả dĩ, gợi ý cải tiến theo Kaizen/Lean.
- Sinh checklist, kế hoạch hành động và báo cáo có trích dẫn nguồn.

Câu hỏi trả lời: “Tại sao đây có thể là vấn đề, tiêu chuẩn nào liên quan và nên cải tiến ra sao?”

## A3 — Vision & Motion Analyzer

Computer vision · Pose estimation · Time study

- Phân tích video thành các đoạn thao tác.
- Theo dõi người, keypoints và quỹ đạo di chuyển.
- Ước lượng thời gian làm việc, đi lại, chờ và chu kỳ thao tác.
- Phát hiện dấu hiệu tư thế bất lợi, thao tác lặp lại và di chuyển dư thừa.

Câu hỏi trả lời: “Điều gì xảy ra trong video, ở đâu, bao lâu và có bằng chứng nào?”

## Integration — Kaizen Decision Engine

Kết hợp bằng chứng và tri thức

- Đối chiếu kết quả phân tích với tiêu chuẩn và mục tiêu sản xuất.
- Xếp hạng cơ hội cải tiến theo mức ảnh hưởng, độ tin cậy và chi phí.
- Sinh đề xuất có căn cứ, cho phép kỹ sư duyệt hoặc bác bỏ.
- Lưu lại kết quả trước/sau để xác minh cải tiến.

Giá trị chính: chuyển từ “AI phát hiện chuyển động” sang “AI hỗ trợ quyết định cải tiến”.

## 2. Kiến trúc tổng thể của hệ thống

1\. Input Layer

Video · Sơ đồ trạm · SOP · Mục tiêu sản xuất · Ghi chú chuyên gia

2A. A3 — Vision Pipeline

Person detection/tracking, pose estimation, hand keypoints, motion trajectories, temporal segmentation.

2B. A1 — Knowledge Pipeline

SOP parsing, expert notes, process rules, Kaizen knowledge base, retrieval with source citations.

3\. Evidence & Process Data Layer

Timeline thao tác · Keypoints · Cycle time · Idle/walking time · Event evidence · SOP references · Confidence

4\. Kaizen Decision Engine

Đối chiếu mục tiêu và tiêu chuẩn → tìm vấn đề → xác định nguyên nhân khả dĩ → xếp hạng giải pháp

5\. Engineer Dashboard & Report

Video có timestamp · Biểu đồ thao tác · Spaghetti diagram · Danh sách vấn đề · Kaizen action plan · Before/after

### Nguyên tắc thiết kế quan trọng

Không để LLM tự suy đoán các chỉ số sản xuất từ video. Các chỉ số như thời gian, số lần di chuyển, góc khớp và quãng đường phải do thuật toán xử lý dữ liệu tạo ra. LLM dùng các kết quả này để giải thích, đối chiếu tài liệu và soạn đề xuất.

Ví dụ:

- Vision pipeline xác định người vận hành mất 8,2 giây đi từ trạm lắp ráp đến khay linh kiện.
- Module phân tích quy trình phát hiện hành trình này lặp lại 12 lần trong chu kỳ được ghi nhận.
- A1 truy xuất sơ đồ trạm và tài liệu hướng dẫn bố trí dụng cụ.
- Decision Engine đề xuất đánh giá việc đưa khay linh kiện lại gần vị trí thao tác.
- Dashboard hiển thị video, các lần di chuyển và căn cứ để kỹ sư xác nhận.

Đây là một giả thuyết cải tiến, chưa phải kết luận tiết kiệm chắc chắn. Sau khi bố trí lại, cần đo video trước/sau trong điều kiện tương đương để xác minh.

## 3. Các module chức năng nên triển khai

| Module                        | Chức năng                                            | Đầu ra                           |
| ----------------------------- | ---------------------------------------------------- | -------------------------------- |
| M1. Video Ingestion           | Upload video, chọn đoạn cần phân tích, nhập metadata | Video và thông tin trạm          |
| M2. Worker Tracking           | Phát hiện người và theo dõi xuyên suốt video         | Track ID, bounding boxes         |
| M3. Pose & Motion             | Keypoints, góc khớp, quỹ đạo chuyển động             | Pose time series                 |
| M4. Action Segmentation       | Chia video thành các đoạn thao tác                   | Timeline thao tác                |
| M5. Time Study                | Tính thời lượng thao tác, đi lại, chờ, chu kỳ        | Cycle-time report                |
| M6. Station & Layout Analysis | Gắn tọa độ các trạm, phân tích tuyến di chuyển       | Spaghetti diagram, sơ đồ đề xuất |
| M7. Ergonomic Screening       | Ước lượng tư thế bất lợi và thời gian duy trì        | Cảnh báo tư thế, chỉ số phù hợp  |
| M8. Expert Knowledge Copilot  | Tìm SOP, nhận xét chuyên gia, quy tắc cải tiến       | Evidence-backed suggestions      |
| M9. Kaizen Report             | Xếp hạng vấn đề, tạo action plan, theo dõi cải tiến  | Báo cáo và kết quả trước/sau     |

Bạn không cần triển khai cả chín module ngay từ đầu. M1–M5 và một phiên bản đơn giản của M8 đủ để chứng minh ý tưởng; M6–M7 là các module mở rộng có giá trị nếu dữ liệu hỗ trợ.

## 4. Một workflow thực tế từ video đến Kaizen

Giả sử DENSO muốn phân tích một trạm lắp ráp linh kiện. Người dùng quay khoảng 3–5 chu kỳ thao tác bằng điện thoại, sau đó upload lên hệ thống.

1. Bước 1 — Khai báo quy trình

   Kỹ sư nhập tên trạm, sản phẩm, mục tiêu cycle time, số lượng chu kỳ và tải SOP hoặc ghi chú của chuyên gia. Có thể khai báo vị trí các khay linh kiện, dụng cụ và khu vực thao tác.
2. Bước 2 — Phân tích video bằng A3

   Chạy person detection, tracking và pose estimation. Hệ thống xác định các khoảng thời gian có người di chuyển, đứng tại trạm, với tay hoặc cúi người; đồng thời lưu timestamp để truy xuất video gốc.
3. Bước 3 — Chia chu kỳ thành các bước thao tác

   Ví dụ: lấy linh kiện → di chuyển đến vị trí lắp → lắp ráp → kiểm tra → đặt thành phẩm. Có thể dùng mô hình nhận diện hành động, quy tắc chuyển động hoặc cho kỹ sư xác nhận thủ công ở phiên bản đầu.
4. Bước 4 — Định lượng lãng phí

   Tính thời gian mỗi bước, thời gian di chuyển, số lần đi lại và khoảng thời gian chưa phân loại được. So sánh giữa các chu kỳ để phát hiện sự biến động bất thường.
5. Bước 5 — Kết hợp A1 để phân tích nguyên nhân

   Hệ thống tìm các quy tắc Kaizen liên quan, đối chiếu SOP và ghi chú chuyên gia. Kết quả phải tách rõ dữ liệu quan sát được, giả thuyết nguyên nhân và khuyến nghị.
6. Bước 6 — Kỹ sư xác nhận và lập kế hoạch cải tiến

   Kỹ sư chấp nhận, chỉnh sửa hoặc bác bỏ phát hiện; chọn hành động cần thử nghiệm. Sau khi thực hiện cải tiến, upload video mới để đo lại.

## 5. Dashboard cuối cùng nên trông như thế nào?

## A. Time & Motion Study

Đo thời gian theo chu kỳ và từng bước thao tác.

Cycle time

# 42.0 s

Walking

# 8.2 s

Unclassified

# 3.4 s

Ví dụ phân bổ thời gian trong một chu kỳ

Số liệu minh họa để thiết kế giao diện, không phải kết quả đo thực tế.

0 s6 s12 s18 s24 sLắp rápLấy linh kiệnDi chuyểnChưa phân loạiKhác

Dữ liệu giả lập. Các nhóm thời gian cần được định nghĩa nhất quán và không được đếm trùng.

## B. Motion & Layout Analysis

[Process Modeling Produces Transformational Recipe for Success | MoreSteam](https://images.openai.com/static-rsc-4/EpwFjA-I6sLTBOxjO2SNp6-RtXGhxXJr85EDovdBuiAS-7dPTHWQfzpmfKpVcvbv8JAbvn_cDBbX6ei0YiS2XnLErw-z_GXwo9vr32gCExzkz9KBDsJJqXt4Aay6Z83EPQ8rGz1-LFbaQujgkSO0QsMWwblKl0-0uSDm28pWZgI?purpose=inline)

[moresteam.com](https://www.moresteam.com/resources/case-studies/pizza-process-challenges)

Hiển thị đường đi của công nhân, số lần di chuyển giữa các vị trí và vị trí có thể bố trí lại. Để tính quãng đường thực tế, cần hiệu chuẩn camera hoặc xác định tỷ lệ và tọa độ mặt bằng; pixel trên video không tự động tương đương mét.

## C. AI Kaizen Findings

Phát hiện 01 — Di chuyển lấy linh kiện

Bằng chứng: 12 lần di chuyển trong các chu kỳ được phân tích; thời gian di chuyển trung bình 8,2 giây/chu kỳ trong ví dụ giả lập.

Giả thuyết: vị trí khay linh kiện có thể chưa tối ưu.

Đề xuất: thử bố trí khay gần vùng thao tác và đo lại.

Trạng thái: Chờ kỹ sư xác nhận.

Phát hiện 02 — Tư thế cúi người

Bằng chứng: góc thân người vượt ngưỡng cấu hình trong một số khoảng thời gian.

Đề xuất: xem lại độ cao bàn và vị trí linh kiện; kỹ sư cần đánh giá theo bối cảnh công việc.

Không được coi riêng góc pose là đánh giá rủi ro công thái học hoàn chỉnh.

Các con số trên chỉ minh họa cách trình bày dashboard, không phải kết quả từ một video DENSO thực tế.

## 6. Phạm vi triển khai phù hợp cho một cuộc thi hackathon

Mình sẽ chia thành ba giai đoạn để tránh ôm quá nhiều bài toán.

Phase 1

MVP

Video → Pose → Time Study

- Upload một video, xem video có skeleton overlay.
- Tracking người và lưu keypoints theo frame.
- Tính thời gian đứng yên/di chuyển theo các quy tắc cơ bản.
- Timeline, biểu đồ thời gian và CSV/JSON.
- Cho phép kỹ sư gắn nhãn thao tác thủ công.

Nên hoàn thành trước

Phase 2

A1 + A3

Evidence-based Kaizen Copilot

- Nhập SOP và tài liệu Kaizen.
- Truy xuất đoạn tài liệu phù hợp với phát hiện từ video.
- Sinh nguyên nhân khả dĩ và đề xuất cải tiến.
- Tạo báo cáo với timestamp, nguồn tham khảo và trạng thái duyệt.

Phase 3

Advanced

Motion Optimization & Validation

- Phân tích quỹ đạo di chuyển và bố trí trạm.
- Phát hiện các bước thao tác theo thời gian.
- Sàng lọc rủi ro công thái học bằng phương pháp phù hợp.
- So sánh trước/sau cải tiến, ước lượng lợi ích và độ tin cậy.

## 7. Công nghệ đề xuất

| Thành phần             | Công nghệ gợi ý                                      | Lý do                                            |
| ---------------------- | ---------------------------------------------------- | ------------------------------------------------ |
| Video processing       | OpenCV                                               | Đọc video, lấy frame, timestamp, crop và overlay |
| Pose estimation        | YOLO Pose hoặc MoveNet                               | Bắt đầu bằng pretrained model                    |
| Tracking               | ByteTrack hoặc tracker tương đương                   | Giữ ID người qua các frame                       |
| Motion analysis        | NumPy, SciPy, Pandas                                 | Góc khớp, vận tốc, thời lượng và thống kê        |
| Action segmentation    | Quy tắc ban đầu, sau đó thử mô hình temporal         | Giảm phụ thuộc vào việc LLM đoán hành động       |
| A1 knowledge retrieval | Embeddings + vector database + RAG                   | Truy xuất SOP và ghi chú chuyên gia              |
| Reasoning              | LLM có structured output                             | Giải thích bằng chứng và tạo action plan         |
| Backend                | FastAPI hoặc Express                                 | Quản lý job phân tích, metadata và kết quả       |
| Frontend               | React + Vite                                         | Video player, timeline, dashboard                |
| Storage                | Local filesystem ban đầu; object storage khi mở rộng | Lưu video và kết quả có cấu trúc                 |

Với bài toán này, không nhất thiết phải dùng một mô hình lớn cho toàn bộ pipeline. Phần định lượng nên chạy bằng code và mô hình thị giác chuyên biệt; LLM chỉ tham gia ở bước giải thích, truy xuất tri thức và hỗ trợ lập kế hoạch.

## 8. Những giới hạn cần xử lý ngay từ thiết kế

- Pose không đồng nghĩa với nhận diện hành động: thấy tay chuyển động chưa đủ để kết luận công nhân đang lắp vít hay kiểm tra chất lượng.
- Idle không đồng nghĩa với lãng phí: người đứng yên có thể đang chờ chu kỳ máy, quan sát chất lượng hoặc thực hiện thao tác nhỏ.
- Camera đơn không đảm bảo đo được 3D: che khuất, góc quay và phối cảnh có thể làm sai góc khớp và quãng đường.
- Rủi ro công thái học cần thêm thông tin: REBA/RULA cần đánh giá các bộ phận cơ thể và bối cảnh phù hợp; lực tác dụng và trọng lượng không thể suy ra đáng tin cậy chỉ từ video RGB thông thường.
- Không để AI tự động sửa SOP hay thay đổi quy trình: mọi thay đổi phải được kỹ sư có thẩm quyền duyệt.
- Bảo vệ dữ liệu nhà máy: dùng video được phép sử dụng, hạn chế lưu hình ảnh nhận dạng, phân quyền và đặt thời hạn lưu trữ phù hợp.

## 9. Định nghĩa thành công của dự án

Không nên chỉ trình diễn skeleton đẹp hoặc chatbot trả lời tốt. Hãy đánh giá bằng các chỉ số cụ thể:

| Nhóm                | Chỉ số                                                                  |
| ------------------- | ----------------------------------------------------------------------- |
| Pose                | Tỷ lệ phát hiện người, keypoint confidence, PCK/OKS nếu có ground truth |
| Tracking            | Tỷ lệ ID switch, tỷ lệ mất track                                        |
| Time study          | Sai số thời lượng so với nhãn chuyên gia, độ ổn định giữa các chu kỳ    |
| Action segmentation | Precision, recall và F1 theo từng loại thao tác                         |
| Kaizen Copilot      | Tỷ lệ đề xuất được kỹ sư chấp nhận, độ chính xác trích dẫn SOP          |
| Giá trị cải tiến    | Thay đổi cycle time, quãng đường và chỉ số công thái học trước/sau      |

### Kết luận

Nếu cần một sản phẩm có câu chuyện rõ ràng cho DENSO, mình sẽ chọn hướng:

A3 đo lường thao tác thực tế + A1 ghi nhận và truy xuất tri thức chuyên gia + Kaizen Copilot tổng hợp bằng chứng, đề xuất cải tiến và xác minh kết quả.

Điểm khác biệt của dự án không phải là thay thế chuyên gia công nghiệp, mà là giảm thời gian thu thập và tổng hợp dữ liệu để chuyên gia dành nhiều thời gian hơn cho cải tiến thực tế.

Bản MVP tốt nhất nên bắt đầu từ một video, một trạm, một người vận hành và một mục tiêu đo lường. Chứng minh được time study có độ tin cậy, sau đó mới mở rộng sang spaghetti diagram, ergonomic screening và tối ưu hóa quy trình.