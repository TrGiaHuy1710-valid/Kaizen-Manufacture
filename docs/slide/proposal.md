Trước hết là lựa chọn quan trọng nhất: **danh mục**. Đề của bạn khớp nhất với **A2 — Accumulate Master Know-How for AI to Learn**. Lý do: bản chất hệ thống là biến kỹ năng ngầm (tacit know-how) của công nhân lành nghề và kỹ sư IE thành dữ liệu đo được — golden cycle của người thợ giỏi, các nguyên tắc Kaizen/ECRS trong knowledge base có trích dẫn, và khoảng cách giữa người mới với người thạo việc. A3 (document AI) chỉ là một mảnh nhỏ (SOP/KB), còn H/D không liên quan. Dưới đây là nội dung điền sẵn cho từng ô, dùng số liệu thật từ pipeline của bạn.

---

**Danh mục quan tâm:** A2 — Accumulate Master Know-How for AI to Learn

**1. Vấn đề bạn nhận thấy**

Bí quyết thao tác của công nhân lành nghề (thứ tự động tác, nhịp làm việc, cách phối hợp hai tay, tư thế an toàn) là tri thức ngầm, chỉ tồn tại trong cơ thể người thợ. Hiện nay việc thu thập tri thức này phụ thuộc vào quan sát thủ công: kỹ sư IE bấm giờ từng công đoạn, ghi chép tay, mỗi lần đo tốn hàng giờ và kết quả phụ thuộc người đo, khó lặp lại. Hệ quả: (1) chuẩn thao tác (standard work) chậm được cập nhật và khó chứng minh bằng số liệu; (2) khi người thợ giỏi nghỉ việc, know-how mất theo; (3) đào tạo người mới thiếu thước đo khách quan để biết họ khác người thạo việc ở bước nào; (4) lãng phí (Muda), biến động (Mura) và quá tải cổ tay (Muri) không được phát hiện kịp vì không ai đo liên tục.

**2. Giải pháp bạn đề xuất**

*Mục đích:* AI Kaizen Motion Analyzer — biến mỗi thao tác của công nhân thành dữ liệu để tích lũy know-how: camera thường quay công đoạn lắp ráp, AI trích xuất khung xương (skeleton), tự động phân đoạn thành các bước SOP, tính các chỉ số Muda/Mura/Muri, so sánh với "chu kỳ vàng" của người thợ giỏi, và sinh đề xuất cải tiến có trích dẫn tiêu chuẩn để kỹ sư duyệt. AI không thay thế chuyên gia — nó rút ngắn khâu thu thập và tổng hợp dữ liệu để chuyên gia dành thời gian cho cải tiến.

*Công nghệ sử dụng:* Pose estimation (MediaPipe, RTMPose/RTMW — 2D whole-body, chạy trên CPU/laptop); lọc nhiễu One Euro filter; phân đoạn bước bằng zone-based rules (shapely) kết hợp phân tích chuỗi thời gian; thống kê Muda/Mura/Muri bằng NumPy/Pandas; LLM (Claude/Gemini) giải thích phát hiện và trích dẫn knowledge base Kaizen, kèm validator tự động kiểm tra LLM không được bịa số liệu hay trích dẫn; dashboard Streamlit; đối chứng độ chính xác bằng skeleton Azure Kinect và nhãn chuyên gia của dataset công khai HA4M.

*Dữ liệu đầu vào:* Video RGB từ camera thường hoặc điện thoại (1 trạm, 1 công nhân); file cấu hình trạm (vùng khay linh kiện, vùng lắp, danh sách bước SOP); knowledge base nguyên tắc Kaizen/ECRS/tiết kiệm động tác; (giai đoạn phát triển) dataset công khai HA4M — 41 người lắp bộ bánh răng, 12 bước, có nhãn chuyên gia làm ground truth.

*Dữ liệu đầu ra:* Timeline từng bước thao tác (Gantt); biểu đồ Yamazumi và %VA/NVA; cycle time và độ biến động giữa các lần lắp/giữa các công nhân (Mura); số lần với tay, quãng đường tay, thời gian tay chờ (Muda); sàng lọc tư thế cổ tay (Muri); danh sách phát hiện xếp hạng theo mức ảnh hưởng, kèm giải thích của LLM có trích dẫn nguồn và trạng thái kiểm chứng; báo cáo Markdown/dashboard, video overlay làm bằng chứng.

**3. Quy trình thực hiện công việc**

*Trước cải tiến:* Kỹ sư IE ra hiện trường bấm giờ thủ công từng bước (2–4 giờ cho một trạm), ghi chép giấy, nhập liệu, vẽ Yamazumi bằng tay; mỗi người đo một kiểu, chỉ đo được vài chu kỳ nên bỏ sót biến động; know-how của người thợ giỏi chỉ truyền miệng qua kèm cặp.

*Sau cải tiến:* Quay video 5–10 phút tại trạm → hệ thống tự trích skeleton, phân đoạn bước, tính toàn bộ chỉ số trong vài phút → kỹ sư nhận dashboard với các phát hiện đã xếp hạng kèm bằng chứng video và trích dẫn tiêu chuẩn → kỹ sư duyệt/bác bỏ, thử cải tiến, quay lại để so before/after bằng số đo cùng phương pháp. Chu kỳ vàng của người thợ giỏi được lưu thành chuẩn số hóa dùng cho đào tạo người mới.

**4. Hiệu quả mang lại**

Kết quả đo thực tế trên dataset HA4M (baseline chưa huấn luyện model): sai số cycle time 0,6 giây trên chu kỳ 77,9 giây (≈0,8%); sai số thời lượng trung bình mỗi bước 0,62 giây; F1@IoU0.5 phân đoạn bước = 0,75; độ lệch cổ tay so với cảm biến Azure Kinect: trung vị 24,5 px, 95% khung hình dưới 50 px (tay thuận). Hiệu quả vận hành (ước tính, cần kiểm chứng tại hiện trường): giảm thời gian một lần time study từ 2–4 giờ xuống dưới 30 phút thao tác người; đo được 100% chu kỳ thay vì mẫu nhỏ, nên phát hiện được biến động và thao tác bất thường; chi phí phần cứng gần bằng 0 (webcam + laptop sẵn có); know-how người thợ giỏi được lưu trữ, không mất khi nhân sự thay đổi.

**5. Sự vượt trội/độc đáo của ý tưởng**

(1) Đóng khung theo đúng ngôn ngữ Lean của DENSO: hệ thống đo đồng thời Muda – Mura – Muri, không chỉ bấm giờ. (2) Kiến trúc "số do thuật toán tính, LLM chỉ giải thích": mọi con số truy vết được về dữ liệu gốc, và validator tự động bác bỏ câu trả lời LLM chứa số liệu hoặc trích dẫn không tồn tại (đã bắt được lỗi thật trong thử nghiệm) — giải quyết đúng nỗi lo AI "bịa" trong môi trường nhà máy. (3) Độ chính xác được kiểm chứng kép: so với nhãn chuyên gia của dataset công khai và so với cảm biến depth công nghiệp, trong khi chỉ dùng camera RGB thường. (4) Chỉ lưu skeleton thay vì video nhận dạng được khuôn mặt — bảo vệ quyền riêng tư, định vị là công cụ cải tiến quy trình chứ không giám sát cá nhân. (5) Không cần robot, PLC, cảm biến lực hay GPU server; triển khai được ngay trên một trạm bằng thiết bị sẵn có.

**6. Kế hoạch triển khai trong bao lâu**

Đã hoàn thành (giai đoạn hackathon): pipeline end-to-end chạy trên dataset công khai, có đánh giá độ chính xác, validator LLM và dashboard demo. Giai đoạn 1 (tuần 1–4): pilot tại 1 trạm thật của DENSO — hiệu chỉnh camera, vẽ zone theo layout thật, đối chiếu với kỹ sư IE, chuẩn hóa thước đo (px→cm). Giai đoạn 2 (tháng 2–3): mở rộng 3–5 trạm, xây thư viện golden cycle, tích hợp SOP nội bộ vào knowledge base, đánh giá tỷ lệ đề xuất được kỹ sư chấp nhận. Giai đoạn 3 (tháng 4–6): nâng phân đoạn bằng model học sâu (ST-GCN/MS-TCN) huấn luyện trên dữ liệu đã tích lũy, hỗ trợ đa camera, đưa vào quy trình đào tạo công nhân mới.



