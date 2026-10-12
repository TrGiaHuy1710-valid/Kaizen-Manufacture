## Đánh giá overview

**Phần tốt, nên giữ nguyên:** nguyên tắc "LLM không tự suy ra chỉ số, mọi con số phải do code tính", việc tách rõ quan sát / giả thuyết / khuyến nghị, kỹ sư là người duyệt cuối, phần giới hạn (idle không đồng nghĩa lãng phí, pose không đồng nghĩa hành động), và bảng chỉ số thành công. Đây là tư duy đúng và giám khảo kỹ thuật sẽ đánh giá cao.

**Phần cần sửa:**

- **Chọn model mâu thuẫn với bài toán.** YOLO Pose và MoveNet chỉ có 17 điểm thân, không có ngón tay. Với trạm lắp ráp ngồi tại bàn, gần như toàn bộ thông tin nằm ở bàn tay. Giữ RTMW hoặc MediaPipe như đã chuẩn bị. YOLO còn vướng license AGPL.
- **Trọng tâm "đi lại / spaghetti diagram" không khớp dữ liệu.** HA4M là lắp ráp tại chỗ, gần như không đi lại. Lãng phí ở loại trạm này là với tay xa, tay chờ, thao tác thừa. Spaghetti diagram nên để Phase 3.
- **ByteTrack không cần thiết** khi mỗi trạm chỉ có một công nhân. Bỏ đi để bớt một nguồn lỗi.
- **A1 không có SOP thật của DENSO.** Vector DB là thừa với lượng tài liệu nhỏ. Thay vào đó, viết một SOP mô phỏng cho bài lắp bánh răng của HA4M và một bộ tri thức Kaizen từ nguồn công khai (nguyên tắc tiết kiệm động tác của Barnes, ECRS, 3M). Vài chục đoạn văn thì đưa thẳng vào prompt hoặc dùng tìm kiếm từ khóa đơn giản.
- **"Decision Engine" còn mơ hồ.** Cần định nghĩa cụ thể bằng các rule phát hiện vấn đề và một công thức xếp hạng đơn giản.
- **Thiếu khung 3M** (Muda / Mura / Muri) đã bàn ở trên. Overview chỉ nói Muda và một chút ergonomic, nên bổ sung vì đây là câu chuyện mạnh nhất với DENSO.

## Kế hoạch test toàn luồng trong 1 ngày (không UI)

Nguyên tắc: **chạy được thông suốt từ đầu đến cuối với phiên bản thô trước, rồi mới cải thiện từng khâu.** Mỗi bước là một script đọc file đầu vào và ghi file đầu ra, nên có thể chạy lại riêng từng bước.

Mẹo quan trọng nhất: **chế độ "oracle".** HA4M đã có nhãn hành động. Bạn có thể đưa nhãn thật thay cho kết quả phân đoạn của mình vào các bước 5–8. Nhờ vậy, kể cả khi phần CV chưa tốt, nửa sau của luồng (chỉ số, phát hiện, LLM, báo cáo) vẫn test được trong ngày. Cuối cùng báo cáo cả hai: kết quả với nhãn thật và với nhãn dự đoán, chênh lệch giữa hai cái chính là sai số của phần CV.

**Bước 0 — Cấu hình (khoảng 1 giờ).** Tạo `station.yaml` gồm: các zone dạng polygon (khay linh kiện, vùng lắp, vùng thành phẩm), danh sách các bước SOP theo thứ tự kèm zone tương ứng, cycle time mục tiêu, scale pixel sang cm. Viết `sop.md` mô phỏng 12 bước lắp bánh răng dựa theo paper HA4M, và `kaizen_kb.md` chứa các nguyên tắc tiết kiệm động tác, ECRS, 3M, mỗi đoạn có ID để trích dẫn.

**Bước 1 — Trích xuất keypoints (khoảng 1 giờ, chạy nền).** Dùng `extract_keypoints.py` đã có cho 3–5 video của 2–3 người. Kiểm tra: tỷ lệ phát hiện cổ tay trên 90%, xem video preview bằng mắt. Lưu ý kiểm tra lại xem mỗi video HA4M có phải là một lần lắp trọn vẹn không. Nếu đúng vậy thì mỗi video là một chu kỳ, việc tách chu kỳ trở nên đơn giản, và so sánh giữa các video chính là phân tích Mura.

**Bước 2 — Tiền xử lý (khoảng 1 giờ).** Nội suy các đoạn mất keypoint ngắn, lọc nhiễu bằng One Euro filter, tính vận tốc và quãng đường của hai cổ tay. Đầu ra: `features.parquet`. Kiểm tra: vẽ đồ thị vận tốc theo thời gian, phải thấy các nhịp lên xuống rõ ràng thay vì nhiễu.

**Bước 3 — Sinh sự kiện (khoảng 1.5 giờ).** Với mỗi tay, mỗi frame: tay đang ở zone nào, đang di chuyển hay đứng yên. Gộp các frame liên tiếp cùng trạng thái thành sự kiện, bỏ các sự kiện quá ngắn. Đầu ra: `events.csv` (bắt đầu, kết thúc, tay, zone, trạng thái). Kiểm tra: in timeline ra và đối chiếu với video ở vài timestamp.

**Bước 4 — Ánh xạ sang bước SOP (khoảng 1 giờ).** Ghép chuỗi sự kiện với chuỗi bước trong `station.yaml` để ra timeline theo bước. Đây là khâu dễ sai nhất, nên làm phiên bản rule đơn giản trước, và dùng chế độ oracle để không bị kẹt ở đây.

**Bước 5 — Tính chỉ số (khoảng 1 giờ).** Muda: thời gian từng bước, số lần với tay vào khay, quãng đường tay, thời gian tay chờ, tỷ lệ VA/NVA. Mura: độ lệch chuẩn và hệ số biến thiên của cycle time và từng bước giữa các video và giữa các người. Muri (bản tối thiểu): tỷ lệ thời gian cổ tay vượt ngưỡng góc. So với nhãn HA4M: sai số cycle time, sai số thời lượng từng bước, F1 phân đoạn. Đầu ra: `metrics.json`.

**Bước 6 — Phát hiện vấn đề bằng rule (khoảng 1 giờ).** Mỗi rule sinh một finding có cấu trúc: loại vấn đề, chỉ số làm bằng chứng, các timestamp, mức độ. Ví dụ: bước nào có thời gian vượt trung vị cộng 2 độ lệch chuẩn; số lần với tay vào khay nhiều hơn số linh kiện cần lấy; tay chờ quá X giây; thứ tự bước lệch so với SOP; một tay làm việc trong khi tay kia đứng yên lâu. Xếp hạng theo công thức đơn giản: thời gian ảnh hưởng mỗi chu kỳ nhân với độ tin cậy. Đầu ra: `findings.json`.

**Bước 7 — LLM giải thích (khoảng 1.5 giờ).** Đưa `findings.json` và các đoạn tri thức liên quan vào LLM, yêu cầu trả về JSON gồm: quan sát, giả thuyết nguyên nhân, đề xuất, ID nguồn trích dẫn. Sau đó **kiểm tra tự động bằng code**: mọi ID trích dẫn phải tồn tại trong `kaizen_kb.md` / `sop.md`, và mọi con số trong câu trả lời phải xuất hiện trong `findings.json`. Hai phép kiểm tra này biến nguyên tắc "LLM không bịa số" trong overview thành thứ đo được, và tỷ lệ vượt qua là một chỉ số đẹp cho slide.

**Bước 8 — Báo cáo (khoảng 1 giờ).** Sinh một file Markdown thay cho UI: bảng chỉ số, biểu đồ timeline dạng Gantt, Yamazumi chart, ảnh frame tại các timestamp bằng chứng có vẽ skeleton và zone, danh sách finding kèm giải thích của LLM. File này dùng luôn làm nội dung minh họa cho slide.

Tổng khoảng 10 giờ, khá sát. Nếu bị trễ, thứ tự cắt bớt nên là: Muri trước, rồi đến kiểm tra số liệu trong câu trả lời LLM, và cuối cùng mới dùng hẳn chế độ oracle cho bước 4.

## Cấu trúc thư mục gợi ý

```
kaizen-motion/
├── config/   station.yaml, sop.md, kaizen_kb.md
├── data/     raw/, keypoints/, labels/
├── src/      01_extract.py ... 08_report.py
├── outputs/  features, events, metrics, findings, report.md
└── run_all.sh   # chạy toàn luồng, có cờ --oracle
```

Mình có thể viết code cho các bước 2 đến 8 (bao gồm cả cờ `--oracle` và phần kiểm tra trích dẫn) để bạn chỉ cần cắm dữ liệu vào. Để khớp đúng định dạng, bạn gửi cho mình cấu trúc thư mục HA4M sau khi tải xong và vài dòng của file nhãn nhé.