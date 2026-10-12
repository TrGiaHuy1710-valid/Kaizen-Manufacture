# Kaizen Knowledge Base (tóm tắt từ nguồn công khai)

> Mỗi đoạn có ID `[KB-xx]`. Tóm tắt từ: Barnes, *Motion and Time Study* (nguyên tắc tiết kiệm động tác);
> Ohno, *Toyota Production System* (7 lãng phí, 3M); tài liệu phổ biến về ECRS, Standardized Work, Yamazumi.
> Đây là kiến thức chung, không chứa số liệu của trạm — mọi số liệu phải lấy từ findings.

[KB-01] 3M — Muda (lãng phí: hoạt động không tạo giá trị), Mura (không đều: biến động thời gian, khối lượng
công việc), Muri (quá tải: tư thế, lực, nhịp vượt khả năng con người). Ba loại liên quan nhau: Mura và Muri
thường sinh ra Muda. Cải tiến nên nhìn cả ba thay vì chỉ cắt thời gian.

[KB-02] Bảy lãng phí (TPS): sản xuất thừa, chờ đợi, vận chuyển, gia công thừa, tồn kho, thao tác thừa
(motion), sản phẩm lỗi. Tại trạm lắp tay, hai loại thường gặp nhất là thao tác thừa và chờ đợi.

[KB-03] Lãng phí chờ đợi (waiting/idle): tay hoặc người đứng yên không thao tác. Nguyên nhân thường gặp:
chờ linh kiện, phân vân/tìm kiếm, không rõ bước tiếp theo, mất cân bằng giữa hai tay. Idle ngắn giữa các
thao tác có thể là nghỉ cần thiết — cần kỹ sư xác nhận trước khi kết luận là lãng phí.

[KB-04] Lãng phí thao tác (motion): với tay xa, lấy lại nhiều lần, chuyển linh kiện qua tay, tìm kiếm linh kiện.
Mỗi lần lấy dư so với số linh kiện cần là một chu trình với–lấy–mang không tạo giá trị.

[KB-05] Barnes — sử dụng cơ thể: hai tay nên bắt đầu và kết thúc chuyển động cùng lúc; hai tay không nên
cùng nghỉ trừ lúc nghỉ chính thức; chuyển động của hai tay nên đối xứng, ngược chiều. Một tay giữ chi tiết
lâu trong khi tay kia làm là dấu hiệu nên dùng đồ gá (fixture) để giải phóng tay giữ.

[KB-06] Barnes — bố trí chỗ làm việc: dụng cụ, vật liệu có vị trí cố định; đặt trong vùng với tay thông
thường (normal working area) và gần điểm sử dụng; sắp xếp theo thứ tự sử dụng để chuỗi động tác ngắn nhất.

[KB-07] Barnes — dùng khay/thùng cấp liệu nạp bằng trọng lực, khay nghiêng, ngăn chia để linh kiện tự dồn
về điểm lấy, giảm thời gian tìm và lấy. Linh kiện nhỏ, nhiều cái (vít, ổ bi) hưởng lợi nhiều nhất.

[KB-08] Barnes — dụng cụ: kết hợp hai dụng cụ thành một khi được; đặt sẵn dụng cụ (pre-position) ở tư
thế sẵn sàng dùng; dùng đồ gá để giữ chi tiết thay cho tay.

[KB-09] ECRS — thứ tự ưu tiên cải tiến: Eliminate (loại bỏ bước/động tác không cần), Combine (gộp), 
Rearrange (sắp xếp lại thứ tự/bố trí), Simplify (đơn giản hoá). Luôn hỏi "có bỏ được không" trước "làm nhanh hơn".

[KB-10] Standardized Work: mỗi bước có trình tự, thời gian chuẩn và tồn kho chuẩn. Biến động lớn giữa các
lần lặp (CV cao) là dấu hiệu bước chưa được chuẩn hoá hoặc khó (Mura); nên quan sát lần chậm nhất để tìm nguyên nhân.

[KB-11] Yamazumi (biểu đồ chồng thời gian bước): so sánh tổng thời gian chu kỳ với takt/mục tiêu; bước cao
bất thường là ứng viên ưu tiên cải tiến hoặc chia lại công việc.

[KB-12] Phân tích biến động: dùng median và MAD (độ lệch tuyệt đối trung vị) thay cho mean/std khi mẫu nhỏ
hoặc có ngoại lai. Một lần thực hiện vượt median + 2·MAD là ứng viên "chậm bất thường" cần xem lại video.

[KB-13] Muri — tư thế cổ tay: làm việc lâu với cổ tay gập/duỗi hoặc lệch mạnh khỏi trung tính làm tăng nguy
cơ rối loạn cơ xương (vd hội chứng ống cổ tay). Biện pháp: hạ/nghiêng bề mặt làm việc, đổi hướng chi tiết,
dụng cụ cán cong. Đo góc 2D từ camera chỉ mang tính sàng lọc, cần xác nhận bằng đánh giá ergonomic chuẩn.

[KB-14] Thứ tự lắp sai hoặc bỏ bước: dùng poka-yoke (chống sai) như đồ gá chỉ cho phép một hướng lắp,
khay theo thứ tự bước, đèn/cảm biến chọn linh kiện (pick-to-light).

[KB-15] Cân bằng tải hai tay: khi một tay rảnh phần lớn thời gian, xem xét chia lại thao tác (tay kia lấy
linh kiện kế tiếp trong lúc tay này lắp — "pre-positioning"), hoặc dùng đồ gá để cả hai tay cùng lắp.

[KB-16] Giới hạn của phân tích video: đứng yên không đồng nghĩa lãng phí (có thể đang kiểm tra bằng mắt);
pose 2D bị che khuất cho kết quả sai; mọi kết luận cần kỹ sư IE xem lại đoạn video bằng chứng.
