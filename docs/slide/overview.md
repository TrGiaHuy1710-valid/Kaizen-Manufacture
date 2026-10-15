# AI Kaizen Copilot — Motion Analyzer · Tổng quan dự án

> Tài liệu tổng hợp cấu trúc logic của repo `KaizenMotionAnalyzer` tại thời điểm làm slide (13/10/2026).
> Số liệu trích từ run `20261013-003627_pred_mediapipe_dry_u001-2trials` (bước CV dự đoán) và run đối chứng
> `20261013-003618_oracle_mediapipe_dry_u001-2trials` (bước theo nhãn HA4M), cùng 2 trial U001_V001, U001_V002.

## 1. Bài toán và mục tiêu

- **Vấn đề:** bí quyết thao tác của người thợ giỏi (thứ tự động tác, nhịp, phối hợp hai tay, tư thế) là tri thức ngầm.
  Đo bằng bấm giờ thủ công thì tốn 2–4 giờ/trạm, phụ thuộc người đo, chỉ lấy được vài chu kỳ, nên Muda/Mura/Muri
  không được theo dõi liên tục và know-how mất khi người thợ nghỉ.
- **Mục tiêu:** pipeline CLI + dashboard biến video lắp ráp thành keypoints → sự kiện → bước SOP → chỉ số
  Muda/Mura/Muri → phát hiện (findings) xếp hạng → LLM giải thích có trích dẫn → báo cáo/dashboard để kỹ sư IE duyệt.
- **Danh mục:** A2 — tích lũy know-how chuyên gia cho AI học (theo `proposal.md`).

## 2. Nguyên tắc thiết kế (bất di bất dịch)

1. **Mọi con số do code tính; LLM chỉ giải thích.** Validator bằng code bác bỏ câu trả lời có số hoặc trích dẫn không có thật.
2. **Mỗi bước pipeline là một script đọc file → ghi file**, chạy lại độc lập được.
3. **Chạy thông toàn luồng trước, tối ưu sau.** Chế độ `--oracle` dùng nhãn chuyên gia để nửa sau (chỉ số, findings,
   LLM, báo cáo) không bị chặn bởi sai số CV; sai khác oracle vs pred chính là sai số của phần CV.
4. **Đơn giản:** pretrained model + rule + thống kê; không train model, không tracking nhiều người, không vector DB.

## 3. Dữ liệu

| Thành phần | Mô tả |
|---|---|
| Dataset | HA4M (Cicirelli et al., Sci Data 2022): 41 người lắp bộ bánh răng hành tinh, 12 hành động, Azure Kinect |
| Cấu trúc thật | `IDU<subject>V<trial>/Color/<serial>/FrameID…_DeviceTimeStamp…us.png` (2048×1536, 30 fps), `Labels.txt` theo frame (`frame action_id repetition`), Skeletons TSV 32 joint |
| Đã dùng | U001_V001 (2427 frame, 80.9 s), U001_V002 (2655 frame) |
| Cấu hình trạm | `config/station.yaml`: fps, ROI, 11 zone (vùng cổ tay khi với vào khay), 12 bước SOP ↔ khay, cờ VA |
| Tri thức | `config/kaizen_kb.md` (16 đoạn `[KB-xx]`: 3M, 7 lãng phí, Barnes, ECRS, Standardized Work, Yamazumi…), `config/sop.md` (16 đoạn `[SOP-xx]`, SOP mô phỏng 12 bước) |

## 4. Pipeline 00 → 08

| Bước | Script | Vào → Ra | Logic chính |
|---|---|---|---|
| 00 | `00_inspect.py` | dataset → `outputs/inspect_report.md` | Chốt schema thật (nhãn theo frame, fps 30 xác nhận bằng timestamp) |
| 01 | `01_extract.py` + `extract_keypoints.py` | ảnh Color → `data/keypoints/<stem>_<model>.npz` | MediaPipe Pose (+ Hand dò quanh cổ tay), ROI bỏ tay người gán nhãn; batch, bỏ qua trial đã có |
| 02 | `02_features.py` | npz → `features/*.parquet` | score < 0.3 → NaN, loại điểm gai, nội suy ≤ 15 frame, lọc One Euro, tốc độ px/s, zone (shapely) |
| 03 | `03_events.py` | features → `events/*.csv` | Trạng thái (zone, moving/still) mỗi tay → event; rule pick/reach/assemble/transfer/idle/lost |
| 04 | `04_steps.py` | events (+ nhãn) → `steps/*_{oracle,pred}.csv`, `*_eval.json` | Oracle từ Labels; predicted = mốc pick đầu tiên ở khay của bước, lùi về lúc tay rời vùng lắp; eval MAE, F1@IoU0.5 |
| 05 | `05_metrics.py` | steps + events + features → `metrics/metrics.json` | Muda (cycle time, dur bước, idle, pick, quãng đường, %VA), Mura (mean/std/CV), Muri (góc cổ tay 2D) — mỗi chỉ số kèm unit + definition |
| 06 | `06_findings.py` | metrics → `findings/findings.json` | Rule R1–R7 (bước chậm, pick dư, idle dài, sai thứ tự, lệch tay, CV cao, cổ tay); xếp hạng impact × confidence |
| 07 | `07_explain.py` | findings + KB/SOP → `explained/explained.json` | LLM đa provider (Gemini / Anthropic / dry-run) → JSON; validator whitelist số + chuẩn hoá citation; retry 1 lần có phản hồi |
| 08 | `08_report.py` | tất cả → `report/report.md` + PNG | Bảng tổng, Gantt, Yamazumi, Mura, Muri, findings + ảnh bằng chứng, mục giả định & giới hạn tự sinh |

`run_all.sh` chạy 00→08 vào **một run mới** `outputs/runs/<thời điểm>_<pred|oracle>_<pose>_<llm>[_tag]/`
có `manifest.json` (lệnh gốc, git hash, data_root, bản chụp `station.yaml`, labels_available từng trial…).
`src/tools/compare_runs.py` so sánh các run (oracle vs pred, Gemini vs Anthropic…).

## 5. Lớp LLM và kiểm chứng

- **Prompt:** toàn bộ KB + SOP + findings JSON; yêu cầu JSON `[{finding_id, observation, hypothesis, recommendation, citations}]`.
- **Validator** (`src/llm_validate.py`): citation phải tồn tại (chấp nhận `kb13`, `[KB-13]`…); mọi số trong 3 trường phải
  thuộc whitelist của finding (giá trị, số trong tên trường như `dev_p90_deg` → 90, số phổ quát, bước 1–12).
- **Retry có phản hồi:** fail → gửi lại lỗi cụ thể (`số lạ: ['8.2']`), lưu cả 2 lần thử; báo pass rate lần 1 và sau retry.
- **Bài học thật:** lần chạy Gemini đầu bị đánh fail oan vì "phân vị 90" → vá whitelist; lần sau Gemini 3.8 Flash đạt ngay lần 1.

## 6. Giao diện demo (`app/`, Streamlit)

8 tab chỉ **đọc** artifact của run: Tổng quan · Video & Timeline (video overlay H.264, Gantt oracle vs pred, xem frame)
· So sánh video (2 trial cạnh nhau, nhảy tới cùng bước SOP) · Muda (Yamazumi) · Mura & Muri · Findings & Copilot
(chip trích dẫn mở nguyên văn KB/SOP) · So sánh run · Chạy pipeline. Deep link `?run=<id>&trial=<t>`.

## 7. Kết quả đo được (HA4M, U001)

| Hạng mục | V001 | V002 |
|---|---|---|
| Cycle time theo nhãn (oracle) | 77.9 s | 84.3 s |
| Cycle time CV dự đoán (pred) | 78.5 s (sai số 0.6 s) | 86.57 s (sai số 2.27 s) |
| MAE thời lượng bước | 0.62 s | 0.92 s |
| F1@IoU0.5 phân đoạn bước | 0.75 | 0.714 |
| Tỷ lệ phát hiện người (detect_rate) | 0.953 | 0.961 |
| %VA theo nhãn | 82.1 % | 85.3 % |

- **Mura:** CV cycle time 0.056 (oracle) / 0.069 (pred); bước biến động nhất (pred): B10 CV 0.28, B12 CV 0.26.
- **Đối chứng cổ tay với Azure Kinect (V001):** tay phải lệch trung vị 24.5 px, 95.1 % frame < 50 px; tay trái kém tin
  cậy hơn (P90 151.6 px, hay bị thân che).
- **Findings (run pred):** F01 R2 lấy dư 1 lần ở khay vít/lục giác (V002: 4 lần, SOP cần 3); F02 R4 hai bước chưa gán
  được ở V002 (3.2, 5 — giới hạn CV); F03 R7 cổ tay trái lệch > 30° trong 23.1 % thời gian (độ phủ chỉ 26.9 %, sàng lọc 2D).
- **Tốc độ:** trích keypoint ≈ 7.2–7.6 fps trên CPU (~6 phút/video); 02→08 ≈ 8–9 s cho 2 trial; render video overlay ≈ 2 phút.

## 8. Giới hạn đã biết

- Zone vẽ tay trên 1 trạm; bước không có khay riêng (bước 9) không tách được ở chế độ pred.
- Quãng đường theo pixel (chưa hiệu chuẩn px→cm); Muri là góc 2D, chỉ sàng lọc.
- Mới 2 trial của 1 subject: R1 (median + 2·MAD) cần ≥ 3 trial; Mura giữa người chưa có ý nghĩa thống kê.
- Ngưỡng moving/idle chỉnh tay theo đồ thị tốc độ, chưa hiệu chuẩn hệ thống.

## 9. Kiểm thử và tái lập

- `python -m pytest` — 30 test: validator, compare_runs, smoke test 8 tab UI, trình phát so sánh video, pipeline tổng hợp.
- Tái lập demo: xem `README.md` (`bash run_all.sh … --dry-run`, `python src/tools/render_demo.py …`, `streamlit run app/ui.py`).
