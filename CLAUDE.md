# CLAUDE.md — AI Kaizen Motion Analyzer (HA4M pipeline)

## Bối cảnh dự án

Hackathon DENSO, thời gian rất ngắn. Mục tiêu: pipeline CLI (KHÔNG UI) phân tích video lắp ráp
→ keypoints → events → chỉ số Kaizen (Muda/Mura/Muri) → findings → LLM giải thích → báo cáo Markdown.

Dataset: HA4M — 41 người lắp bộ bánh răng hành tinh (Epicyclic Gear Train), 12 hành động/lần lắp,
quay bằng Azure Kinect. Nhãn hành động có sẵn → dùng làm ground truth và cho chế độ `--oracle`.

Nguyên tắc BẤT DI BẤT DỊCH:
1. Mọi con số (thời gian, quãng đường, góc, tỷ lệ) do code tính. LLM CHỈ giải thích, không sinh số.
2. Mỗi bước pipeline = 1 script đọc file vào / ghi file ra, chạy lại độc lập được.
3. Làm bản thô chạy thông toàn luồng TRƯỚC, tối ưu sau. Có `--oracle` để nửa sau không bị chặn bởi nửa trước.
4. Ưu tiên đơn giản: KHÔNG ByteTrack (1 người/trạm), KHÔNG vector DB (KB nhỏ, nhét thẳng vào prompt),
   KHÔNG train model. Chỉ pretrained + rule + thống kê.

## Cấu trúc dữ liệu HA4M (ĐÃ XÁC NHẬN từ máy người dùng)

```
data/ha4m/<SubjectID>/
├── Color/<TrialID>/            # CHUỖI ẢNH RGB (từng frame, KHÔNG phải .mp4)
├── ColorAligned/<TrialID>/     # RGB căn theo depth — không dùng
├── Depth/<TrialID>/            # ảnh depth — không dùng ở baseline
├── Infrared/<TrialID>/         # không dùng
├── PointClouds_DepthGeometry/<TrialID>/  # .ply — không dùng
├── Skeletons/<TrialID>/        # skeleton Azure Kinect (txt) — dùng ĐỐI CHỨNG, không phải nguồn chính
└── Labels.txt                  # nhãn hành động
```

QUAN TRỌNG — việc đầu tiên phải làm là viết `src/00_inspect.py`:
- In cây thư mục 2 cấp của 1 subject.
- In 10 dòng đầu `Labels.txt`, đoán schema (phân cách gì? cột gì? nhãn theo frame index hay timestamp?
  TrialID nằm ở đâu — trong tên file hay trong cột?).
- In tên 5 file đầu trong `Color/<TrialID>/` (suy ra pattern đánh số frame + đuôi ảnh) và 1 file
  trong `Skeletons/<TrialID>/` (in vài dòng đầu).
- Đếm số frame của 1 trial, đọc kích thước 1 ảnh.
- Ghi kết quả vào `outputs/inspect_report.md` và ĐỊNH NGHĨA lại parser trong `src/ha4m_io.py` theo thực tế.
KHÔNG hard-code schema Labels.txt theo giả định — paper nói 12 action class nhưng format file phải đọc từ dữ liệu thật.
Nếu không chắc fps: paper HA4M dùng Azure Kinect; tìm fps trong tài liệu kèm dataset, nếu không có thì
để `fps` là tham số config (mặc định 30) và ghi rõ giả định vào report.

## Module dùng chung

### `src/ha4m_io.py`
- `list_trials(root) -> [(subject, trial)]`
- `frames_iter(root, subject, trial) -> yield (frame_idx, bgr_image)` — đọc ảnh theo thứ tự SỐ trong tên file
  (sort số, không sort chuỗi — "10.jpg" phải sau "9.jpg").
- `load_labels(root, subject) -> DataFrame[trial, action_id, action_name, start_frame, end_frame]`
  (điều chỉnh theo schema thật từ 00_inspect).
- `load_kinect_skeleton(root, subject, trial) -> DataFrame` (chỉ cần cho bước đối chứng, làm sau cùng).

### `extract_keypoints.py` (ĐÃ CÓ SẴN, đã chạy được với .mp4)
Schema output .npz: `kpts (T,K,3) float32 [x_px,y_px,score], t (T,) giây, names (K,), meta json`.
MediaPipe = 75 điểm (33 pose + 21×2 tay, tay ghép theo cổ tay gần nhất). RTMW = 133 điểm.
CẦN SỬA: thêm `--frames-dir <dir>` nhận thư mục chuỗi ảnh (dùng `ha4m_io.frames_iter`),
`--fps <float>` bắt buộc khi dùng frames-dir (t = frame_idx/fps). Giữ nguyên schema output.
Tên file output: `<subject>_<trial>_<model>.npz` vào `data/keypoints/`.

## Pipeline (mỗi script có `--help`, nhận đường dẫn qua args, mặc định theo cấu trúc dưới)

```
config/station.yaml  config/sop.md  config/kaizen_kb.md
src/00_inspect.py → src/01_extract (wrapper gọi extract_keypoints cho N trial)
→ src/02_features.py → src/03_events.py → src/04_steps.py → src/05_metrics.py
→ src/06_findings.py → src/07_explain.py → src/08_report.py
run_all.sh [--oracle] [--subjects S1,S2] [--trials ...] [--model mediapipe|rtmw]
```

### 02_features.py — keypoints → features.parquet
Vào: .npz. Ra: `outputs/features/<subject>_<trial>.parquet`, mỗi dòng 1 frame:
`t, lw_x, lw_y, rw_x, rw_y` (cổ tay trái/phải đã xử lý), `lw_speed, rw_speed` (px/s),
`lw_zone, rw_zone` (tên zone hoặc "none"), score từng cổ tay, cờ `interp` (frame được nội suy).
Xử lý:
- Keypoint score < 0.3 → coi như NaN.
- Nội suy tuyến tính các đoạn NaN ≤ 15 frame; dài hơn thì giữ NaN và loại khỏi tính quãng đường.
- Lọc One Euro (lib `OneEuroFilter`, fallback: `scipy.signal.savgol_filter` window lẻ ~ 0.3s).
- Speed = khoảng cách giữa frame liên tiếp × fps, tính SAU lọc.
- Zone: point-in-polygon (shapely) với polygon trong station.yaml.
Check tự động in ra: % frame có cổ tay hợp lệ (cảnh báo nếu < 85%), đồ thị speed lưu
`outputs/features/<...>_speed.png` (matplotlib, 2 đường lw/rw).

### station.yaml (tạo bản mẫu, người dùng sẽ chỉnh polygon theo frame thật)
```yaml
fps: 30
px_per_cm: null          # null = báo quãng đường theo px, có giá trị thì đổi sang cm
moving_speed_px_s: 120    # ngưỡng phân biệt moving/still — TINH CHỈNH theo đồ thị speed
min_event_s: 0.25
zones:
  parts_bin_1: [[x,y],[x,y],[x,y],[x,y]]
  assembly_area: [...]
  done_area: [...]
steps:                    # 12 bước SOP theo Labels của HA4M — điền tên thật sau 00_inspect
  - {id: 1, name: "...", zone: parts_bin_1, va: false}   # va: value-added hay không
  - ...
cycle_time_target_s: null
```
Kèm tiện ích `src/tools/pick_zones.py`: mở 1 frame bằng OpenCV, click chuột để vẽ polygon,
in YAML ra stdout. (Cửa sổ OpenCV — chạy local, không cần trong CI.)

### 03_events.py — features → events.csv
Mỗi tay là 1 chuỗi trạng thái: `(zone, moving|still)`. Gộp frame liên tiếp cùng trạng thái thành event,
xóa event < min_event_s rồi gộp lại các event kề nhau cùng trạng thái.
Ra: `outputs/events/<...>.csv`: `start_s, end_s, hand, zone, motion, dur_s`.
Thêm cột suy diễn `event_type`: vào zone khay + still ngắn → "pick"; assembly_area + chuỗi still/moving xen kẽ
→ "assemble"; none + still > 1.5s → "idle"; none + moving → "transfer". Rule đơn giản, ghi rõ trong docstring.
Check: `--print-timeline` in timeline ASCII 80 cột để đối chiếu mắt với video.

### 04_steps.py — events (+nhãn) → steps.csv
Hai chế độ:
- `--oracle`: đọc Labels.txt qua ha4m_io, xuất `outputs/steps/<...>_oracle.csv`:
  `step_id, name, start_s, end_s, dur_s, source=oracle`.
- mặc định (predicted): ghép chuỗi event với chuỗi `steps` trong station.yaml.
  Baseline chấp nhận được: với mỗi bước có zone đặc trưng, lấy event "pick" đầu tiên ở zone đó làm mốc
  bắt đầu, kết thúc = mốc bước sau. Nếu không ghép được hết → các đoạn còn lại gán "unclassified".
  KHÔNG cố làm thông minh ở v1; DTW/ruptures là việc của v2, ghi TODO.
Ra thêm: so khớp predicted vs oracle nếu cả hai tồn tại → `outputs/steps/<...>_eval.json`:
sai số start/end từng bước (s), MAE thời lượng bước, sai số cycle time, F1@IoU0.5 theo bước.

### 05_metrics.py — steps + events + features → metrics.json (+ per-trial CSV)
Tính cho từng trial và tổng hợp theo subject + toàn bộ:
- Muda: cycle_time (end bước cuối − start bước đầu), dur từng bước, tổng idle, tổng transfer,
  số pick/chu kỳ theo zone, quãng đường mỗi cổ tay (px hoặc cm), %VA = Σdur(bước va=true)/cycle_time.
- Mura: mean/std/CV của cycle_time và từng bước, giữa các trial và giữa các subject;
  top bước có CV cao nhất.
- Muri (tối thiểu, chỉ khi model=mediapipe hoặc rtmw có đủ điểm tay): góc cổ tay 2D =
  góc giữa vector (elbow→wrist) và (wrist→middle_mcp của tay tương ứng); % thời gian lệch > 30°
  so với trung vị cá nhân. GHI RÕ trong output: "ước lượng 2D, chỉ mang tính sàng lọc".
- Eval (nếu có oracle): copy các số từ 04 eval vào.
Mỗi metric trong JSON kèm `unit` và `definition` ngắn — 07 sẽ đưa nguyên văn cho LLM.

### 06_findings.py — metrics → findings.json (rule-based, KHÔNG LLM)
Mỗi finding: `{id, type, severity, evidence: {metric, value, unit, trials, timestamps[]},
impact_s_per_cycle, confidence (high|med|low), rule}`.
Rule tối thiểu:
- R1 bước chậm bất thường: dur bước > median + 2·MAD (so giữa các trial cùng subject).
- R2 pick dư: số pick vào 1 zone trong 1 chu kỳ > số bước SOP cần zone đó.
- R3 idle dài: tổng idle > idle_threshold_s (config, mặc định 3s) trong 1 chu kỳ.
- R4 sai thứ tự / thiếu bước: chuỗi step_id ≠ chuỗi SOP (chỉ chạy với predicted; với oracle = sanity check).
- R5 mất cân bằng tay: 1 tay idle > 60% thời gian 1 bước trong khi tay kia hoạt động.
- R6 biến động lớn: CV cycle_time giữa trial của 1 subject > 0.25.
Xếp hạng: `impact_s_per_cycle × {high:1, med:0.6, low:0.3}` giảm dần.

### 07_explain.py — findings + KB → explained.json
- Đọc `config/kaizen_kb.md` và `config/sop.md`: các đoạn đánh dấu `[KB-xx]` / `[SOP-xx]`.
- Gọi LLM qua Gemini API (lib `google-genai`, model mặc định `gemini-3.8-flash`, env `GEMINI_API_KEY`;
  thêm cờ `--dry-run` sinh explanation giả để test luồng không cần key/mạng).
- Prompt: toàn bộ KB + findings JSON. Yêu cầu trả JSON THUẦN:
  `[{finding_id, observation, hypothesis, recommendation, citations: ["KB-03", ...]}]`.
  Ghi rõ trong prompt: observation chỉ được dùng số có trong findings; không được tạo số mới.
- VALIDATOR (bắt buộc, chạy bằng code sau khi nhận response):
  a) mọi citation tồn tại trong KB/SOP; b) mọi số thực/nguyên trong observation+hypothesis
  (regex các token số, bỏ qua số trong citation id) phải xuất hiện trong findings.json của finding đó
  (so khớp sau khi làm tròn 1 chữ số thập phân).
  Finding fail → đánh dấu `validation: failed`, giữ lại để báo cáo tỷ lệ pass (đây là metric cho slide).

### 08_report.py — tất cả → outputs/report.md (+ ảnh PNG)
- Bảng tổng: cycle time từng trial, %VA, idle, pick count, (nếu có) sai số vs oracle.
- Gantt timeline từng trial (matplotlib broken_barh, màu theo bước) → PNG.
- Yamazumi: bar chồng dur các bước theo trial, kẻ ngang cycle_time_target nếu có → PNG.
- Phân bố cycle time (hist) + bảng CV (Mura) → PNG.
- Mỗi finding: evidence + explanation + citations + trạng thái validation; chèn 1 frame bằng chứng
  (đọc lại ảnh tại timestamp, vẽ keypoints cổ tay + polygon zone bằng OpenCV) → PNG.
- Cuối report: mục "Giả định & giới hạn" tự sinh (fps giả định?, px_per_cm null?, oracle hay predicted?,
  Muri 2D...).

## run_all.sh
Chạy 00→08 theo thứ tự, dừng khi lỗi (`set -e`), nhận `--oracle`, `--model`, `--subjects`, `--trials`,
`--dry-run` (truyền xuống 07). In thời gian từng bước.

## Thứ tự thực hiện cho Claude Code (QUAN TRỌNG)
1. `00_inspect.py` + `ha4m_io.py` — chốt schema thật từ dữ liệu. Mọi thứ sau phụ thuộc bước này.
2. Sửa `extract_keypoints.py` thêm `--frames-dir/--fps`. Chạy thử 1 trial, xem preview.
3. `station.yaml` mẫu + `pick_zones.py`; nhờ người dùng vẽ zone trên 1 frame.
4. 02 → 03, kiểm tra timeline ASCII với video.
5. 04 với `--oracle` TRƯỚC → 05 → 06 → 07 `--dry-run` → 08. Lúc này toàn luồng đã thông.
6. Quay lại 04 predicted, tính eval; bật 07 thật khi có API key.
7. (Nếu còn thời gian) đối chứng skeleton Kinect; TODO v2: DTW golden cycle, ruptures, RTMW3D.

## Quy ước code
- Python 3.10+, chỉ dùng lib trong requirements.txt sẵn có (numpy<2, pandas, scipy, shapely,
  OneEuroFilter, matplotlib, opencv qua mediapipe, google-genai — đã thêm `google-genai` vào requirements).
- Type hint + docstring ngắn cho hàm public. Không class nếu hàm đủ. Log bằng print có prefix `[02]`...
- Mọi script: chạy được từ repo root, tạo thư mục output nếu chưa có, không ghi đè im lặng
  (thêm `--force`).
- Test nhanh: mỗi script có `if __name__ == "__main__"` + ít nhất 1 đường chạy với dữ liệu tổng hợp
  (sinh keypoints giả hình sin) trong `tests/synthetic.py` để CI không cần dataset.