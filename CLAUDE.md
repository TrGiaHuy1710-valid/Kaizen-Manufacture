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
4. Ưu tiên đơn giản: KHÔNG ByteTrack (1 người/trạm), KHÔNG train model. Chỉ pretrained + rule + thống kê.
   Đường mặc định của 07: KB nhỏ nhét thẳng vào prompt. RAG (Chroma local, không server) chỉ là chế độ tuỳ chọn
   (`07 --use-rag`, API) cho tài liệu người dùng tải lên — citation là chunk_id, validator kiểm tồn tại trong store.

## Package `kaizen/` (code thật nằm ở đây — scripts src/0X là wrapper mỏng, CLI + schema artifact không đổi)
- `kaizen/skeleton`: Source (`open_source("ha4m://…"|"video://…"|"frames://…?fps=")`, registry `SOURCES`),
  PoseBackend (`POSE_BACKENDS`), `extract`, `compute_features` (02), `detect_events` (03), ha4m_io, overlay.
- `kaizen/analyzer`: THUẦN tính toán, import không cần key/mạng — `align_steps`/`evaluate` (04), `compute_metrics`/
  `build_metrics_report` (05), `detect_findings` + `RULES: list[Rule]` (06; thêm rule = thêm 1 phần tử), report (08).
- `kaizen/llm`: `get_provider(name)` (anthropic | gemini | openai | dry-run; env LLM_PROVIDER / LLM_MODEL),
  `parsing.py` (parse lượng thứ chung), `prompts.py` (template có version EXPLAIN_V1, EXPLAIN_RAG_V1, EXTRACT_STEPS_V1…),
  `validator.py` (whitelist số + citation tĩnh KB/SOP hoặc chunk_id động), `explain.py`.
- `kaizen/rag`: ingest (PDF/DOCX/MD/TXT) → chunking heading-aware 300–600 token → embed (`KAIZEN_EMBEDDER`: e5 | hash…)
  → `DocumentStore` (data/rag_store/) → `extract_steps` / `to_station`. `kaizen/api/app.py`: FastAPI.
- `kaizen/core`: run_io, config, `schemas.py` (pydantic cho mọi artifact; dump `exclude_unset` → byte-giống file cũ).
- `src/{ha4m_io,run_io,overlay,llm_validate,llm_providers}.py` là shim alias (`sys.modules`) cho app/ và tools cũ.
- Refactor phải giữ artifact byte-giống: so `metrics.json`/`findings.json`… giữa code cũ và mới trên cùng input.

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
run_all.sh [--oracle] [--subjects S1,S2|all] [--trials ...|all] [--model mediapipe|rtmw] [--provider ...] [--tag ...]
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
- Đa provider qua 1 interface (`kaizen/llm/providers.py`): `--provider anthropic|gemini|openai|dry-run`
  (mặc định env `LLM_PROVIDER`, không có thì gemini; model `--llm-model` > env `LLM_MODEL` > env `<PROVIDER>_MODEL` >
  mặc định). anthropic: `claude-sonnet-4-6`, env `ANTHROPIC_API_KEY`; gemini: SDK `google-genai`, `gemini-3.8-flash`,
  env `GEMINI_API_KEY`; openai: SDK `openai`, `gpt-4.1-mini`, env `OPENAI_API_KEY` (json_object mode).
  Thiếu key → `MissingAPIKeyError` ngay khi khởi tạo. dry-run sinh explanation giả để test luồng không cần key/mạng.
- Parse lượng thứ (`kaizen/llm/parsing.py`, chung mọi provider): bỏ code fence/lời dẫn, bọc object đơn, key thiếu → chuỗi rỗng +
  `format_warning`; không parse được → hỏi lại 1 lần "chỉ JSON array"; vẫn hỏng → `validation: parse_failed`
  (pipeline không dừng).
- Prompt: toàn bộ KB + findings JSON. Yêu cầu trả JSON THUẦN:
  `[{finding_id, observation, hypothesis, recommendation, citations: ["KB-03", ...]}]`.
  Ghi rõ trong prompt: observation chỉ được dùng số có trong findings; không được tạo số mới.
- VALIDATOR (bắt buộc, chạy bằng code sau khi nhận response — `kaizen/llm/validator.py`):
  a) citation chuẩn hoá (`KB13`/`kb-13`/`[KB-13]` → `KB-13`) phải tồn tại trong KB/SOP;
  b) mọi token số trong observation+hypothesis+recommendation phải thuộc WHITELIST của finding: giá trị số
  (làm tròn 0 và 1 chữ số, cả |x|), số trong tên trường/chuỗi (`dev_p90_deg` → 90, "lệch > 30°" → 30,
  `U001_V001` → 1), số phổ quát 0/1/2/3/100, số bước 1–12. Số dính chữ phía trước (P90, U001, F01) không tính.
  Thông báo lỗi chỉ liệt kê phần sai thật (`số lạ: ['8.2']`, `citation không tồn tại: ['KB-99']`).
- Fail → gọi lại 1 lần kèm thông báo lỗi validator. explained.json lưu `{attempts, final, validation:
  passed|failed_after_retry|parse_failed}`; báo `pass_rate` (sau retry) và `pass_rate_first_attempt` (metric cho slide).
- Test: `python -m pytest tests/test_validator.py` (không gọi mạng).

### 08_report.py — tất cả → <run>/report/report.md (+ ảnh PNG)
- Bảng tổng: cycle time từng trial, %VA, idle, pick count, (nếu có) sai số vs oracle.
- Gantt timeline từng trial (matplotlib broken_barh, màu theo bước) → PNG.
- Yamazumi: bar chồng dur các bước theo trial, kẻ ngang cycle_time_target nếu có → PNG.
- Phân bố cycle time (hist) + bảng CV (Mura) → PNG.
- Mỗi finding: evidence + explanation + citations + trạng thái validation; chèn 1 frame bằng chứng
  (đọc lại ảnh tại timestamp, vẽ keypoints cổ tay + polygon zone bằng OpenCV) → PNG.
- Cuối report: mục "Giả định & giới hạn" tự sinh (fps giả định?, px_per_cm null?, oracle hay predicted?,
  Muri 2D...).

## run_all.sh
Chạy 00→08 theo thứ tự vào MỘT run mới, dừng khi lỗi (`set -e`). Cờ: `--oracle`, `--model`, `--subjects`,
`--trials` (`all` = tất cả), `--data-root a,b`, `--provider anthropic|gemini|dry-run`, `--llm-model`, `--dry-run`,
`--tag <text>` (nối vào run_id), `--run <id>` (ghi vào run có sẵn), `--force` (chạy lại 00/01). In thời gian từng bước
và đường dẫn report cuối.

## Output layout (theo run — `src/run_io.py`)
Mỗi lần chạy ra một thư mục riêng, không ghi đè run cũ:
```
outputs/
├── inspect_report.md                      # 00 — schema dữ liệu thật (không thuộc run)
├── kinect_compare_<stem>.json             # tools/compare_kinect.py
└── runs/
    ├── LATEST                             # file text: run_id gần nhất (không dùng symlink → chạy được Windows)
    └── <YYYYmmdd-HHMMSS>_<pred|oracle>_<posemodel>_<anthropic|gemini|dry>[_<tag>]/   # trùng → -2, -3
        ├── manifest.json                  # thời điểm, git hash, lệnh gốc, data_root, lựa chọn subject/trial,
        │                                  # pose meta + labels_available từng trial, provider/model LLM, argv từng stage
        ├── station.yaml                   # bản chụp config dùng cho run (02→08 đọc bản này nếu không truyền --config)
        ├── features/ events/ steps/ metrics/ findings/ explained/
        └── report/report.md + report/report_assets/*.png
data/keypoints/<stem>_<model>.npz          # KHÔNG thuộc run — tái dùng giữa các run; manifest.json ghi detect_rate, nhãn
```
- Mọi script 02→08 nhận `--run <run_id|latest>` (mặc định latest). Chưa có `outputs/runs/` → cảnh báo và dùng
  đường dẫn phẳng cũ `outputs/<stage>/` (tương thích ngược). Các cờ `--*-dir/--out` tường minh vẫn thắng.
- Tạo run thủ công: `python src/run_io.py new --source pred --pose mediapipe --llm gemini --tag demo` (in run_id).
- So sánh: `python src/tools/compare_runs.py --list` · `python src/tools/compare_runs.py <run1> <run2>` ·
  `python src/tools/compare_runs.py --last 2` → bảng Markdown cycle time, %VA, idle, F1@IoU0.5, MAE, pass rate LLM.

## UI demo (`app/`, Streamlit)
- `streamlit run app/ui.py`: dashboard 8 tab (Tổng quan · Video & Timeline · So sánh video · Muda · Mura & Muri ·
  Findings & Copilot · So sánh run · Chạy pipeline). Deep link: `?run=<run_id>&trial=<trial>`.
- So sánh video (`app/video_compare.py`): 2 video overlay cạnh nhau, phát/tua/tốc độ chung, "Nhảy tới bước" đưa mỗi
  video tới đầu bước theo mốc của chính trial (mặc định căn theo nhãn oracle), khoá đồng bộ giữ độ lệch B − A;
  bên nào thiếu bước → báo và tắt khoá. UI CHỈ ĐỌC artifact của run qua `app/data_access.py` (cache theo mtime) — không tính
  lại chỉ số; artifact thiếu → cảnh báo, không crash. Màu bước dùng chung `src/overlay.step_color` (Gantt, Yamazumi, video).
- Video overlay: `python src/tools/render_demo.py --run <id|latest> --trial U001_V001` → `<run>/report/demo_<trial>.mp4`
  (H.264 + faststart qua imageio-ffmpeg; mp4v của OpenCV KHÔNG phát được trên trình duyệt).
- So sánh run dùng chung hàm `compare_runs.compare()` với CLI.

## Nạp dữ liệu mới (batch)
- `python src/01_extract.py --data-root data/ha4m_new --subjects all [--trials all] [--limit N] [--force]`:
  quét phòng thủ (thư mục có `Color*` ở độ sâu ≤ 2; tên `IDU001V001` / `U001_V001` / `U001/V001` / `U001`→V001;
  `Labels*.txt` không phân biệt hoa thường), bỏ qua trial đã có .npz, in bảng tiến độ + trial detect_rate < 0.85,
  gợi ý lệnh run_all tiếp theo. Thiếu nhãn → vẫn trích, `labels_available=false`, 04 oracle/eval bỏ qua kèm cảnh báo.
- Nhiều nguồn cùng lúc: `bash run_all.sh --data-root data/ha4m,data/ha4m_new ...` (02 chỉ lấy .npz thuộc data_root).

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
  OneEuroFilter, matplotlib, opencv qua mediapipe, anthropic, google-genai, openai; RAG/API: fastapi, uvicorn,
  python-multipart, pymupdf, chromadb, sentence-transformers (torch CPU); pytest cho tests). Không Docker, không DB server.
- Type hint + docstring ngắn cho hàm public. Không class nếu hàm đủ. Log bằng print có prefix `[02]`...
- Mọi script: chạy được từ repo root, tạo thư mục output nếu chưa có, không ghi đè im lặng
  (thêm `--force`).
- Test nhanh: mỗi script có `if __name__ == "__main__"` + ít nhất 1 đường chạy với dữ liệu tổng hợp
  (sinh keypoints giả hình sin) trong `tests/synthetic.py` để CI không cần dataset.