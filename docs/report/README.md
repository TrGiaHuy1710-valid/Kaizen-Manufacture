# Báo cáo môn Học máy nâng cao (LaTeX)

`main.tex` dùng class `article`, 1 cột, A4 12pt, gồm bìa → Abstract → Mục lục → nội dung → References.
Biên dịch bằng **XeLaTeX** (tiếng Việt, font TeX Gyre Termes).
Thông tin bìa (trường, khoa, môn, **mã môn học**, giảng viên, đề tài, thời gian) sửa ở đầu `main.tex`; MSSV sửa trong khối
`titlepage`. Logo: đặt file `figures/logo.png` thì bìa tự chèn.

```bash
cd docs/report
latexmk            # → build/main.pdf  (latexmkrc đã chọn XeLaTeX + BibTeX)
latexmk -c         # dọn file trung gian
```

Trên Overleaf: upload cả thư mục `docs/report/`, rồi chọn Menu → Compiler → **XeLaTeX**.

## Cấu trúc

| File | Nội dung |
|---|---|
| `main.tex` | preamble, thông tin bìa, bìa, mục lục, thứ tự các mục |
| `sections/00_abstract.tex` | Abstract + Index Terms |
| `sections/01_introduction.tex` | 1. Introduction: bài toán, 4 nguyên tắc, đóng góp |
| `sections/02_related_work.tex` | 2. Overview – Related Work: Kaizen/IE, pose, action segmentation, dataset, LLM/RAG |
| `sections/03_method.tex` | 3. Method: bài toán, pose + làm sạch, events, phân đoạn bước, chỉ số 3M, R1–R8, LLM + validator |
| `sections/04_implementation.tex` | 4. Implementation: 4.1 Code Implementation · 4.2 Model Architectures · 4.3 Datasets · 4.4 Training (**draft**) |
| `sections/05_experiments.tex` | 5. Experiments and Results: số đã đo + các bảng MS-TCN/pose fine-tune (**draft**) |
| `sections/06_conclusion.tex` | 6. Conclusion + hướng phát triển + Acknowledgment |
| `references.bib` | 11 mục: dataset HA4M, cuộc thi DENSO, link tải model (MediaPipe, RTMW/rtmlib, RTMPose, e5), Chroma, Gemini API, MS-TCN, One Euro |
| `figures/architecture.tex` | sơ đồ kiến trúc tổng thể (TikZ, sửa trực tiếp được) |
| `figures/mstcn.tex` | sơ đồ MS-TCN (TikZ) |
| `figures/*.png` | ảnh demo, biểu đồ, chụp màn hình UI (xem bên dưới) |

## Ảnh trong `figures/`

| Ảnh | Nguồn | Đang dùng ở |
|---|---|---|
| `evidence_F01.png` | report_assets của run `20261016-055308_oracle_mediapipe_gemini_rag-report-2t` | Hình trạm + zone (IV-C) |
| `gantt_U001_V001.png`, `gantt_U001_V002.png` | cùng run | Gantt oracle vs pred (V-C) |
| `yamazumi.png` | cùng run | Yamazumi (V-D) |
| `frame_V001_step3.png` | `docs/slide/assets` | video overlay (V-D) |
| `ui2_brain.png` | `docs/slide/assets` | UI Kaizen Brain (IV-A) |
| `cycle_time_dist.png`, `evidence_F02/F03.png`, `speed_U001_V001.png`, `ui2_overlay_frame.png`, `ui2_3m_yamazumi.png`, `ui_video_compare.png`, `ui_timeline.png` | run trên / slide | chưa dùng, có sẵn để chèn thêm |

Thêm ảnh mới: chép vào `figures/`, rồi chèn bằng `\includegraphics[width=\columnwidth]{ten_anh.png}`
(`\graphicspath` đã trỏ sẵn tới `figures/`).

## Phần cần bổ sung

- `\todo{...}` hiện chữ đỏ **[TODO: …]** ở chỗ cần viết thêm; `\tbd` hiện **?** đỏ ở ô bảng chờ số liệu.
  Tìm nhanh: `grep -n 'todo\|tbd' sections/*.tex`.
- Số liệu train MS-TCN: lấy từ đầu ra của các lệnh trong [docs/training.md](../training.md):
  - `segmenter cv --folds 0` → bảng `tab:mstcn` (dòng LOSO) và `tab:seg` (dòng MS-TCN) trong `05_experiments.tex`;
  - `segmenter eval --out .../test_eval.json` → dòng "Test giữ riêng";
  - `models/segmenter/<name>/config.json` → `val_step_eval`, epoch tốt nhất;
  - `models/segmenter/<name>/train_log.csv` → đường cong train, lưu `figures/train_curve.png`
    rồi bỏ comment khối figure ở `04_implementation.tex`.
- Pose fine-tune: `python -m kaizen.train.pose eval --pose-model rtmw|rtmpose-custom ...` → bảng `tab:pck`.
- Claude / GPT: chạy `bash run_all.sh --provider anthropic|openai --tag ...` rồi `python src/tools/compare_runs.py --last 3`
  → bảng `tab:llm`.
- Điền mã môn học (`\mamonhoc`) và MSSV ở `main.tex`.

Mọi số liệu đã điền đều lấy từ artifact thật: run `20261016-055308_oracle_mediapipe_gemini_rag-report-2t`,
`outputs/kinect_compare_U001_V001.json`, `outputs/inspect_report.md` và lần chạy `kaizen.train.pose eval` (MediaPipe, U001_V002).
