# Kaizen-Manufacture

AI Kaizen Motion Analyzer cho dataset HA4M: video lắp ráp → keypoints → events → chỉ số Kaizen
(Muda/Mura/Muri) → findings → LLM giải thích (có validator) → báo cáo. Chi tiết kiến trúc: [CLAUDE.md](CLAUDE.md).

## Chạy demo

```bash
conda activate kaizen-py311 && pip install -r requirements.txt
# 1) Chuẩn bị run (lần đầu trích keypoints ~6 phút/video; --dry-run = không cần API key)
bash run_all.sh --subjects U001 --trials all --oracle --dry-run --tag demo     # số theo nhãn HA4M
bash run_all.sh --subjects U001 --trials all --dry-run --tag demo              # số theo CV dự đoán (+ sai số vs nhãn)
#    LLM thật: --provider gemini | anthropic | openai (key qua env GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY;
#    model: --llm-model hoặc env LLM_MODEL). So sánh pass-rate giữa provider:
#    for p in openai gemini anthropic; do bash run_all.sh --provider $p --tag $p; done; python src/tools/compare_runs.py --last 3
# 2) Chuẩn bị dự án demo (mỗi video = 1 dự án; thiếu gì tự chạy bù: run, explained, video overlay oracle)
python -m kaizen.core.projects prepare-demo --trials U001_V001,U001_V002     # in bảng READY/MISSING
python -m kaizen.core.projects adopt <run_id>                               # bọc run cũ outputs/runs/ thành dự án
# 3) Mở UI 4 trang (server remote: forward cổng 8501 trong VS Code)
streamlit run app/ui.py                  # 📁 Dự án · 🎬 Chi tiết · 📚 Tài liệu · 📊 So sánh 3M
KAIZEN_DEMO=1 streamlit run app/ui.py    # demo mode: dự án tính sẵn → tiến độ mô phỏng, có badge báo rõ
```

## Cấu trúc package

Scripts `src/00…08_*.py`, `extract_keypoints.py`, `run_all.sh` giữ nguyên CLI — là wrapper mỏng gọi vào `kaizen/`
(schema artifact không đổi; `src/{ha4m_io,run_io,overlay,common,llm_*}.py` là shim tương thích cho UI/tools cũ).

| Module | Vai trò | API chính |
|---|---|---|
| `kaizen/skeleton` | Tầng 1: nguồn video → pose → keypoints → features → events | `open_source("ha4m://…" \| "video://…" \| "frames://…?fps=")`, `extract`, `compute_features`, `detect_events` |
| `kaizen/analyzer` | Tầng 2 định lượng, KHÔNG LLM/mạng/key | `align_steps`, `evaluate`, `compute_metrics`, `detect_findings` (`RULES` R1–R7) |
| `kaizen/llm` | provider + prompt có version + validator | `get_provider(name)`, `provider.complete(...)`, `explain(findings, kb, provider, retriever=None)` |
| `kaizen/rag` | tài liệu: parse → chunk → embed → Chroma | `DocumentStore.ingest/search`, `extract_steps`, `to_station` |
| `kaizen/core` | run_io, config, schema pydantic mọi artifact | `run_io`, `load_config`, `core.schemas` |
| `kaizen/api` | FastAPI service | `uvicorn kaizen.api.app:app` |

```bash
python -m kaizen.skeleton extract "video://data/raw/test.mp4" --model mediapipe     # → data/keypoints/test_mediapipe.npz
python -m kaizen.skeleton info "ha4m://data/ha4m?subject=U001&trial=V001"
```

## RAG service

```
upload → chunks (heading-aware, trang + chunk_id) → (query | extract-steps) → explain có trích dẫn chunk_id
```

```bash
uvicorn kaizen.api.app:app --port 8000          # chạy từ repo root; store data/rag_store/, embedder e5 (lần đầu tải ~470 MB)
#   KAIZEN_EMBEDDER=hash → không tải model (CI/offline) · KAIZEN_RAG_STORE=<dir> → đổi chỗ lưu · HF_HOME → chỗ cache model
#   Docs tương tác: http://localhost:8000/docs · CORS mở cho localhost (Streamlit gọi được)
python tests/fixtures/make_sop_pdf.py           # sinh SOP mẫu tests/fixtures/sop_sample.pdf (2 trang, 12 bước, 1 bảng)

curl -F "file=@tests/fixtures/sop_sample.pdf" localhost:8000/documents
#   → {"doc_id": "DOC-a5b86e", "pages": 2, "chunks": 3, "needs_ocr_pages": [], ...}
curl -X POST localhost:8000/rag/query -H 'Content-Type: application/json' \
     -d '{"query": "lắp bánh răng hành tinh", "k": 3}'
curl -X POST localhost:8000/rag/extract-steps -H 'Content-Type: application/json' \
     -d '{"doc_id": "DOC-a5b86e", "provider": "dry-run"}'                       # hoặc gemini | anthropic | openai
curl -X POST localhost:8000/analyze/explain -H 'Content-Type: application/json' \
     -d '{"run_id": "latest", "provider": "dry-run", "use_rag": true}'
```

Khác: `GET /documents`, `GET /runs`, `GET /runs/{id}/metrics|findings|explained`, `GET /health`. Thiếu key → HTTP 400 nêu tên biến env.

- **Trích bước chuẩn** → `outputs/extractions/<doc_id>.json` (mỗi bước có trang nguồn). `dry-run` trích bằng code
  (dòng "Bước N:" + bảng); provider thật dùng LLM nhưng code hậu kiểm: chunk nguồn phải thuộc tài liệu, `std_time_s`
  phải có trong văn bản (không thì bỏ). Bản nháp cấu hình trạm cho kỹ sư sửa:
  `python -m kaizen.rag to-station DOC-a5b86e` → `outputs/extractions/DOC-a5b86e_station.yaml`.
- **Giải thích có trích dẫn**: 07 tự bật RAG khi store đã có chunk (`--no-rag` để tắt) — `kaizen_kb.md` + `sop.md` được
  ingest như tài liệu; citation = chunk_id (`DOC-xxxxxx:pN:cM`), validator kiểm chunk_id có trong store + định dạng
  (prompt `explain_v2`: quan sát ≤2 câu, giả thuyết 1 câu, đề xuất ≤2 câu, ECRS, "Đo lại" ≤12 từ (≤20 tiếng)).
- **Report truy vết theo tài liệu** (`kaizen/rag/report_queries.py`): 06 lấy TG chuẩn từng bước (bảng "TG chuẩn" của
  work instruction, khớp nguyên văn tên bước trong `station.yaml`; `std_time_s` ghi tay trong station.yaml được ưu tiên)
  và ngưỡng ("chênh lệch … trên 30%", "không gập cổ tay quá 30°") kèm citation → `findings.json["standards"]`; R8 bắn
  khi |lệch| > ngưỡng (chạy được với 1 video). Report: tóm tắt ≤ 6 dòng, bảng bước có `TG chuẩn · Lệch % · Nguồn chuẩn`,
  mục 3M tổng hợp hoãn khi < 2 video, bảng "Tài liệu tham chiếu" khớp 2 chiều với mọi nhãn `[WI-EGT-001 · tr.1]`
  (08 in `truy vết trích dẫn: khớp 2 chiều ✓`). Store rỗng → citation KB-xx tĩnh + ghi chú, report vẫn build.
  ```bash
  python -m kaizen.rag ingest docs/WI-EGT-001_work_instruction.pdf      # 1 lần; sau đó run_all tự dùng
  bash run_all.sh --oracle --subjects U001 --trials V001                # key/LLM_PROVIDER đọc từ .env
  ```
- CLI: `python -m kaizen.rag ingest <files> | list | query "<q>" | extract-steps <doc_id> | to-station <doc_id>`.
- PDF scan (ít text, ảnh phủ trang) → `needs_ocr_pages`, bỏ qua ở v1 (TODO OCR).

## Training (tuỳ chọn, tự chạy)

Model phân đoạn bước MS-TCN (học từ nhãn HA4M) và fine-tune pose RTMPose (nhãn giả từ Kinect) — toàn bộ lệnh, thời gian
ước tính và cách so với baseline: [docs/training.md](docs/training.md). Pipeline không tự train; chưa có model thì vẫn
chạy đường rule mặc định.

```bash
python -m kaizen.train.segmenter status                                   # dữ liệu đủ chưa
python -m kaizen.train.segmenter train --name mstcn_v1 --test-subjects U037,U038,U039,U040,U041
bash run_all.sh --segmenter mstcn --segmenter-model models/segmenter/mstcn_v1 --subjects U037 --trials all --dry-run
```

Kiểm thử: `python -m pytest` (validator, providers LLM với FakeProvider, analyzer R1–R8, adapter skeleton, RAG, API,
compare_runs, UI 4 trang + demo mode (AppTest), dự án/pipeline/3M nhiều video, pipeline dữ liệu tổng hợp) — không cần mạng/key/dataset.
