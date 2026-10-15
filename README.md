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
# 2) (tuỳ chọn) render sẵn video overlay h264 — cũng bấm được trong UI
python src/tools/render_demo.py --run latest --trial U001_V001
# 3) Mở dashboard (server remote: forward cổng 8501 trong VS Code)
streamlit run app/ui.py          # mở thẳng 1 run: http://localhost:8501/?run=<run_id>&trial=U001_V001
#    Tab "So sánh video": phát 2 trial cạnh nhau (vd V001 vs V002), nhảy tới cùng một bước SOP ở cả hai
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
- **Giải thích có trích dẫn**: `python src/07_explain.py --run latest --use-rag [--provider ...] --force` — `kaizen_kb.md` +
  `sop.md` được ingest như tài liệu; citation = chunk_id (`DOC-xxxxxx:pN:cM`), validator kiểm chunk_id có trong store.
- CLI: `python -m kaizen.rag ingest <files> | list | query "<q>" | extract-steps <doc_id> | to-station <doc_id>`.
- PDF scan (ít text, ảnh phủ trang) → `needs_ocr_pages`, bỏ qua ở v1 (TODO OCR).

Kiểm thử: `python -m pytest` (validator, providers LLM với FakeProvider, analyzer R1–R7, adapter skeleton, RAG, API,
compare_runs, smoke test 8 tab UI, trình phát so sánh video, pipeline dữ liệu tổng hợp) — không cần mạng/key/dataset.
