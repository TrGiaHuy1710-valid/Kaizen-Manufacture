# Kaizen-Manufacture

AI Kaizen Motion Analyzer cho dataset HA4M: video lắp ráp → keypoints → events → chỉ số Kaizen
(Muda/Mura/Muri) → findings → LLM giải thích (có validator) → báo cáo. Chi tiết kiến trúc: [CLAUDE.md](CLAUDE.md).

## Chạy demo

```bash
conda activate kaizen-py311 && pip install -r requirements.txt
# 1) Chuẩn bị run (lần đầu trích keypoints ~6 phút/video; --dry-run = không cần API key)
bash run_all.sh --subjects U001 --trials all --oracle --dry-run --tag demo     # số theo nhãn HA4M
bash run_all.sh --subjects U001 --trials all --dry-run --tag demo              # số theo CV dự đoán (+ sai số vs nhãn)
#    LLM thật: export GEMINI_API_KEY=... rồi thay --dry-run bằng --provider gemini (hoặc anthropic + ANTHROPIC_API_KEY)
# 2) (tuỳ chọn) render sẵn video overlay h264 — cũng bấm được trong UI
python src/tools/render_demo.py --run latest --trial U001_V001
# 3) Mở dashboard (server remote: forward cổng 8501 trong VS Code)
streamlit run app/ui.py          # mở thẳng 1 run: http://localhost:8501/?run=<run_id>&trial=U001_V001
#    Tab "So sánh video": phát 2 trial cạnh nhau (vd V001 vs V002), nhảy tới cùng một bước SOP ở cả hai
```

Kiểm thử: `python -m pytest` (validator, compare_runs, smoke test 8 tab UI, trình phát so sánh video, pipeline dữ liệu tổng hợp).
