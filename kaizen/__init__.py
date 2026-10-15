"""AI Kaizen Motion Analyzer — package lõi.

    skeleton/  Tầng 1: nguồn video (adapter), pose, features, events
    analyzer/  Tầng 2 định lượng: steps, metrics 3M, findings (rule, KHÔNG LLM), report
    llm/       provider LLM + prompt + validator (LLM chỉ giải thích, code kiểm chứng)
    rag/       tài liệu: parse, chunk, embed, truy xuất, trích bước chuẩn
    core/      run_io, config station.yaml, schema artifact, tiện ích I/O
    api/       FastAPI service

Scripts src/0X_*.py là wrapper mỏng gọi vào đây; schema artifact giữ nguyên.
"""
__version__ = "0.2.0"
