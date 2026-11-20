"""Unit test validator + parser của 07 (không gọi mạng).

    python -m pytest tests/test_validator.py -q
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import llm_validate as lv  # noqa: E402

KB_IDS = set(lv.load_ids(REPO / "config" / "kaizen_kb.md", REPO / "config" / "sop.md"))

# Finding thật của U001_V001 (outputs/findings/findings.json, chạy oracle)
F01 = {
    "id": "F01", "type": "wrist_posture", "severity": "low",
    "evidence": {"metric": "% thời gian góc cổ tay left lệch > 30°", "value": 21.4, "unit": "%", "hand": "left",
                 "threshold_pct": 20.0, "median_angle_deg": 19.6, "dev_p90_deg": 65.4, "coverage_pct": 25.8,
                 "note": "ước lượng 2D, chỉ mang tính sàng lọc", "timestamps": [], "trials": ["U001_V001"]},
    "impact_s_per_cycle": 0.0, "confidence": "low", "rule": "R7", "rank_score": 0.0,
}

# Câu trả lời Gemini thật từng bị validator cũ đánh fail oan vì token "90" (tên chỉ số P90)
GEMINI_REAL = {
    "finding_id": "F01",
    "observation": "Ở tay left trong trial U001_V001, tỷ lệ thời gian góc cổ tay lệch lớn hơn 30° là 21.4%, vượt "
                   "ngưỡng 20.0%. Góc trung vị ghi nhận 19.6°, phân vị 90 là 65.4% và độ bao phủ đạt 25.8%.",
    "hypothesis": "Giả thuyết là hướng đặt chi tiết hoặc độ cao bề mặt thao tác chưa phù hợp khiến cổ tay trái phải "
                  "gập hoặc duỗi quá mức khi thao tác.",
    "recommendation": "Sắp xếp lại chiều cao hoặc độ nghiêng của khay linh kiện và bề mặt làm việc để đưa cổ tay về "
                      "tư thế trung tính; kỹ sư IE cần quan sát video để xác nhận đánh giá tư thế thực tế.",
    "citations": ["KB-13", "KB-16"],
}


# Cùng câu trả lời theo schema EXPLAIN_V2 (07 mặc định --prompt v2): thêm ecrs + remeasure
GEMINI_V2 = dict(GEMINI_REAL, ecrs="R", remeasure="Đo lại % thời gian lệch cổ tay sau khi chỉnh khay")


def test_gemini_real_case_passes():
    v = lv.validate(GEMINI_REAL, F01, KB_IDS)
    assert v["status"] == "passed", v["message"]
    assert v["message"] == ""


def test_fabricated_number_fails_with_clean_message():
    exp = dict(GEMINI_REAL, recommendation="Đổi hướng khay giúp tiết kiệm 8.2 s mỗi chu kỳ.")
    v = lv.validate(exp, F01, KB_IDS)
    assert v["status"] == "failed"
    assert "8.2" in v["message"]
    assert "[]" not in v["message"]
    assert v["unknown_numbers"] == ["8.2"]
    assert v["bad_citations"] == []


def test_number_with_unit_suffix_is_checked():
    exp = dict(GEMINI_REAL, observation="Tay trái lệch 77.9s.")
    assert lv.validate(exp, F01, KB_IDS)["unknown_numbers"] == ["77.9"]


@pytest.mark.parametrize("cite", ["kb13", "KB13", "kb-13", "[KB-13]", "KB 13"])
def test_citation_normalized_passes(cite):
    v = lv.validate(dict(GEMINI_REAL, citations=[cite]), F01, KB_IDS)
    assert v["status"] == "passed", v["message"]
    assert v["citations_normalized"] == ["KB-13"]


def test_unknown_citation_fails():
    v = lv.validate(dict(GEMINI_REAL, citations=["KB-99"]), F01, KB_IDS)
    assert v["status"] == "failed"
    assert "KB-99" in v["message"] and "số lạ" not in v["message"]


def test_whitelist_contents():
    w = lv.whitelist(F01)
    for x in (90.0, 30.0, 65.4, 65.0, 21.4, 20.0, 1.0, 12.0, 100.0):
        assert x in w
    assert 8.2 not in w


def test_thousands_separator():
    f = {"id": "F02", "evidence": {"metric": "distance_right", "value": 13715.4, "unit": "px", "trials": ["U001_V001"]}}
    exp = {"finding_id": "F02", "observation": "Quãng đường 13,715.4 px (≈ 13715 px).", "hypothesis": "",
           "recommendation": "", "citations": ["KB-04"]}
    assert lv.validate(exp, f, KB_IDS)["status"] == "passed"


def test_parse_fenced_with_preamble():
    text = ("Dưới đây là kết quả phân tích:\n```json\n" + json.dumps([GEMINI_REAL], ensure_ascii=False)
            + "\n```\nHy vọng hữu ích! [ghi chú]")
    items = lv.parse_explanations(text)
    assert len(items) == 1 and items[0]["finding_id"] == "F01"
    assert items[0]["citations"] == ["KB-13", "KB-16"]


def test_parse_single_object_missing_and_extra_keys():
    text = '{"finding_id": "F01", "observation": "a [x] b", "citations": "KB-13, KB-16", "score": 5}'
    items = lv.parse_explanations(text)
    assert items[0]["hypothesis"] == "" and items[0]["citations"] == ["KB-13", "KB-16"]
    assert any("hypothesis" in w for w in items[0]["format_warning"])
    assert any("score" in w for w in items[0]["format_warning"])


def test_parse_garbage_raises():
    with pytest.raises(lv.ParseError):
        lv.parse_explanations("Xin lỗi, tôi không thể trả lời.")


# ---------------- Luồng 07 với provider giả (retry + parse_failed) ----------------
def _run_07(tmp_path, monkeypatch, responses: list[str]) -> dict:
    m7 = importlib.import_module("07_explain")
    import kaizen.llm as llm
    monkeypatch.chdir(tmp_path)          # không đụng outputs/runs/ thật của repo
    seq = iter(responses)

    class Scripted:                      # provider giả: trả lần lượt các response (không mạng, không key)
        name, model = "gemini", "fake-model"

        def complete(self, system, user, **kw):
            return llm.LLMResult(next(seq), {"provider": "fake"})
    monkeypatch.setattr(llm, "get_provider", lambda name=None, model=None: Scripted())
    fpath = tmp_path / "findings.json"
    fpath.write_text(json.dumps({"findings": [F01]}, ensure_ascii=False))
    out = tmp_path / "explained.json"
    monkeypatch.setattr(sys, "argv", ["07", "--provider", "gemini", "--findings", str(fpath), "--out", str(out),
                                      "--kb", str(REPO / "config/kaizen_kb.md"), "--sop", str(REPO / "config/sop.md")])
    m7.main()
    return json.loads(out.read_text())


def test_retry_fixes_fabricated_number(tmp_path, monkeypatch):
    bad = dict(GEMINI_V2, observation="Tay trái lệch nhiều, tiết kiệm được 8.2 s.")
    res = _run_07(tmp_path, monkeypatch, [json.dumps([bad], ensure_ascii=False),
                                          "```json\n" + json.dumps([GEMINI_V2], ensure_ascii=False) + "\n```"])
    x = res["explanations"][0]
    assert x["validation"] == "passed" and len(x["attempts"]) == 2
    assert res["pass_rate"] == 1.0 and res["pass_rate_first_attempt"] == 0.0


def test_retry_still_bad_marks_failed_after_retry(tmp_path, monkeypatch):
    bad = json.dumps([dict(GEMINI_V2, citations=["KB-99"])])
    res = _run_07(tmp_path, monkeypatch, [bad, bad])
    assert res["explanations"][0]["validation"] == "failed_after_retry"


def test_v2_format_three_sentence_observation_fails_then_retry(tmp_path, monkeypatch):
    long = dict(GEMINI_V2, observation="Tay trái lệch 21.4% thời gian. Ngưỡng là 20.0%. Độ phủ 25.8%.")
    res = _run_07(tmp_path, monkeypatch, [json.dumps([long], ensure_ascii=False), json.dumps([GEMINI_V2], ensure_ascii=False)])
    x = res["explanations"][0]
    a1 = x["attempts"][0]["check"]
    assert a1["status"] == "failed" and a1["format_errors"] == ["quan sát 3 câu > 2"] and "format:" in a1["message"]
    assert x["validation"] == "passed" and len(x["attempts"]) == 2 and x["final"]["ecrs"] == "R"
    assert res["pass_rate_first_attempt"] == 0.0 and res["pass_rate"] == 1.0 and res["prompt"] == "explain_v2_kb"
    assert res["citation_mode"] == "kb" and "KB-xx" in res["citation_note"]
    assert [c["short"] for c in x["citations_detail"]] == ["[KB-13]", "[KB-16]"]


def test_v2_format_checks_unit():
    fmt = lv.FORMAT_V2
    assert lv.validate(GEMINI_V2, F01, KB_IDS, fmt=fmt)["status"] == "passed"
    gem = "Đo lại phần trăm thời gian cổ tay lệch góc sau khi điều chỉnh tư thế làm việc."   # 18 tiếng ≈ 10 từ (run thật)
    assert lv.validate(dict(GEMINI_V2, remeasure=gem), F01, KB_IDS, fmt=fmt)["status"] == "passed"
    assert lv.count_sentences("Lệch 21.4% (bước 2, 7). Cần xem lại!") == 2 and lv.count_sentences("") == 0
    assert [lv.normalize_ecrs(v) for v in ("r", "S - Simplify", "Sắp xếp lại", "Loại bỏ", "X", "")] == \
        ["R", "S", "R", "E", None, None]
    v = lv.validate(dict(GEMINI_V2, hypothesis="A. B.", ecrs="Z", remeasure="đo " * 21), F01, KB_IDS, fmt=fmt)
    assert v["status"] == "failed" and v["format_errors"] == ["giả thuyết 2 câu > 1",
                                                              "đo lại 21 tiếng > 20 (≈ 12 từ, đếm theo khoảng trắng)",
                                                              "ecrs không hợp lệ: 'Z' (cần E|C|R|S)"]
    assert lv.validate(dict(GEMINI_V2, remeasure="Đo lại sau 50 chu kỳ"), F01, KB_IDS, fmt=fmt)["unknown_numbers"] == ["50"]
    assert "format_errors" not in lv.validate(GEMINI_REAL, F01, KB_IDS)          # V1: không kiểm định dạng


def test_parse_failed_does_not_crash(tmp_path, monkeypatch):
    res = _run_07(tmp_path, monkeypatch, ["không có json", "vẫn không có json"])
    assert res["explanations"][0]["validation"] == "parse_failed"
    assert res["n_parse_failed"] == 1 and "attempt1" in res["raw_responses"]
