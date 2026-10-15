"""kaizen.llm — provider abstraction, parse lượng thứ chung, validator (KB + chunk_id động), explain() có retry.

Không gọi mạng: FakeProvider kịch bản sẵn; provider thật được khởi tạo với key giả và thay client bằng mock
để kiểm tra đúng tham số gửi SDK (json_mode, max_tokens, history).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaizen.llm as llm
from kaizen.llm import providers as P
from kaizen.llm.explain import KnowledgeBase, explain
from kaizen.llm.validator import normalize_citation, number_tokens, validate

REPO = Path(__file__).resolve().parents[1]
KB = KnowledgeBase.from_files(REPO / "config" / "kaizen_kb.md", REPO / "config" / "sop.md")
KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
            "LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_MODEL", "GEMINI_MODEL", "OPENAI_MODEL")

F01 = {"id": "F01", "type": "wrist_posture", "severity": "low", "rule": "R7", "confidence": "low",
       "impact_s_per_cycle": 0.0, "rank_score": 0.0,
       "evidence": {"metric": "% thời gian góc cổ tay left lệch > 30°", "value": 21.4, "unit": "%", "hand": "left",
                    "threshold_pct": 20.0, "median_angle_deg": 19.6, "dev_p90_deg": 65.4, "coverage_pct": 25.8,
                    "timestamps": [], "trials": ["U001_V001"]}}
F02 = {"id": "F02", "type": "long_idle", "severity": "med", "rule": "R3", "confidence": "med",
       "impact_s_per_cycle": 4.2, "rank_score": 2.52,
       "evidence": {"metric": "idle_right_s", "value": 7.2, "unit": "s", "hand": "right", "threshold_s": 3.0,
                    "timestamps": [[10.5, 14.0]], "trials": ["U001_V002"]}}
GOOD1 = {"finding_id": "F01", "observation": "Tay left lệch > 30° trong 21.4% thời gian, vượt ngưỡng 20.0%.",
         "hypothesis": "Giả thuyết: độ cao khay chưa phù hợp.", "recommendation": "Hạ bề mặt làm việc.",
         "citations": ["KB-13"]}
GOOD2 = {"finding_id": "F02", "observation": "Tay phải idle 7.2 s, ngưỡng 3.0 s.",
         "hypothesis": "Giả thuyết: chờ linh kiện.", "recommendation": "Pre-position linh kiện.", "citations": ["KB-03"]}


@pytest.fixture
def no_keys(monkeypatch):
    for k in KEY_VARS:
        monkeypatch.delenv(k, raising=False)


class FakeProvider:
    """Provider kịch bản: trả lần lượt `responses` (str hoặc Exception); ghi lại mọi lời gọi."""

    def __init__(self, responses, name="fake", model="fake-1"):
        self.name, self.model, self.calls = name, model, []
        self._seq = iter(responses)

    def complete(self, system, user, *, json_mode=True, max_tokens=2000, history=None):
        self.calls.append({"system": system, "user": user, "history": list(history or []), "max_tokens": max_tokens})
        r = next(self._seq)
        if isinstance(r, Exception):
            raise r
        return llm.LLMResult(r, {"provider": self.name, "n": len(self.calls)})


# ---------------------------------------------------------------- interface + parse chung
def test_fake_provider_satisfies_protocol():
    assert isinstance(FakeProvider([]), llm.LLMProvider)
    assert isinstance(P.DryRunProvider(), llm.LLMProvider)


@pytest.mark.parametrize("text", [
    "Đây là kết quả:\n```json\n[{\"a\": 1}]\n```\nHy vọng hữu ích [ghi chú]",
    "```\n{\"a\": 1, \"s\": \"có ] và } trong chuỗi\"}\n```",
    "  [ {\"a\": 1} ] trailing",
])
def test_llmresult_parsed_strips_fence_and_preamble(text):
    p = llm.LLMResult(text).parsed
    assert p is not None and (p[0]["a"] if isinstance(p, list) else p["a"]) == 1


def test_llmresult_unparseable():
    r = llm.LLMResult("Xin lỗi, tôi không thể trả lời.")
    assert r.parsed is None and "không thấy" in r.parse_error


# ---------------------------------------------------------------- explain() + retry
def test_explain_pass_first_attempt():
    fp = FakeProvider([json.dumps([GOOD1, GOOD2], ensure_ascii=False)])
    res = explain([F01, F02], KB, fp)
    assert (res["pass_rate"], res["pass_rate_first_attempt"], res["n_retried"]) == (1.0, 1.0, 0)
    assert res["provider"] == "fake" and res["llm_model"] == "fake-1" and len(fp.calls) == 1
    assert "<kaizen_kb>" in fp.calls[0]["user"] and fp.calls[0]["max_tokens"] == P.MAX_TOKENS


def test_explain_retry_with_validator_feedback_fixes_number():
    bad = dict(GOOD1, recommendation="Đổi hướng khay tiết kiệm 8.2 s mỗi chu kỳ.")
    fp = FakeProvider([json.dumps([bad, GOOD2], ensure_ascii=False), "```json\n" + json.dumps([GOOD1]) + "\n```"])
    res = explain([F01, F02], KB, fp)
    x1 = res["explanations"][0]
    assert x1["validation"] == "passed" and len(x1["attempts"]) == 2
    assert x1["attempts"][0]["check"]["unknown_numbers"] == ["8.2"]
    assert (res["pass_rate"], res["pass_rate_first_attempt"], res["n_retried"]) == (1.0, 0.5, 1)
    retry = fp.calls[1]
    assert "8.2" in retry["user"] and "F01" in retry["user"] and "F02" not in retry["user"].split("finding:")[-1]
    assert [m["role"] for m in retry["history"]] == ["user", "assistant"]          # có phản hồi, nhiều lượt


def test_explain_retry_still_bad_and_bad_citation():
    bad = json.dumps([dict(GOOD1, citations=["KB-99"])])
    res = explain([F01], KB, FakeProvider([bad, bad]))
    x = res["explanations"][0]
    assert x["validation"] == "failed_after_retry" and x["final_check"]["bad_citations"] == ["KB-99"]


def test_explain_parse_retry_then_parse_failed_does_not_crash():
    ok = explain([F01], KB, FakeProvider(["không có json", json.dumps([GOOD1])]))
    assert ok["explanations"][0]["validation"] == "passed" and ok["raw_responses"]["parse_errors_attempt1"]
    fp = FakeProvider(["không có json", "vẫn không"])
    res = explain([F01], KB, fp)
    assert res["explanations"][0]["validation"] == "parse_failed" and res["n_parse_failed"] == 1
    assert fp.calls[1]["user"].startswith("Response trước không parse được")


def test_explain_llm_error_first_call_raises_retry_error_recorded():
    with pytest.raises(llm.LLMError):
        explain([F01], KB, FakeProvider([llm.LLMError("HTTP 500")]))
    bad = json.dumps([dict(GOOD1, observation="lệch 99.9%")])
    res = explain([F01], KB, FakeProvider([bad, llm.LLMError("HTTP 529")]))
    assert res["explanations"][0]["validation"] == "failed_after_retry"


def test_explain_dry_run_and_no_findings():
    res = explain([F01, F02], KB, llm.get_provider("dry-run"))
    assert res["mode"] == "dry-run" and res["pass_rate"] == 1.0 and res["calls"] == [] and res["llm_model"] is None
    assert explain([], KB, FakeProvider([]))["mode"] == "none"


# ---------------------------------------------------------------- validator: KB tĩnh + chunk_id động
def test_validator_static_kb_pass_fail():
    ids = set(KB.ids)
    assert validate(GOOD1, F01, ids)["status"] == "passed"
    v = validate(dict(GOOD1, observation="Chậm 999.9 s.", citations=["KB-77"]), F01, ids)
    assert v["status"] == "failed" and v["unknown_numbers"] == ["999.9"] and v["bad_citations"] == ["KB-77"]


def test_validator_dynamic_chunk_citations():
    store = {"DOC-3f9a2c:p2:c1", "DOC-3f9a2c:p8:c3"}
    exists = store.__contains__
    exp = dict(GOOD1, citations=["doc-3F9A2C:p08:c3", "[DOC-3f9a2c:p2:c1]"])
    v = validate(exp, F01, set(KB.ids), exists)
    assert v["status"] == "passed" and v["citations_normalized"] == ["DOC-3f9a2c:p8:c3", "DOC-3f9a2c:p2:c1"]
    v = validate(dict(GOOD1, citations=["DOC-3f9a2c:p9:c0"]), F01, set(KB.ids), exists)
    assert v["status"] == "failed" and v["bad_citations"] == ["DOC-3f9a2c:p9:c0"]
    assert validate(dict(GOOD1, citations=["DOC-3f9a2c:p2:c1"]), F01, set(KB.ids))["status"] == "failed"  # không store
    # số trong chunk_id (trang 88, hash 777777) KHÔNG được nới whitelist: nhắc "88" trong văn bản vẫn là số lạ
    v = validate(dict(GOOD1, observation="Lệch 21.4%, trang 88.", citations=["DOC-777777:p88:c3"]), F01, set(KB.ids),
                 {"DOC-777777:p88:c3"}.__contains__)
    assert v["status"] == "failed" and v["unknown_numbers"] == ["88"] and v["bad_citations"] == []
    assert number_tokens("theo DOC-3f9a2c:p8:c3 thì lệch 21.4%") == ["21.4"]
    assert normalize_citation("KB 3") == "KB-03" and normalize_citation("xyz") is None


# ---------------------------------------------------------------- get_provider / env / key
def test_get_provider_reads_env(no_keys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "dry-run")
    assert llm.get_provider().name == "dry-run"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = llm.get_provider()
    assert p.name == "openai" and p.model == P.DEFAULT_MODELS["openai"]
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x-env")
    assert llm.get_provider().model == "gpt-x-env"
    monkeypatch.setenv("LLM_MODEL", "global-model")
    assert llm.get_provider().model == "global-model"
    assert llm.get_provider(model="cli-model").model == "cli-model"                     # --llm-model thắng env
    assert llm.get_provider("dry-run").name == "dry-run"                                 # tham số thắng env
    with pytest.raises(ValueError, match="không hỗ trợ"):
        llm.get_provider("mistral")


def test_default_provider_is_gemini(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    p = llm.get_provider()
    assert p.name == "gemini" and p.model == P.DEFAULT_MODELS["gemini"]


@pytest.mark.parametrize("name,env", [("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                                      ("openai", "OPENAI_API_KEY")])
def test_missing_key_fails_at_init_with_env_name(no_keys, name, env):
    with pytest.raises(llm.MissingAPIKeyError) as ei:
        llm.get_provider(name)
    assert env in str(ei.value) and ei.value.env == env
    assert name not in llm.available_providers() and "dry-run" in llm.available_providers()


def test_cli_07_missing_key_exits_with_env_name(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
    (tmp_path / "f.json").write_text(json.dumps({"findings": [F01]}))
    r = subprocess.run([sys.executable, str(REPO / "src" / "07_explain.py"), "--provider", "openai",
                        "--findings", str(tmp_path / "f.json"), "--out", str(tmp_path / "e.json"),
                        "--kb", str(REPO / "config/kaizen_kb.md"), "--sop", str(REPO / "config/sop.md")],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "OPENAI_API_KEY" in r.stderr and not (tmp_path / "e.json").exists()


# ---------------------------------------------------------------- wiring SDK (mock client, không mạng)
def test_openai_wiring_json_mode(no_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = llm.get_provider("openai", "gpt-test")
    seen = {}

    def create(**kw):
        seen.update(kw)
        msg = SimpleNamespace(content='{"explanations": [{"finding_id": "F01"}]}', refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="gpt-test",
                               id="resp_1", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    p.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    r = p.complete("SYS", "U", history=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}],
                   max_tokens=123)
    assert seen["response_format"] == {"type": "json_object"} and seen["max_completion_tokens"] == 123
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]
    assert r.parsed["explanations"][0]["finding_id"] == "F01" and r.usage["provider"] == "openai"
    seen.clear()
    p.complete("SYS", "U", json_mode=False)
    assert "response_format" not in seen


def test_gemini_wiring_json_mode(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    p = llm.get_provider("gemini", "gemini-test")
    seen = {}

    def generate_content(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return SimpleNamespace(text='[{"finding_id": "F01"}]', usage_metadata=None, candidates=[],
                               model_version="gemini-test", response_id="r1")
    p.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    r = p.complete("SYS", "U", history=[{"role": "assistant", "content": "a0"}], max_tokens=77)
    assert seen["config"].response_mime_type == "application/json" and seen["config"].max_output_tokens == 77
    assert [c.role for c in seen["contents"]] == ["model", "user"] and r.parsed[0]["finding_id"] == "F01"
    p.complete("SYS", "U", json_mode=False)
    assert seen["config"].response_mime_type is None


def test_anthropic_wiring(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = llm.get_provider("anthropic")
    assert p.model == P.DEFAULT_MODELS["anthropic"]
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='[{"finding_id": "F01"}]')],
                               model=kw["model"], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=3, output_tokens=4))
    p.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    r = p.complete("SYS", "U", history=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}])
    assert seen["system"] == "SYS" and seen["max_tokens"] == 2000 and len(seen["messages"]) == 3
    assert r.parsed[0]["finding_id"] == "F01" and r.usage["output_tokens"] == 4
