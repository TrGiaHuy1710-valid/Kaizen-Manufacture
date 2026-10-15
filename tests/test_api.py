"""FastAPI service (kaizen.api.app) qua TestClient — không mạng: embedder băm, provider dry-run, run giả trong tmp."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")
from fastapi.testclient import TestClient  # noqa: E402

from kaizen.api.app import create_app  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RID = "20260101-000000_oracle_mediapipe_dry_api"
FINDING = {"id": "F01", "type": "long_idle", "severity": "med", "rule": "R3", "confidence": "med",
           "impact_s_per_cycle": 4.2, "rank_score": 2.52,
           "evidence": {"metric": "idle_right_s", "value": 7.2, "unit": "s", "hand": "right", "threshold_s": 3.0,
                        "timestamps": [[10.5, 14.0]], "trials": ["U001_V001"]}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    run = tmp_path / "outputs" / "runs" / RID
    for st in ("metrics", "findings", "explained"):
        (run / st).mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": RID, "created_at": "2026-01-01T00:00:00",
                                                   "source": "oracle", "pose_model": "mediapipe",
                                                   "llm": {"provider": "dry-run"}, "trials": {"U001_V001": {}}}))
    (run / "metrics" / "metrics.json").write_text(json.dumps({"meta": {"source": "oracle"}, "trials": {}, "mura": {}}))
    (run / "findings" / "findings.json").write_text(json.dumps({"source": "oracle", "findings": [FINDING]}))
    (tmp_path / "outputs" / "runs" / "LATEST").write_text(RID)
    shutil.copytree(REPO / "config", tmp_path / "config")
    monkeypatch.chdir(tmp_path)
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
              "LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    app = create_app(store_path=tmp_path / "store", embedder="hash", extractions_dir=tmp_path / "extractions")
    return TestClient(app)


def upload(client, path: Path):
    with open(path, "rb") as f:
        return client.post("/documents", files={"file": (path.name, f, "application/pdf")})


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["providers_available"] == ["dry-run"] and r["store_chunks"] == 0


def test_upload_query_extract_flow(client, sop_pdf, tmp_path):
    r = upload(client, sop_pdf)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["doc_id"].startswith("DOC-") and d["pages"] == 2 and d["chunks"] > 0 and d["needs_ocr_pages"] == []
    assert upload(client, sop_pdf).json()["already_present"] is True
    assert client.get("/health").json()["store_chunks"] == d["chunks"]
    assert [x["doc_id"] for x in client.get("/documents").json()] == [d["doc_id"]]

    q = client.post("/rag/query", json={"query": "lắp bánh răng hành tinh", "k": 2}).json()
    top = q["results"][0]
    assert top["page"] == 1 and top["doc_id"] == d["doc_id"] and top["chunk_id"].startswith(d["doc_id"] + ":p1:")
    assert {"title", "section", "text", "score"} <= set(top)

    x = client.post("/rag/extract-steps", json={"doc_id": d["doc_id"], "provider": "dry-run"})
    assert x.status_code == 200, x.text
    steps = x.json()["steps"]
    assert [s["order"] for s in steps] == list(range(1, 13)) and all(s["page"] in (1, 2) for s in steps)
    assert "bánh răng hành tinh" in steps[2]["name"] and steps[2]["page"] == 1 and steps[11]["page"] == 2
    assert (tmp_path / "extractions" / f"{d['doc_id']}.json").exists()


def test_upload_errors(client, tmp_path):
    bad = tmp_path / "a.xlsx"
    bad.write_bytes(b"xx")
    assert upload(client, bad).status_code == 415
    empty = tmp_path / "e.pdf"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 not really")
    assert upload(client, broken).status_code == 422
    assert client.post("/rag/extract-steps", json={"doc_id": "DOC-000000"}).status_code == 404


def test_explain_endpoint_rag_save_and_provider_errors(client, tmp_path):
    r = client.post("/analyze/explain", json={"run_id": "latest", "provider": "dry-run", "use_rag": True, "k": 2})
    assert r.status_code == 200, r.text
    ex = r.json()
    assert ex["run_id"] == RID and ex["pass_rate"] == 1.0 and ex["rag"]["prompt"] == "explain_rag_v1"
    assert all(c.startswith("DOC-") for c in ex["explanations"][0]["final"]["citations"])
    plain = client.post("/analyze/explain", json={"run_id": RID, "provider": "dry-run", "save": True}).json()
    assert plain["explanations"][0]["final"]["citations"] == ["KB-03", "KB-16"] and "rag" not in plain
    saved = tmp_path / "outputs" / "runs" / RID / "explained" / "explained.json"
    assert json.loads(saved.read_text())["pass_rate"] == 1.0
    assert client.get(f"/runs/{RID}/explained").json()["provider"] == "dry-run"

    e = client.post("/analyze/explain", json={"run_id": RID, "provider": "openai"})
    assert e.status_code == 400 and "OPENAI_API_KEY" in e.json()["detail"]
    assert client.post("/analyze/explain", json={"run_id": RID, "provider": "gemini"}).json()["detail"].count("GEMINI_API_KEY")
    assert client.post("/analyze/explain", json={"run_id": RID, "provider": "mistral"}).status_code == 400
    assert client.post("/rag/extract-steps", json={"doc_id": "x", "provider": "anthropic"}).status_code == 404


def test_runs_endpoints(client):
    runs = client.get("/runs").json()
    assert [r["run_id"] for r in runs] == [RID] and runs[0]["n_trials"] == 1
    assert client.get(f"/runs/{RID}/metrics").json()["meta"]["source"] == "oracle"
    assert client.get("/runs/latest/findings").json()["findings"][0]["id"] == "F01"
    assert client.get("/runs/nope/metrics").status_code == 404
    assert client.get("/runs/bad id!/metrics").status_code == 400
    assert client.post("/analyze/explain", json={"run_id": "nope", "provider": "dry-run"}).status_code == 404


def test_cors_localhost_only(client):
    ok = client.options("/health", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"
    evil = client.options("/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in evil.headers
