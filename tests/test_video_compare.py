"""video_compare: payload bước + HTML trình phát 2 video (không cần trình duyệt)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import video_compare as vc  # noqa: E402

STEPS = pd.DataFrame({"step_id": [2, 1, 0, 3], "repetition": [1, 0, 0, 0], "name": ["Gear", "Carrier", "unclassified", "Planet </script>"],
                      "start_s": [4.0, 1.0, 9.0, 6.0], "end_s": [6.0, 4.0, 12.0, 9.0], "dur_s": [2.0, 3.0, 3.0, 3.0]})


def test_steps_payload_sorted_with_keys_and_colors():
    p = vc.steps_payload(STEPS)
    assert [s["key"] for s in p] == ["1", "2.1", "3", "?"]
    assert p[0]["label"] == "B1 · Carrier" and p[0]["start"] == 1.0 and p[0]["color"].startswith("#")
    assert vc.steps_payload(None) == []


def test_build_html_has_both_videos_union_steps_and_is_escaped():
    a = {"title": "A · U001_V001", "b64": "QUFB", "steps": vc.steps_payload(STEPS)}
    b = {"title": "B · U001_V002", "b64": "QkJC", "steps": vc.steps_payload(STEPS.iloc[:2])}
    h = vc.build_html(a, b)
    assert '"QUFB"' in h and '"QkJC"' in h
    assert h.count('<option value="') == 4            # '— chọn —' + 1, 2.1, 3 (không có unclassified)
    assert "Planet </script>" not in h                 # tên bước không phá thẻ script
    assert h.index('value="1"') < h.index('value="2.1"') < h.index('value="3"')
