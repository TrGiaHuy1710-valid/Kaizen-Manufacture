"""Prompt là code: mọi chuỗi gửi LLM nằm ở đây, có version (không rải chuỗi trong script).

Đổi nội dung prompt = tạo version mới (EXPLAIN_V2...) để kết quả các run cũ vẫn truy được prompt đã dùng
(manifest.stages.07_explain.prompt ghi id).

    EXPLAIN_V1         giải thích findings với KB/SOP tĩnh (citation KB-xx / SOP-xx)
    EXPLAIN_RAG_V1     giải thích findings với chunk truy xuất từ RAG store (citation = chunk_id)
    ONLY_JSON_V1       hỏi lại khi response không parse được
    RETRY_FEEDBACK_V1  gọi lại kèm thông báo lỗi validator
    EXTRACT_STEPS_V1   trích bước chuẩn từ tài liệu SOP
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """system cố định + user là template str.format (giá trị chèn vào không bị format lại)."""
    id: str
    system: str
    user: str

    def render(self, **kw) -> str:
        """User message với các placeholder đã điền."""
        return self.user.format(**kw)


_RULES_NUM = """1. observation chỉ được dùng các con số CÓ SẴN trong JSON của finding đó, chép đúng giá trị (được làm tròn 1 chữ số thập phân). KHÔNG tự tính, cộng, trừ, quy đổi hay tạo số mới. Không có số phù hợp thì diễn đạt bằng lời.
2. hypothesis là giả thuyết nguyên nhân, nêu rõ là giả thuyết; cũng không được đưa số mới.
3. recommendation: đề xuất cải tiến cụ thể, ưu tiên theo ECRS; không đưa số mới."""

EXPLAIN_V1 = PromptTemplate(
    id="explain_v1",
    system="""Bạn là kỹ sư Kaizen/IE hỗ trợ phân tích trạm lắp ráp. Bạn CHỈ giải thích các finding đã được code tính sẵn.
Quy tắc bắt buộc:
""" + _RULES_NUM + """
4. citations: chỉ dùng ID có trong tài liệu (dạng KB-xx hoặc SOP-xx), ít nhất 1 ID cho mỗi finding.
5. Viết tiếng Việt, ngắn gọn. Dùng dấu chấm cho số thập phân.
6. Khi nhắc tên chỉ số thống kê hãy viết P90/IoU/CV nguyên dạng, không diễn giải thành số.
7. Chỉ trả về JSON array, không kèm giải thích hay markdown:
[{"finding_id": "F01", "observation": "...", "hypothesis": "...", "recommendation": "...", "citations": ["KB-03"]}]""",
    user="<kaizen_kb>\n{kb_text}\n</kaizen_kb>\n\n<sop>\n{sop_text}\n</sop>\n\n"
         "<findings>\n{findings_json}\n</findings>\n\n"
         "Giải thích TỪNG finding theo đúng định dạng JSON đã quy định.")

EXPLAIN_RAG_V1 = PromptTemplate(
    id="explain_rag_v1",
    system="""Bạn là kỹ sư Kaizen/IE hỗ trợ phân tích trạm lắp ráp. Bạn CHỈ giải thích các finding đã được code tính sẵn,
dựa trên các đoạn tài liệu (chunk) được truy xuất cho từng finding.
Quy tắc bắt buộc:
""" + _RULES_NUM + """
4. citations: chỉ dùng chunk_id có trong <documents> (dạng DOC-xxxxxx:pN:cM), ít nhất 1 chunk_id cho mỗi finding,
   ưu tiên các chunk trong "relevant_chunks" của finding đó. KHÔNG viết số trang hay mã tài liệu vào văn bản —
   nguồn chỉ nằm trong citations.
5. Viết tiếng Việt, ngắn gọn. Dùng dấu chấm cho số thập phân.
6. Khi nhắc tên chỉ số thống kê hãy viết P90/IoU/CV nguyên dạng, không diễn giải thành số.
7. Chỉ trả về JSON array, không kèm giải thích hay markdown:
[{"finding_id": "F01", "observation": "...", "hypothesis": "...", "recommendation": "...", "citations": ["DOC-3f9a2c:p2:c1"]}]""",
    user="<documents>\n{documents}\n</documents>\n\n<findings>\n{findings_json}\n</findings>\n\n"
         "Giải thích TỪNG finding theo đúng định dạng JSON đã quy định; citations chỉ gồm chunk_id.")

ONLY_JSON_V1 = PromptTemplate(
    id="only_json_v1", system="",
    user="Response trước không parse được thành JSON. Chỉ trả JSON array, không markdown, không lời dẫn.")

RETRY_FEEDBACK_V1 = PromptTemplate(
    id="retry_feedback_v1", system="",
    user="Validator phát hiện lỗi trong các explanation sau:\n{feedback}\n\n"
         "Hãy sửa đúng các lỗi trên, chỉ dùng số có trong JSON của finding tương ứng, chỉ dùng citation có "
         "trong tài liệu. Trả JSON thuần (array) chỉ gồm các finding: {finding_ids}.")

EXTRACT_STEPS_V1 = PromptTemplate(
    id="extract_steps_v1",
    system="""Bạn là kỹ sư IE đọc tài liệu hướng dẫn công việc (SOP / work instruction) của trạm lắp ráp.
Nhiệm vụ: trích danh sách BƯỚC CHUẨN theo đúng thứ tự thực hiện trong tài liệu.
Quy tắc bắt buộc:
1. Chỉ dùng thông tin có trong <documents>. Không bịa bước, không gộp hay tách bước khác với tài liệu.
2. name: tên bước ngắn gọn, giữ ngôn ngữ của tài liệu. order: số thứ tự 1, 2, 3... theo tài liệu. id = order.
3. zone_hint: vị trí/khay/khu vực tài liệu nêu cho bước đó (vd "khay vít bên trái"), không có thì null.
4. std_time_s: CHỈ điền khi tài liệu ghi rõ thời gian chuẩn của bước đó bằng giây — chép đúng số; không tự quy đổi
   hay ước lượng; không có thì null.
5. safety_note: lưu ý an toàn/chất lượng tài liệu nêu cho bước đó, không có thì null.
6. source_chunk_id: chunk_id (dạng DOC-xxxxxx:pN:cM) của đoạn mô tả bước đó.
7. Chỉ trả về JSON object, không markdown:
{"steps": [{"id": 1, "name": "...", "order": 1, "zone_hint": null, "std_time_s": null, "safety_note": null, "source_chunk_id": "DOC-3f9a2c:p1:c0"}]}""",
    user="<documents>\n{documents}\n</documents>\n\nTrích các bước chuẩn của tài liệu \"{title}\" theo đúng định dạng JSON.")

# Tương thích tên cũ (07_explain, tests)
SYSTEM = EXPLAIN_V1.system
ONLY_JSON = ONLY_JSON_V1.user

DRY_CITES = {"slow_step": ["KB-10", "KB-12"], "extra_pick": ["KB-04", "KB-07"], "long_idle": ["KB-03", "KB-16"],
             "sequence_deviation": ["KB-14", "SOP-20"], "hand_imbalance": ["KB-05", "KB-15"],
             "high_variation": ["KB-10", "KB-01"], "wrist_posture": ["KB-13", "KB-01"]}


def findings_json(findings: list[dict]) -> str:
    """JSON findings đưa vào prompt (indent 1, giữ Unicode)."""
    return json.dumps(findings, ensure_ascii=False, indent=1)


def user_prompt(findings: list[dict], kb_text: str, sop_text: str) -> str:
    """User message EXPLAIN_V1: toàn bộ KB + SOP + findings JSON."""
    return EXPLAIN_V1.render(kb_text=kb_text, sop_text=sop_text, findings_json=findings_json(findings))
