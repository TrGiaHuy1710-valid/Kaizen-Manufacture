"""Schema (pydantic) cho mọi artifact của pipeline — ĐÚNG field/thứ tự đang ghi ra file.

Quy ước:
- Model dựng từ dict artifact bằng `Model.model_validate(d)`; ghi lại bằng `to_json_dict()` (= model_dump
  exclude_unset) → key thiếu trong dict gốc vẫn thiếu, thứ tự key theo thứ tự khai báo → file byte-giống bản cũ.
- extra="allow": artifact cũ có thêm key vẫn đọc được (forward/backward compatible).
- Số đo dùng `Any` ở những chỗ code có thể đưa numpy scalar / dict lồng — schema mô tả, không ép kiểu.

Artifact:
    data/keypoints/<stem>_<model>.npz      → KeypointsMeta (trường `meta` JSON)
    <run>/features/<stem>.parquet          → FEATURE_COLUMNS
    <run>/events/<stem>.csv                → Event (mỗi dòng)
    <run>/steps/<stem>_{oracle,pred}.csv   → Step (mỗi dòng); _pred_info.json → PredInfo; _eval.json → EvalReport
    <run>/metrics/metrics.json             → MetricsReport{meta, trials: {stem: Metrics}, mura}
    <run>/findings/findings.json           → FindingsReport{findings: [Finding]}
    <run>/explained/explained.json         → ExplainedReport{explanations: [ExplanationRecord{final: Explanation}]}
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


Num = Union[int, float]
"""Số giữ nguyên kiểu gốc (yaml `fps: 30` vẫn là int 30 khi ghi lại, không thành 30.0)."""


class Artifact(BaseModel):
    """Base: cho phép key thừa, dump giữ nguyên key đã set."""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True)

    def to_json_dict(self) -> dict:
        """dict để ghi JSON — chỉ key có trong dữ liệu gốc, theo thứ tự khai báo."""
        return self.model_dump(mode="python", exclude_unset=True, by_alias=True)


# ---------------------------------------------------------------- station.yaml
class StepDef(Artifact):
    """1 bước SOP trong station.yaml."""
    id: int
    name: str = ""
    zone: Optional[str] = None
    reps: int = 1
    va: bool = False
    std_time_s: Optional[Num] = None


class StationConfig(Artifact):
    """station.yaml (các khoá đang dùng; khoá khác giữ nguyên qua extra)."""
    fps: Num = 30
    px_per_cm: Optional[Num] = None
    moving_speed_px_s: Num = 120
    min_event_s: Num = 0.25
    idle_threshold_s: Num = 3.0
    idle_still_s: Num = 1.5
    pick_max_still_s: Num = 2.0
    extract_roi: Optional[list[int]] = None
    features: dict[str, Any] = Field(default_factory=dict)
    zones: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepDef] = Field(default_factory=list)
    step_min_gap_s: Num = 1.0
    muri_dev_deg: Num = 30
    muri_pct_threshold: Num = 20
    cycle_time_target_s: Optional[Num] = None

    @classmethod
    def from_yaml(cls, path) -> "StationConfig":
        """Đọc station.yaml (không điền mặc định vào dict — mặc định nằm ở model)."""
        import yaml
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        """dict giống hệt load_config(): key gốc của yaml + mặc định của load_config."""
        from kaizen.core.config import with_defaults
        return with_defaults(self.to_json_dict())


# ---------------------------------------------------------------- keypoints
class KeypointsMeta(Artifact):
    """Trường `meta` (JSON) trong .npz; mảng: kpts (T,K,3) [x_px,y_px,score], t (T,), frame (T,), names (K,)."""
    model: str
    video: Optional[str] = None
    frames_dir: Optional[str] = None
    src_fps: Num
    stride: int = 1
    width: int
    height: int
    roi: Optional[list[int]] = None
    hand_crop: Optional[bool] = None
    proc_fps: Num
    detect_rate: Num
    valid_rate_score03: dict[str, Num] = Field(default_factory=dict)


FEATURE_COLUMNS = ["t", "frame", "lw_x", "lw_y", "lw_speed", "lw_score", "lw_interp", "lw_zone",
                   "rw_x", "rw_y", "rw_speed", "rw_score", "rw_interp", "rw_zone", "interp"]
"""Cột bắt buộc của features.parquet (+ tuỳ chọn l_elbow_*, r_elbow_*, l_mcp_*, r_mcp_* cho Muri)."""


# ---------------------------------------------------------------- events / steps
class Event(Artifact):
    """1 dòng events.csv."""
    start_s: Num
    end_s: Num
    hand: Literal["left", "right"]
    zone: str
    motion: Literal["moving", "still", "lost"]
    dur_s: Num
    event_type: str


EVENT_COLUMNS = list(Event.model_fields)


class Step(Artifact):
    """1 dòng steps csv (oracle hoặc pred; `missing` chỉ có ở pred)."""
    step_id: int
    repetition: int = 0
    name: str = ""
    start_s: Num
    end_s: Num
    dur_s: Num
    source: Literal["oracle", "pred"]
    missing: Optional[str] = None


STEP_COLUMNS = ["step_id", "repetition", "name", "start_s", "end_s", "dur_s", "source"]


class PredInfo(Artifact):
    """<stem>_pred_info.json."""
    not_found: list[str] = Field(default_factory=list)
    no_zone: list[str] = Field(default_factory=list)
    matched_order: list[int] = Field(default_factory=list)


class StepEval(Artifact):
    """Sai số 1 bước pred vs oracle."""
    step_id: int
    repetition: int
    start_err_s: Optional[Num] = None
    end_err_s: Optional[Num] = None
    dur_err_s: Optional[Num] = None
    iou: Optional[Num] = None


class EvalReport(Artifact):
    """<stem>_eval.json — predicted vs nhãn."""
    n_pred: int
    n_oracle: int
    n_unclassified: int
    mae_step_dur_s: Optional[Num] = None
    mae_start_s: Optional[Num] = None
    mae_end_s: Optional[Num] = None
    cycle_time_pred_s: Optional[Num] = None
    cycle_time_oracle_s: Optional[Num] = None
    cycle_time_err_s: Optional[Num] = None
    precision_iou05: Optional[Num] = None
    recall_iou05: Optional[Num] = None
    f1_iou05: Optional[Num] = None
    per_step: list[StepEval] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)
    no_zone: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- metrics
class MetricValue(Artifact):
    """Mọi metric: {value, unit, definition} — 07 đưa nguyên văn cho LLM."""
    value: Any = None
    unit: str
    definition: str


class StepsTable(Artifact):
    """metrics.trials[stem].steps."""
    unit: dict[str, str] = Field(default_factory=dict)
    definition: dict[str, str] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class Metrics(Artifact):
    """Chỉ số 1 trial (metrics.json → trials[stem]). Thứ tự field = thứ tự ghi của 05."""
    window_s: list[Any]
    cycle_time_s: MetricValue
    n_steps: MetricValue
    unclassified_s: MetricValue
    steps: StepsTable
    va_pct: MetricValue
    idle_left_s: Optional[MetricValue] = None
    transfer_left_s: Optional[MetricValue] = None
    lost_left_pct: Optional[MetricValue] = None
    idle_right_s: Optional[MetricValue] = None
    transfer_right_s: Optional[MetricValue] = None
    lost_right_pct: Optional[MetricValue] = None
    idle_both_s: Optional[MetricValue] = None
    picks_by_zone: Optional[MetricValue] = None
    picks_total: Optional[MetricValue] = None
    pick_mean_dur_s: Optional[MetricValue] = None
    distance_left: Optional[MetricValue] = None
    valid_left_pct: Optional[MetricValue] = None
    distance_right: Optional[MetricValue] = None
    valid_right_pct: Optional[MetricValue] = None
    muri_wrist: Optional[dict[str, Any]] = None
    eval_vs_oracle: Optional[dict[str, Any]] = None

    _steps_table: Any = PrivateAttr(default=None)   # pd.DataFrame bảng bước (ghi <stem>_steps.csv)

    @property
    def steps_table(self):
        """DataFrame bảng theo bước (giống steps.rows) — None nếu model dựng từ JSON."""
        return self._steps_table

    def value(self, key: str):
        """metrics.<key>.value hoặc None."""
        m = getattr(self, key, None)
        return m.value if isinstance(m, MetricValue) else None


class MetricsMeta(Artifact):
    """metrics.json → meta."""
    source: Literal["oracle", "pred"]
    fps: Num
    px_per_cm: Optional[Num] = None
    moving_speed_px_s: Num
    n_trials: int
    subjects: list[str]
    muri_note: str


class MetricsReport(Artifact):
    """metrics.json."""
    meta: MetricsMeta
    trials: dict[str, Metrics]
    mura: dict[str, Any]


# ---------------------------------------------------------------- findings
class Finding(Artifact):
    """1 finding rule-based (findings.json → findings[])."""
    id: str
    type: str
    severity: Literal["high", "med", "low"]
    evidence: dict[str, Any]
    impact_s_per_cycle: Optional[Num] = None
    confidence: Literal["high", "med", "low"]
    rule: str
    rank_score: Optional[Num] = None


class FindingsReport(Artifact):
    """findings.json."""
    source: str
    rank_formula: str
    notes: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


# ---------------------------------------------------------------- explanation
class Explanation(Artifact):
    """Giải thích của LLM cho 1 finding (sau parse lượng thứ)."""
    finding_id: str = ""
    observation: str = ""
    hypothesis: str = ""
    recommendation: str = ""
    citations: list[str] = Field(default_factory=list)
    format_warning: Optional[list[str]] = None


class ValidationCheck(Artifact):
    """Kết quả validator cho 1 explanation."""
    status: Literal["passed", "failed"] = "failed"
    unknown_numbers: list[str] = Field(default_factory=list)
    bad_citations: list[str] = Field(default_factory=list)
    citations_normalized: list[str] = Field(default_factory=list)
    message: str = ""


class ExplanationRecord(Artifact):
    """explained.json → explanations[]."""
    finding_id: str
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    final: Explanation
    final_check: dict[str, Any] = Field(default_factory=dict)
    validation: Literal["passed", "failed", "failed_after_retry", "parse_failed"]


class ExplainedReport(Artifact):
    """explained.json."""
    provider: str
    mode: str
    llm_model: Optional[str] = None
    calls: list[dict[str, Any]] = Field(default_factory=list)
    n_findings: int
    n_passed: int
    pass_rate: Optional[Num] = None
    n_passed_first_attempt: int
    pass_rate_first_attempt: Optional[Num] = None
    n_retried: int
    n_parse_failed: int
    validator: str
    raw_responses: dict[str, Any] = Field(default_factory=dict)
    explanations: list[ExplanationRecord] = Field(default_factory=list)
