import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ce_studies.sqlite3"

# ── Legacy table (kept for backward compatibility) ──────────────────────────
LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS ce_studies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT UNIQUE,
  figure_group TEXT,
  condition TEXT,
  time_horizon TEXT,
  perspective TEXT,
  outcome_type TEXT,
  comparator_type TEXT,
  quadrant TEXT,
  notes TEXT,
  evidence_pages TEXT
);
"""

# ── Rich multi-comparison table (primary table) ──────────────────────────────
# One row per intervention-comparator pair; a paper can produce 2+ rows.
COMPARISONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS ce_comparisons (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  comparison_id        TEXT UNIQUE NOT NULL,   -- paper_id + _c1, _c2 …
  paper_id             TEXT NOT NULL,

  -- Figure classification (Fig 4 = physio vs non-physio; Fig 5 = physio vs physio)
  figure_group         TEXT DEFAULT 'unknown',

  -- Body region (granular joint/region level)
  body_region          TEXT DEFAULT 'unknown',
  condition            TEXT DEFAULT 'unknown',

  -- Study characteristics (Table 1 equivalent)
  country              TEXT DEFAULT 'unknown',
  setting              TEXT DEFAULT 'unknown',  -- primary_care | hospital | community | home
  sample_size          TEXT DEFAULT 'unknown',
  study_design         TEXT DEFAULT 'unknown',  -- RCT | observational | model-based
  time_horizon         TEXT DEFAULT 'unknown',
  perspective          TEXT DEFAULT 'unknown',  -- societal | healthcare | patient | mixed
  outcome_type         TEXT DEFAULT 'unknown',  -- QALY | clinical | both
  outcome_measure      TEXT DEFAULT 'unknown',  -- e.g. EQ-5D, WOMAC, VAS

  -- Intervention details (Table 2 equivalent)
  intervention_type    TEXT DEFAULT 'unknown',
  intervention_detail  TEXT DEFAULT 'unknown',
  frequency            TEXT DEFAULT 'unknown',  -- sessions per week
  total_sessions       TEXT DEFAULT 'unknown',  -- total # of sessions
  session_length       TEXT DEFAULT 'unknown',  -- minutes per session
  duration_weeks       TEXT DEFAULT 'unknown',  -- total weeks
  supervision          TEXT DEFAULT 'unknown',  -- supervised | home-based | group | mixed

  -- Comparator
  comparator_type      TEXT DEFAULT 'unknown',
  comparator_detail    TEXT DEFAULT 'unknown',

  -- Cost-effectiveness results
  delta_cost_direction   TEXT DEFAULT 'unknown', -- more_costly | less_costly | similar
  delta_effect_direction TEXT DEFAULT 'unknown', -- more_effective | less_effective | similar
  icer                   TEXT DEFAULT 'unknown', -- e.g. "£23,456/QALY"
  wtp_threshold          TEXT DEFAULT 'unknown', -- e.g. "£20,000/QALY"
  quadrant               TEXT DEFAULT 'unclear', -- dominant | dominated | NE | SW | unclear
  ce_conclusion          TEXT DEFAULT 'inconclusive', -- cost_effective | not_cost_effective | inconclusive

  -- Evidence / metadata
  notes                TEXT DEFAULT '',
  evidence_pages       TEXT DEFAULT '',
  extraction_confidence TEXT DEFAULT 'low',     -- high | medium | low
  created_at           TEXT DEFAULT (datetime('now'))
);
"""

# Allowed value sets for validation
VALID_BODY_REGIONS = {
    "shoulder", "knee", "hip", "low_back", "neck", "ankle",
    "elbow", "wrist", "hand", "foot", "multi_region", "other", "unknown"
}
VALID_INTERVENTION_TYPES = {
    "exercise", "manual_therapy", "education",
    "exercise+manual_therapy", "exercise+education",
    "manual_therapy+education", "mixed_physiotherapy",
    "other_physiotherapy", "other", "unknown"
}
VALID_COMPARATOR_TYPES = {
    "usual_care", "medical_care", "surgery", "injection",
    "wait_list", "other_physiotherapy", "education", "other", "unknown"
}
VALID_QUADRANTS = {"dominant", "dominated", "NE", "SW", "unclear", "unknown"}
VALID_FIGURE_GROUPS = {"Fig4", "Fig5", "unknown"}
VALID_PERSPECTIVES = {"societal", "healthcare", "patient", "mixed", "unknown"}
VALID_OUTCOME_TYPES = {"QALY", "clinical", "both", "unknown"}
VALID_DIRECTIONS = {"more_costly", "less_costly", "similar", "unknown"}
VALID_EFFECT_DIRECTIONS = {"more_effective", "less_effective", "similar", "unknown"}
VALID_CONCLUSIONS = {"cost_effective", "not_cost_effective", "inconclusive", "unknown"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    conn.execute(LEGACY_SCHEMA)
    conn.execute(COMPARISONS_SCHEMA)
    conn.commit()
    conn.close()


# ── Normalisation helpers ────────────────────────────────────────────────────

def _clean(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s else "unknown"


def _norm_enum(raw: str, valid: set, fallback: str = "unknown") -> str:
    s = _clean(raw).lower().strip()
    # direct hit
    if s in {v.lower() for v in valid}:
        for v in valid:
            if v.lower() == s:
                return v
    return fallback


def normalize_body_region(raw: str) -> str:
    s = _clean(raw).lower()
    if any(k in s for k in ["shoulder", "rotator", "impingement", "subacromial"]):
        return "shoulder"
    if any(k in s for k in ["knee", "patell", "acl", "anterior cruciate", "osteoarth"]):
        if "hip" not in s:
            return "knee"
    if any(k in s for k in ["hip", "femoral", "trochant"]):
        return "hip"
    if any(k in s for k in ["low back", "low_back", "lumbar", "lbp", "chronic back"]):
        return "low_back"
    if any(k in s for k in ["neck", "cervical", "whiplash"]):
        return "neck"
    if any(k in s for k in ["ankle", "achilles"]):
        return "ankle"
    if any(k in s for k in ["elbow", "epicondyl", "tennis elbow"]):
        return "elbow"
    if any(k in s for k in ["wrist", "carpal"]):
        return "wrist"
    if any(k in s for k in ["hand", "finger", "thumb"]):
        return "hand"
    if any(k in s for k in ["foot", "plantar", "heel"]):
        return "foot"
    if any(k in s for k in ["multi", "multiple region", "generalised", "rheumatoid"]):
        return "multi_region"
    # Broad fallbacks
    if any(k in s for k in ["spine", "back", "spinal"]):
        return "low_back"
    if any(k in s for k in ["lower limb", "lower extrem"]):
        return "knee"  # most common lower limb
    if any(k in s for k in ["upper limb", "upper extrem"]):
        return "shoulder"  # most common upper limb
    return raw if raw.strip() and raw.strip() != "unknown" else "unknown"


def normalize_intervention_type(raw: str) -> str:
    s = _clean(raw).lower()
    has_ex = any(k in s for k in ["exercise", "strength", "aerobic", "yoga", "walking", "fitness", "aqua"])
    has_mt = any(k in s for k in ["manual", "manipul", "mobiliz", "massage", "chiro"])
    has_ed = any(k in s for k in ["education", "cognitive", "CBT", "pain management", "advice", "self-management"])
    has_mixed = any(k in s for k in ["mixed", "multimodal", "combined", "multidisciplin"])

    if has_mixed:
        return "mixed_physiotherapy"
    if has_ex and has_mt and has_ed:
        return "mixed_physiotherapy"
    if has_ex and has_mt:
        return "exercise+manual_therapy"
    if has_ex and has_ed:
        return "exercise+education"
    if has_mt and has_ed:
        return "manual_therapy+education"
    if has_ex:
        return "exercise"
    if has_mt:
        return "manual_therapy"
    if has_ed:
        return "education"
    # direct enum match
    return _norm_enum(raw, VALID_INTERVENTION_TYPES)


def normalize_comparator_type(raw: str) -> str:
    s = _clean(raw).lower()
    if any(k in s for k in ["usual care", "usual medical", "standard care", "gp", "general practitioner"]):
        return "usual_care"
    if any(k in s for k in ["medical care", "physician", "doctor", "medical doctor"]):
        return "medical_care"
    if any(k in s for k in ["surg", "operation", "arthroplasty", "replacement"]):
        return "surgery"
    if any(k in s for k in ["inject", "cortisone", "steroid", "PRP", "viscosupplement"]):
        return "injection"
    if any(k in s for k in ["wait", "no treatment", "control", "sham"]):
        return "wait_list"
    if any(k in s for k in ["physio", "exercise", "manual", "rehabilitation"]):
        return "other_physiotherapy"
    if any(k in s for k in ["education", "advice"]):
        return "education"
    return _norm_enum(raw, VALID_COMPARATOR_TYPES)


def normalize_figure_group(raw: str) -> str:
    s = _clean(raw).lower()
    if "fig4" in s or "fig 4" in s or "figure 4" in s:
        return "Fig4"
    if "fig5" in s or "fig 5" in s or "figure 5" in s:
        return "Fig5"
    return _norm_enum(raw, VALID_FIGURE_GROUPS)


def normalize_quadrant(raw: str) -> str:
    s = _clean(raw).lower()
    if s in {"dominant", "dominance"}:
        return "dominant"
    if s in {"dominated"}:
        return "dominated"
    if s in {"ne", "north-east", "northeast", "north east"}:
        return "NE"
    if s in {"sw", "south-west", "southwest", "south west"}:
        return "SW"
    if "not cost" in s or "cost-ineffective" in s:
        return "dominated"
    if "no significant" in s or "no difference" in s:
        return "unclear"
    return "unclear"


def normalize_perspective(raw: str) -> str:
    s = _clean(raw).lower()
    if "societal" in s:
        return "societal"
    if "healthcare" in s or "health care" in s or "payer" in s:
        return "healthcare"
    if "patient" in s or "individual" in s:
        return "patient"
    if "mixed" in s or ("societal" in s and "healthcare" in s):
        return "mixed"
    return "unknown"


def normalize_ce_conclusion(raw: str) -> str:
    s = _clean(raw).lower()
    if any(k in s for k in ["cost_effective", "cost-effective", "cost effective", "dominant", "acceptable"]):
        if "not" in s or "no " in s:
            return "not_cost_effective"
        return "cost_effective"
    if any(k in s for k in ["not cost", "cost-ineffective", "dominated", "not acceptable"]):
        return "not_cost_effective"
    return "inconclusive"


# ── Write functions ──────────────────────────────────────────────────────────

def upsert_comparison(row: Dict[str, Any]):
    """Upsert a single comparison row into ce_comparisons."""
    conn = get_conn()
    conn.execute(COMPARISONS_SCHEMA)

    cid = _clean(row.get("comparison_id"))
    pid = _clean(row.get("paper_id"))

    conn.execute("""
    INSERT INTO ce_comparisons (
      comparison_id, paper_id, figure_group, body_region, condition,
      country, setting, sample_size, study_design, time_horizon,
      perspective, outcome_type, outcome_measure,
      intervention_type, intervention_detail,
      frequency, total_sessions, session_length, duration_weeks, supervision,
      comparator_type, comparator_detail,
      delta_cost_direction, delta_effect_direction, icer, wtp_threshold,
      quadrant, ce_conclusion, notes, evidence_pages, extraction_confidence
    ) VALUES (
      ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
    )
    ON CONFLICT(comparison_id) DO UPDATE SET
      figure_group=excluded.figure_group,
      body_region=excluded.body_region,
      condition=excluded.condition,
      country=excluded.country,
      setting=excluded.setting,
      sample_size=excluded.sample_size,
      study_design=excluded.study_design,
      time_horizon=excluded.time_horizon,
      perspective=excluded.perspective,
      outcome_type=excluded.outcome_type,
      outcome_measure=excluded.outcome_measure,
      intervention_type=excluded.intervention_type,
      intervention_detail=excluded.intervention_detail,
      frequency=excluded.frequency,
      total_sessions=excluded.total_sessions,
      session_length=excluded.session_length,
      duration_weeks=excluded.duration_weeks,
      supervision=excluded.supervision,
      comparator_type=excluded.comparator_type,
      comparator_detail=excluded.comparator_detail,
      delta_cost_direction=excluded.delta_cost_direction,
      delta_effect_direction=excluded.delta_effect_direction,
      icer=excluded.icer,
      wtp_threshold=excluded.wtp_threshold,
      quadrant=excluded.quadrant,
      ce_conclusion=excluded.ce_conclusion,
      notes=excluded.notes,
      evidence_pages=excluded.evidence_pages,
      extraction_confidence=excluded.extraction_confidence
    """, (
        cid, pid,
        normalize_figure_group(_clean(row.get("figure_group", "unknown"))),
        normalize_body_region(_clean(row.get("body_region", "unknown"))),
        _clean(row.get("condition", "unknown")),
        _clean(row.get("country", "unknown")),
        _clean(row.get("setting", "unknown")),
        _clean(row.get("sample_size", "unknown")),
        _clean(row.get("study_design", "unknown")),
        _clean(row.get("time_horizon", "unknown")),
        normalize_perspective(_clean(row.get("perspective", "unknown"))),
        _norm_enum(_clean(row.get("outcome_type", "unknown")), VALID_OUTCOME_TYPES),
        _clean(row.get("outcome_measure", "unknown")),
        normalize_intervention_type(_clean(row.get("intervention_type", "unknown"))),
        _clean(row.get("intervention_detail", "unknown")),
        _clean(row.get("frequency", "unknown")),
        _clean(row.get("total_sessions", "unknown")),
        _clean(row.get("session_length", "unknown")),
        _clean(row.get("duration_weeks", "unknown")),
        _clean(row.get("supervision", "unknown")),
        normalize_comparator_type(_clean(row.get("comparator_type", "unknown"))),
        _clean(row.get("comparator_detail", "unknown")),
        _norm_enum(_clean(row.get("delta_cost_direction", "unknown")), VALID_DIRECTIONS),
        _norm_enum(_clean(row.get("delta_effect_direction", "unknown")), VALID_EFFECT_DIRECTIONS),
        _clean(row.get("icer", "unknown")),
        _clean(row.get("wtp_threshold", "unknown")),
        normalize_quadrant(_clean(row.get("quadrant", "unclear"))),
        normalize_ce_conclusion(_clean(row.get("ce_conclusion", "inconclusive"))),
        _clean(row.get("notes", "")),
        _clean(row.get("evidence_pages", "")),
        _norm_enum(_clean(row.get("extraction_confidence", "low")), VALID_CONFIDENCE, "low"),
    ))
    conn.commit()
    conn.close()


def upsert_row(row: Dict[str, Any]):
    """Legacy upsert into ce_studies (kept for backward compat)."""
    conn = get_conn()
    conn.execute(LEGACY_SCHEMA)

    conn.execute("""
    INSERT INTO ce_studies
    (paper_id, figure_group, condition, time_horizon, perspective, outcome_type,
     comparator_type, quadrant, notes, evidence_pages)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(paper_id) DO UPDATE SET
      figure_group=excluded.figure_group,
      condition=excluded.condition,
      time_horizon=excluded.time_horizon,
      perspective=excluded.perspective,
      outcome_type=excluded.outcome_type,
      comparator_type=excluded.comparator_type,
      quadrant=excluded.quadrant,
      notes=excluded.notes,
      evidence_pages=excluded.evidence_pages
    """, (
        _clean(row.get("paper_id")),
        _clean(row.get("figure_group")),
        _clean(row.get("condition")),
        _clean(row.get("time_horizon")),
        normalize_perspective(_clean(row.get("perspective", "unknown"))),
        _clean(row.get("outcome_type")),
        _clean(row.get("comparator_type")),
        normalize_quadrant(_clean(row.get("quadrant", "unclear"))),
        _clean(row.get("notes")),
        _clean(row.get("evidence_pages")),
    ))
    conn.commit()
    conn.close()


def query_sql(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_comparisons_summary() -> Dict[str, Any]:
    """High-level counts for the structured dataset."""
    rows = query_sql("SELECT * FROM ce_comparisons")
    if not rows:
        return {"total": 0, "by_quadrant": {}, "by_body_region": {}, "by_intervention": {}}

    def counts(key):
        c: Dict[str, int] = {}
        for r in rows:
            v = str(r.get(key) or "unknown")
            c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    return {
        "total": len(rows),
        "by_quadrant": counts("quadrant"),
        "by_body_region": counts("body_region"),
        "by_intervention": counts("intervention_type"),
        "by_figure_group": counts("figure_group"),
        "by_comparator": counts("comparator_type"),
        "by_perspective": counts("perspective"),
        "by_ce_conclusion": counts("ce_conclusion"),
    }
