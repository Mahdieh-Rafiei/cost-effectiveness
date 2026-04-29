"""
Multi-comparison extraction from cost-effectiveness paper excerpts.

Each paper may contain multiple intervention-vs-comparator pairs.
This module returns a list of comparison dicts (one per pair).
Fields map directly to the ce_comparisons table in ce_db.py.
"""

import json
import re
from typing import Any, Dict, List
from .ollama_client import OllamaClient


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM = """You are a health economics expert extracting structured data from physiotherapy cost-effectiveness papers.
Return ONLY valid JSON — no markdown, no explanation text outside the JSON.
If a value is not found in the evidence, use "unknown".
Use short values (not long paragraphs) except for notes and intervention_detail.
Be conservative: only report what is explicitly stated or clearly implied."""


# ── Main extraction prompt ───────────────────────────────────────────────────

EXTRACTION_TEMPLATE = """Paper ID: {paper_id}

Extract ALL cost-effectiveness comparisons from this paper. A paper may have multiple comparisons
(e.g., "Exercise vs Usual Care" AND "Manual Therapy vs Usual Care").

Return a JSON object with this exact structure:
{{
  "comparisons": [
    {{
      "comparison_index": 1,
      "figure_group": "<Fig4 or Fig5 or unknown>",
      "body_region": "<shoulder|knee|hip|low_back|neck|ankle|elbow|wrist|hand|foot|multi_region|other|unknown>",
      "condition": "<specific diagnosis e.g. knee OA, chronic LBP, shoulder impingement>",
      "country": "<country name or unknown>",
      "setting": "<primary_care|hospital|community|home|unknown>",
      "sample_size": "<total sample size or per-group size as text>",
      "study_design": "<RCT|economic alongside RCT|model-based|other|unknown>",
      "time_horizon": "<e.g. 12 months, 2 years, 6 weeks>",
      "perspective": "<societal|healthcare|patient|mixed|unknown>",
      "outcome_type": "<QALY|clinical|both|unknown>",
      "outcome_measure": "<e.g. EQ-5D, WOMAC, VAS, SF-6D, NRS>",
      "intervention_type": "<exercise|manual_therapy|education|exercise+manual_therapy|exercise+education|manual_therapy+education|mixed_physiotherapy|other_physiotherapy|other|unknown>",
      "intervention_detail": "<1-2 sentence description of intervention content>",
      "frequency": "<sessions per week e.g. 2x/week>",
      "total_sessions": "<total number of sessions e.g. 12>",
      "session_length": "<minutes per session e.g. 60 min>",
      "duration_weeks": "<number of weeks e.g. 8>",
      "supervision": "<supervised|home-based|group|mixed|unknown>",
      "comparator_type": "<usual_care|medical_care|surgery|injection|wait_list|other_physiotherapy|education|other|unknown>",
      "comparator_detail": "<1 sentence description of comparator>",
      "delta_cost_direction": "<more_costly|less_costly|similar|unknown>",
      "delta_effect_direction": "<more_effective|less_effective|similar|unknown>",
      "icer": "<ICER value if reported e.g. £23456/QALY, or unknown>",
      "wtp_threshold": "<WTP threshold used e.g. £20000/QALY, or unknown>",
      "quadrant": "<dominant|dominated|NE|SW|unclear>",
      "ce_conclusion": "<cost_effective|not_cost_effective|inconclusive>",
      "notes": "<1-2 sentences explaining why this comparison landed in this quadrant>",
      "evidence_pages": "<page numbers used e.g. p.3; p.7>",
      "extraction_confidence": "<high|medium|low>"
    }}
  ]
}}

Rules for figure_group:
- Fig4 = physiotherapy compared to a NON-physiotherapy comparator (usual care, surgery, injection, medical care, GP, wait-list, sham)
- Fig5 = physiotherapy compared to ANOTHER physiotherapy modality (exercise vs manual therapy, etc.)

Rules for quadrant:
- dominant = intervention costs LESS AND is MORE effective (clear winner)
- dominated = intervention costs MORE AND is LESS effective (clear loser)
- NE = intervention costs MORE AND is MORE effective (trade-off; check ICER vs WTP)
- SW = intervention costs LESS AND is LESS effective (trade-off)
- unclear = cost and/or effect direction uncertain

Rules for extraction_confidence:
- high = ICER or cost/effect numbers explicitly reported
- medium = quadrant or cost-effectiveness stated in plain language without exact numbers
- low = inferred or ambiguous

Evidence excerpts:
{context}
"""


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _extract_json_block(raw: str) -> Any:
    """Extract the first valid JSON object or array from a raw LLM response."""
    # Try direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Find the outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass

    # Find the outermost [ ... ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass

    raise ValueError(f"No valid JSON found in LLM response: {raw[:200]}")


def _sanitize_comparison(c: Dict[str, Any], paper_id: str, idx: int) -> Dict[str, Any]:
    """Ensure all expected keys are present and values are strings/scalars."""
    defaults = {
        "comparison_index": idx,
        "figure_group": "unknown",
        "body_region": "unknown",
        "condition": "unknown",
        "country": "unknown",
        "setting": "unknown",
        "sample_size": "unknown",
        "study_design": "unknown",
        "time_horizon": "unknown",
        "perspective": "unknown",
        "outcome_type": "unknown",
        "outcome_measure": "unknown",
        "intervention_type": "unknown",
        "intervention_detail": "unknown",
        "frequency": "unknown",
        "total_sessions": "unknown",
        "session_length": "unknown",
        "duration_weeks": "unknown",
        "supervision": "unknown",
        "comparator_type": "unknown",
        "comparator_detail": "unknown",
        "delta_cost_direction": "unknown",
        "delta_effect_direction": "unknown",
        "icer": "unknown",
        "wtp_threshold": "unknown",
        "quadrant": "unclear",
        "ce_conclusion": "inconclusive",
        "notes": "",
        "evidence_pages": "",
        "extraction_confidence": "low",
    }
    out: Dict[str, Any] = {}
    for k, dv in defaults.items():
        v = c.get(k, dv)
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        elif v is None:
            v = dv
        out[k] = str(v).strip() if v != "" else ""

    out["paper_id"] = paper_id
    out["comparison_id"] = f"{paper_id}_c{idx}"
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def extract_comparisons(
    paper_id: str,
    context: str,
    env_path: str,
) -> List[Dict[str, Any]]:
    """
    Extract all CE comparisons from evidence excerpts for one paper.
    Returns a list of sanitized comparison dicts ready for ce_db.upsert_comparison().
    """
    llm = OllamaClient(env_path=env_path)

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": EXTRACTION_TEMPLATE.format(
                paper_id=paper_id,
                context=context,
            ),
        },
    ]

    raw = llm.chat(messages, temperature=0.0, think=False, timeout=600)

    try:
        parsed = _extract_json_block(raw)
    except Exception as e:
        raise ValueError(f"JSON parse error for {paper_id}: {e}\nRaw: {raw[:300]}")

    # Handle both {"comparisons": [...]} and direct [...]
    if isinstance(parsed, dict):
        comparisons_raw = parsed.get("comparisons", [parsed])
    elif isinstance(parsed, list):
        comparisons_raw = parsed
    else:
        comparisons_raw = [parsed]

    if not comparisons_raw:
        # Return a minimal placeholder so the paper is not lost
        comparisons_raw = [{}]

    result = []
    for i, c in enumerate(comparisons_raw, start=1):
        if not isinstance(c, dict):
            c = {}
        result.append(_sanitize_comparison(c, paper_id, i))

    return result


QUADRANT_FOCUS_TEMPLATE = """Paper ID: {paper_id}

A previous extraction could not determine the cost-effectiveness quadrant.
Focus ONLY on: which quadrant does this intervention fall in?

dominant = costs LESS AND is MORE effective
dominated = costs MORE AND is LESS effective
NE = costs MORE AND is MORE effective (may be CE if ICER < WTP threshold)
SW = costs LESS AND is LESS effective

Look specifically for:
- Explicit words: "dominant", "dominated", "cost-effective", "cost saving"
- Cost direction: "more costly", "less costly", "cost saving", "similar cost"
- Effect direction: "more effective", "better outcome", "similar effect", "no difference"
- Any ICER value (£/QALY, $/QALY, €/QALY)
- WTP threshold comparison

Return ONLY this JSON with no other text:
{{
  "delta_cost_direction": "<more_costly|less_costly|similar|unknown>",
  "delta_effect_direction": "<more_effective|less_effective|similar|unknown>",
  "icer": "<ICER value or unknown>",
  "quadrant": "<dominant|dominated|NE|SW|unclear>",
  "ce_conclusion": "<cost_effective|not_cost_effective|inconclusive>",
  "notes": "<1-2 sentences citing the key evidence>",
  "extraction_confidence": "<high|medium|low>"
}}

Evidence:
{context}
"""


def extract_quadrant_focused(paper_id: str, context: str, env_path: str) -> Dict[str, Any]:
    """Second-pass extraction focused only on quadrant determination for unclear papers."""
    llm = OllamaClient(env_path=env_path)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUADRANT_FOCUS_TEMPLATE.format(
            paper_id=paper_id, context=context,
        )},
    ]
    try:
        raw = llm.chat(messages, temperature=0.0, think=False, timeout=300)
        parsed = _extract_json_block(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def make_fallback_comparisons(paper_id: str, reason: str = "") -> List[Dict[str, Any]]:
    """Return a single placeholder comparison when extraction fails."""
    c = _sanitize_comparison({}, paper_id, 1)
    c["notes"] = reason[:200] if reason else "Extraction failed"
    return [c]
