import json
from typing import Dict, Any, List
from .ollama_client import OllamaClient

SYSTEM = """You extract structured information from evidence excerpts of a cost-effectiveness paper.
Return ONLY valid JSON (no markdown).
If a field is not supported by evidence, return "unknown".
Use short values, not long paragraphs.
"""

USER_TEMPLATE = """Paper ID: {paper_id}

Task:
From the evidence excerpts, extract these fields:
- figure_group: "Fig4" if physiotherapy vs non-physiotherapy comparator, "Fig5" if physiotherapy vs physiotherapy comparator, else "unknown"
- condition: "spine" / "lower_limb" / "upper_limb" / "other" / "unknown"
- time_horizon: e.g., "12 months", "1 year", "6 weeks", else "unknown"
- perspective: "societal" / "healthcare" / "unknown"
- outcome_type: "QALY" / "clinical" / "unknown"
- comparator_type: short description (e.g., "usual care", "medical doctor", "injection", "other physiotherapy", "surgery"), else "unknown"
- quadrant: "dominant" / "dominated" / "NE" / "SW" / "unclear"
- notes: 1–2 short sentences explaining why quadrant might be as reported (only if evidence supports it)
- evidence_pages: list the cited pages you used, like "p.3; p.7"

Evidence excerpts:
{context}

Return JSON with exactly these keys:
paper_id, figure_group, condition, time_horizon, perspective, outcome_type, comparator_type, quadrant, notes, evidence_pages
"""

def extract_ce_fields(paper_id: str, context: str, env_path: str) -> Dict[str, Any]:
    llm = OllamaClient(env_path=env_path)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(paper_id=paper_id, context=context)}
    ]
    raw = llm.chat(messages, temperature=0.0)

    # Robust JSON parse
    try:
        data = json.loads(raw)
    except Exception:
        # last resort: try to find JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise

    # Ensure paper_id
    data["paper_id"] = paper_id

    # ✅ sanitize values so SQLite won't crash on list/dict
    import json as _json
    for k, v in list(data.items()):
        if isinstance(v, (list, dict)):
            data[k] = _json.dumps(v, ensure_ascii=False)
        elif v is None:
            data[k] = "unknown"
        else:
            data[k] = str(v)

    return data

