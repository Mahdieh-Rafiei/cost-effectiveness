"""
FastAPI backend for the Cost-Effectiveness Physiotherapy RAG system.

Endpoints:
  POST /ingest                  – index one PDF into Chroma
  POST /ask                     – RAG Q&A on one or all papers
  POST /ask_compare             – cross-paper comparison using ce_comparisons
  POST /build_ce_table          – extract CE data for all PDFs → DB
  GET  /list_papers             – list paper IDs in DB
  GET  /status                  – vector-store chunk count
  GET  /fig4                    – Fig 4 summary (physio vs non-physio)
  GET  /fig5                    – Fig 5 summary (physio vs physio)
  POST /body_region_analysis    – CE breakdown for one body region
  GET  /intervention_analysis   – intervention-dose vs quadrant table
  POST /compare_figures         – LLM explanation of Fig4 vs Fig5 differences
  POST /explain_drivers         – why are some studies CE and others not?
  GET  /export_comparisons      – all ce_comparisons rows as JSON
  GET  /dataset_summary         – high-level counts

  ── Next-step endpoints ──────────────────────────────────────────────────────
  GET  /validate_extraction/{paper_id}
       Are Tables 1 & 2 correctly extracted? Compares extracted DB fields
       against the raw source evidence from the vector store.

  POST /review_fig5_placements
       Have a look at Fig 5 — do you agree with each comparison's quadrant?
       Re-reads source evidence and asks LLM to agree/disagree with placement.

  POST /table2_by_region
       For a given body region, present Table-2-style intervention details
       (type, frequency, sessions, session length, duration) and explain
       whether CE vs non-CE interventions differ on those characteristics.
"""

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os

from .pdf_extract import extract_pdf_pages
from .chunking import make_chunks
from .vectorstore import VectorStore
from .rag import answer_question, answer_question_stream
from .ce_build import build_ce_table, rebuild_unclear_papers
from .ce_db import query_sql, init_db, get_comparisons_summary
from .ollama_client import OllamaClient
from .vision import (is_figure_question, get_pages_for_question,
                     render_page_as_base64, extract_figure_ref)

_default_env = Path(__file__).resolve().parent.parent.parent / ".env"
ENV_PATH = os.getenv("ENV_PATH", str(_default_env) if _default_env.exists() else None)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
VALIDATION_REPORT_PATH = DATA_DIR / "validation_report.json"

app = FastAPI(title="Cost-Effectiveness Physiotherapy RAG")
store = VectorStore(persist_dir=str(CHROMA_DIR), env_path=ENV_PATH, collection_name="papers")

# ── Global model state (changed at runtime via /set_model) ────────────────────
from dotenv import load_dotenv
load_dotenv(ENV_PATH)
_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b")


# ── Pydantic models ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    pdf_path: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    paper_id: Optional[str] = None
    history: List[dict] = []

class CompareRequest(BaseModel):
    question: str
    body_region: Optional[str] = None  # filter to specific region if given

class ComparePapersRequest(BaseModel):
    paper_id_a: str
    paper_id_b: str
    question: str = "Compare these two studies"

class BodyRegionRequest(BaseModel):
    body_region: str   # e.g. "knee", "shoulder", "low_back"

class DriverRequest(BaseModel):
    body_region: Optional[str] = None

class SetModelRequest(BaseModel):
    model: str


# ── Utility helpers ───────────────────────────────────────────────────────────

def _detect_condition(question: str) -> Optional[str]:
    q = question.lower()
    mapping = {
        "knee": "knee", "patell": "knee", "acl": "knee",
        "hip": "hip", "trochant": "hip",
        "shoulder": "shoulder", "rotator": "shoulder", "subacromial": "shoulder",
        "low back": "low_back", "lumbar": "low_back", "lbp": "low_back",
        "back": "low_back",
        "neck": "neck", "cervical": "neck",
        "ankle": "ankle", "achilles": "ankle",
        "elbow": "elbow", "epicondyl": "elbow",
        "wrist": "wrist",
    }
    for kw, region in mapping.items():
        if kw in q:
            return region
    return None


def _top_counts(rows: List[dict], key: str, top_n: int = 8) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for r in rows:
        v = str(r.get(key) or "unknown").strip().lower()
        counts[v] = counts.get(v, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    return [{"value": k, "count": v} for k, v in sorted_counts]


def _llm_narrative(messages: list, temperature: float = 0.1) -> str:
    llm = OllamaClient(env_path=ENV_PATH)
    return llm.chat(messages, temperature=temperature, model=_chat_model)


# ── Model management endpoints ────────────────────────────────────────────────

@app.get("/list_models")
def list_models():
    llm = OllamaClient(env_path=ENV_PATH)
    models = llm.list_models()
    return {"models": models, "current": _chat_model}


@app.get("/current_model")
def current_model():
    return {"model": _chat_model}


@app.post("/set_model")
def set_model(req: SetModelRequest):
    global _chat_model
    _chat_model = req.model.strip()
    return {"model": _chat_model, "status": "ok"}


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.post("/ingest")
def ingest(req: IngestRequest):
    pages = extract_pdf_pages(req.pdf_path)
    chunks = make_chunks(pages, max_chars=2500, overlap=300)
    store.add_chunks(chunks, batch_size=16)
    return {
        "status": "ok",
        "paper_id": Path(req.pdf_path).stem,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask(req: AskRequest):
    history = [{"role": h["role"], "content": h["content"]}
               for h in req.history if h.get("role") in ("user", "assistant")]
    return answer_question(
        req.question,
        store=store,
        env_path=ENV_PATH,
        k=req.top_k,
        paper_id=req.paper_id,
        chat_model=_chat_model,
        pdf_dir=str(PDF_DIR),
        history=history,
    )


@app.post("/ask_stream")
def ask_stream(req: AskRequest):
    history = [{"role": h["role"], "content": h["content"]}
               for h in req.history if h.get("role") in ("user", "assistant")]

    def generate():
        for chunk in answer_question_stream(
            req.question,
            store=store,
            env_path=ENV_PATH,
            paper_id=req.paper_id,
            chat_model=_chat_model,
            pdf_dir=str(PDF_DIR),
            history=history,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/status")
def status():
    return {"chunks_indexed": store.col.count()}


@app.post("/build_ce_table")
def build_ce():
    init_db()
    out = build_ce_table(store=store, env_path=ENV_PATH, pdf_dir=str(PDF_DIR))
    return out


@app.post("/rebuild_unclear")
def rebuild_unclear():
    """Second-pass extraction for papers still marked unclear quadrant."""
    return rebuild_unclear_papers(store=store, env_path=ENV_PATH)


@app.get("/paper_info")
def paper_info(paper_id: str):
    """Return metadata for a specific paper from the CE database."""
    rows = query_sql(
        "SELECT * FROM ce_comparisons WHERE paper_id = ?", (paper_id,)
    )
    if not rows:
        return {"found": False, "paper_id": paper_id}

    import re as _re
    year_m = _re.search(r'\b(19|20)\d{2}\b', paper_id)
    year = year_m.group(0) if year_m else "unknown"

    first = rows[0]
    quadrants = [r["quadrant"] for r in rows]
    icers = [r["icer"] for r in rows if r.get("icer") not in (None, "unknown", "")]
    conclusions = [r["ce_conclusion"] for r in rows]

    return {
        "found":            True,
        "paper_id":         paper_id,
        "year":             year,
        "body_region":      first.get("body_region", "unknown"),
        "condition":        first.get("condition", "unknown"),
        "perspective":      first.get("perspective", "unknown"),
        "time_horizon":     first.get("time_horizon", "unknown"),
        "intervention_type": first.get("intervention_type", "unknown"),
        "comparator_type":  first.get("comparator_type", "unknown"),
        "quadrants":        quadrants,
        "icer":             icers[0] if len(icers) == 1 else (icers if icers else "unknown"),
        "ce_conclusion":    conclusions[0] if len(set(conclusions)) == 1 else conclusions,
        "num_comparisons":  len(rows),
    }


@app.post("/compare_papers")
def compare_papers(req: ComparePapersRequest):
    """Side-by-side structured comparison of two papers."""
    llm = OllamaClient(env_path=ENV_PATH)

    def _paper_summary(pid: str) -> str:
        rows = query_sql("SELECT * FROM ce_comparisons WHERE paper_id = ?", (pid,))
        chunks = store.query(req.question, k=6, where={"paper_id": pid})
        text = "\n".join(f"[p.{h['meta']['page']}] {h['text']}" for h in chunks)
        db = ""
        if rows:
            r = rows[0]
            db = (
                f"Body region: {r.get('body_region','unknown')} | "
                f"Intervention: {r.get('intervention_type','unknown')} | "
                f"Comparator: {r.get('comparator_type','unknown')} | "
                f"ICER: {r.get('icer','unknown')} | "
                f"Quadrant: {r.get('quadrant','unclear')} | "
                f"CE conclusion: {r.get('ce_conclusion','inconclusive')} | "
                f"Time horizon: {r.get('time_horizon','unknown')} | "
                f"Perspective: {r.get('perspective','unknown')} | "
                f"Outcome: {r.get('outcome_measure','unknown')}"
            )
        return f"=== {pid} ===\nDatabase: {db}\n\nText excerpts:\n{text}"

    summary_a = _paper_summary(req.paper_id_a)
    summary_b = _paper_summary(req.paper_id_b)

    messages = [
        {"role": "system", "content": (
            "You are a health economics expert comparing two physiotherapy cost-effectiveness studies. "
            "Produce a structured side-by-side comparison as a markdown table followed by a narrative summary. "
            "Cover: study design, body region, intervention, comparator, ICER, quadrant, CE conclusion, time horizon, perspective, outcome measure. "
            "Be concise and factual. Use 'not reported' when data is missing."
        )},
        {"role": "user", "content": (
            f"Question: {req.question}\n\n"
            f"{summary_a}\n\n{summary_b}\n\n"
            "Produce: 1) A markdown comparison table, 2) A brief narrative (3-5 sentences) answering the question."
        )},
    ]
    answer = llm.chat(messages, temperature=0.1, model=_chat_model, timeout=180)
    return {"answer": answer, "paper_a": req.paper_id_a, "paper_b": req.paper_id_b}


@app.get("/list_papers")
def list_papers():
    # Papers with CE extractions
    rows = query_sql("SELECT DISTINCT paper_id FROM ce_comparisons ORDER BY paper_id")
    if not rows:
        rows = query_sql("SELECT DISTINCT paper_id FROM ce_studies ORDER BY paper_id")
    db_ids = {r["paper_id"] for r in rows}

    # All papers indexed in the vector store
    try:
        all_meta = store.col.get(include=["metadatas"])["metadatas"]
        vs_ids = {m["paper_id"] for m in all_meta if m.get("paper_id")}
    except Exception:
        vs_ids = set()

    combined = sorted(db_ids | vs_ids)
    return {"papers": combined}


@app.get("/export_comparisons")
def export_comparisons():
    rows = query_sql("SELECT * FROM ce_comparisons ORDER BY paper_id, comparison_id")
    return {"total": len(rows), "comparisons": rows}


# ── Figure 4 & 5 endpoints ────────────────────────────────────────────────────

def _fig_summary(figure_group: str) -> Dict[str, Any]:
    """
    Produce the data that replicates Fig 4 or Fig 5 from the review paper.

    Fig 4 = physiotherapy vs non-physiotherapy (usual care, surgery, injection…)
    Fig 5 = physiotherapy vs another physiotherapy modality
    """
    rows = query_sql(
        "SELECT * FROM ce_comparisons WHERE figure_group = ?",
        (figure_group,)
    )

    quadrant_map: Dict[str, List[dict]] = {
        "dominant": [], "NE": [], "SW": [], "dominated": [], "unclear": []
    }
    for r in rows:
        q = r.get("quadrant", "unclear")
        quadrant_map.setdefault(q, []).append(r)

    def _summary_list(lst: List[dict]) -> List[dict]:
        return [
            {
                "paper_id": r["paper_id"],
                "comparison_id": r["comparison_id"],
                "body_region": r.get("body_region"),
                "condition": r.get("condition"),
                "intervention_type": r.get("intervention_type"),
                "comparator_type": r.get("comparator_type"),
                "time_horizon": r.get("time_horizon"),
                "perspective": r.get("perspective"),
                "ce_conclusion": r.get("ce_conclusion"),
                "icer": r.get("icer"),
                "notes": r.get("notes"),
            }
            for r in lst
        ]

    return {
        "figure_group": figure_group,
        "total_comparisons": len(rows),
        "quadrant_counts": {k: len(v) for k, v in quadrant_map.items()},
        "by_quadrant": {k: _summary_list(v) for k, v in quadrant_map.items()},
        "by_body_region": _top_counts(rows, "body_region"),
        "by_intervention_type": _top_counts(rows, "intervention_type"),
        "by_comparator_type": _top_counts(rows, "comparator_type"),
        "by_perspective": _top_counts(rows, "perspective"),
        "by_time_horizon": _top_counts(rows, "time_horizon"),
    }


@app.get("/fig4")
def fig4_summary():
    return _fig_summary("Fig4")


@app.get("/fig5")
def fig5_summary():
    return _fig_summary("Fig5")


# ── Body-region analysis ──────────────────────────────────────────────────────

@app.post("/body_region_analysis")
def body_region_analysis(req: BodyRegionRequest):
    region = req.body_region.lower().strip()
    rows = query_sql(
        "SELECT * FROM ce_comparisons WHERE body_region = ?", (region,)
    )
    if not rows:
        return {
            "body_region": region,
            "message": "No comparisons found for this body region.",
            "available_regions": [
                r["body_region"]
                for r in query_sql(
                    "SELECT DISTINCT body_region FROM ce_comparisons ORDER BY body_region"
                )
            ],
        }

    ce_rows = [r for r in rows if r.get("ce_conclusion") == "cost_effective"]
    not_ce_rows = [r for r in rows if r.get("ce_conclusion") == "not_cost_effective"]
    unclear_rows = [r for r in rows if r.get("ce_conclusion") == "inconclusive"]

    # LLM explanation
    payload = {
        "body_region": region,
        "total": len(rows),
        "cost_effective": len(ce_rows),
        "not_cost_effective": len(not_ce_rows),
        "inconclusive": len(unclear_rows),
        "ce_intervention_types": _top_counts(ce_rows, "intervention_type"),
        "not_ce_intervention_types": _top_counts(not_ce_rows, "intervention_type"),
        "ce_time_horizons": _top_counts(ce_rows, "time_horizon"),
        "not_ce_time_horizons": _top_counts(not_ce_rows, "time_horizon"),
        "ce_perspectives": _top_counts(ce_rows, "perspective"),
        "ce_comparator_types": _top_counts(ce_rows, "comparator_type"),
        "not_ce_comparator_types": _top_counts(not_ce_rows, "comparator_type"),
        "ce_frequency": _top_counts(ce_rows, "frequency"),
        "not_ce_frequency": _top_counts(not_ce_rows, "frequency"),
        "ce_duration_weeks": _top_counts(ce_rows, "duration_weeks"),
        "not_ce_duration_weeks": _top_counts(not_ce_rows, "duration_weeks"),
        "ce_session_length": _top_counts(ce_rows, "session_length"),
        "ce_total_sessions": _top_counts(ce_rows, "total_sessions"),
        "not_ce_total_sessions": _top_counts(not_ce_rows, "total_sessions"),
        "example_ce_notes": [r["notes"] for r in ce_rows[:6] if r.get("notes")],
        "example_not_ce_notes": [r["notes"] for r in not_ce_rows[:6] if r.get("notes")],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health economics expert. "
                "Analyse the structured data provided and write a concise, evidence-grounded "
                "narrative explaining cost-effectiveness patterns for the given body region. "
                "Focus on: which interventions are cost-effective, which are not, "
                "and what study characteristics (type, dose, time horizon, comparator, perspective) "
                "appear to drive the difference. "
                "Do NOT invent data not present. "
                "Write 3-5 paragraphs. No markdown headers."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Body region: {region}\n\n"
                f"Structured data (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    narrative = _llm_narrative(messages)

    return {
        "body_region": region,
        "total_comparisons": len(rows),
        "cost_effective": len(ce_rows),
        "not_cost_effective": len(not_ce_rows),
        "inconclusive": len(unclear_rows),
        "quadrant_counts": _top_counts(rows, "quadrant"),
        "intervention_breakdown": {
            "cost_effective": _top_counts(ce_rows, "intervention_type"),
            "not_cost_effective": _top_counts(not_ce_rows, "intervention_type"),
        },
        "dose_breakdown": {
            "frequency_ce": _top_counts(ce_rows, "frequency"),
            "duration_weeks_ce": _top_counts(ce_rows, "duration_weeks"),
            "total_sessions_ce": _top_counts(ce_rows, "total_sessions"),
            "session_length_ce": _top_counts(ce_rows, "session_length"),
        },
        "narrative": narrative,
        "comparisons": rows,
    }


# ── Intervention analysis ─────────────────────────────────────────────────────

@app.get("/intervention_analysis")
def intervention_analysis(
    body_region: Optional[str] = Query(None),
    intervention_type: Optional[str] = Query(None),
):
    """
    Dose-response style analysis: how do intervention characteristics
    (frequency, duration, total sessions, session length) relate to quadrant placement?
    Optionally filter by body_region and/or intervention_type.
    """
    sql = "SELECT * FROM ce_comparisons WHERE 1=1"
    params: list = []
    if body_region:
        sql += " AND body_region = ?"
        params.append(body_region)
    if intervention_type:
        sql += " AND intervention_type = ?"
        params.append(intervention_type)

    rows = query_sql(sql, tuple(params))
    if not rows:
        return {"message": "No rows match these filters.", "rows": []}

    ce_rows = [r for r in rows if r.get("ce_conclusion") == "cost_effective"]
    not_ce_rows = [r for r in rows if r.get("ce_conclusion") == "not_cost_effective"]

    return {
        "filter": {"body_region": body_region, "intervention_type": intervention_type},
        "total": len(rows),
        "cost_effective": len(ce_rows),
        "not_cost_effective": len(not_ce_rows),
        "characteristics": {
            "frequency": {
                "cost_effective": _top_counts(ce_rows, "frequency"),
                "not_cost_effective": _top_counts(not_ce_rows, "frequency"),
            },
            "duration_weeks": {
                "cost_effective": _top_counts(ce_rows, "duration_weeks"),
                "not_cost_effective": _top_counts(not_ce_rows, "duration_weeks"),
            },
            "total_sessions": {
                "cost_effective": _top_counts(ce_rows, "total_sessions"),
                "not_cost_effective": _top_counts(not_ce_rows, "total_sessions"),
            },
            "session_length": {
                "cost_effective": _top_counts(ce_rows, "session_length"),
                "not_cost_effective": _top_counts(not_ce_rows, "session_length"),
            },
            "supervision": {
                "cost_effective": _top_counts(ce_rows, "supervision"),
                "not_cost_effective": _top_counts(not_ce_rows, "supervision"),
            },
            "time_horizon": {
                "cost_effective": _top_counts(ce_rows, "time_horizon"),
                "not_cost_effective": _top_counts(not_ce_rows, "time_horizon"),
            },
        },
        "rows": rows,
    }


# ── Compare figures ───────────────────────────────────────────────────────────

@app.post("/compare_figures")
def compare_figures():
    """
    LLM explanation of why Fig 4 and Fig 5 differ.
    Now also callable via /ask_compare when user asks about the figures in Chat.
    """
    fig4 = _fig_summary("Fig4")
    fig5 = _fig_summary("Fig5")

    if fig4["total_comparisons"] == 0 and fig5["total_comparisons"] == 0:
        return {
            "explanation": (
                "No data found in ce_comparisons. "
                "Please run /build_ce_table first to extract CE data from the papers."
            ),
            "fig4_summary": fig4,
            "fig5_summary": fig5,
        }

    payload = {
        "fig4": {
            "description": "Physiotherapy vs non-physiotherapy (usual care, surgery, injection, medical care)",
            "total": fig4["total_comparisons"],
            "quadrant_counts": fig4["quadrant_counts"],
            "top_body_regions": fig4["by_body_region"][:5],
            "top_interventions": fig4["by_intervention_type"][:5],
            "top_comparators": fig4["by_comparator_type"][:5],
        },
        "fig5": {
            "description": "Physiotherapy vs physiotherapy (different modalities compared head-to-head)",
            "total": fig5["total_comparisons"],
            "quadrant_counts": fig5["quadrant_counts"],
            "top_body_regions": fig5["by_body_region"][:5],
            "top_interventions": fig5["by_intervention_type"][:5],
            "top_comparators": fig5["by_comparator_type"][:5],
        },
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health economics expert specialising in musculoskeletal physiotherapy. "
                "Use the structured summary to explain the differences between Figure 4 and Figure 5 "
                "in the context of a systematic review of cost-effectiveness studies. "
                "Address: (1) what each figure represents conceptually, "
                "(2) why the quadrant distributions differ, "
                "(3) clinical and methodological reasons for the differences, "
                "(4) what these patterns mean for practice. "
                "Write 4-6 paragraphs. Be specific about the data. Do NOT invent data."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain the differences between Figure 4 and Figure 5.\n\n"
                f"Data (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    narrative = _llm_narrative(messages, temperature=0.15)
    return {
        "fig4_summary": fig4,
        "fig5_summary": fig5,
        "explanation": narrative,
    }


# ── Quadrant-driver explanation ───────────────────────────────────────────────

@app.post("/explain_drivers")
def explain_drivers(req: DriverRequest):
    """
    Explain which study characteristics drive cost-effectiveness quadrant placement.
    Optionally filter to one body region.
    """
    sql = "SELECT * FROM ce_comparisons WHERE 1=1"
    params: list = []
    if req.body_region:
        sql += " AND body_region = ?"
        params.append(req.body_region)

    rows = query_sql(sql, tuple(params))
    if not rows:
        return {"message": "No data found for the requested filter."}

    ce_rows = [r for r in rows if r.get("ce_conclusion") == "cost_effective"]
    not_ce_rows = [r for r in rows if r.get("ce_conclusion") == "not_cost_effective"]

    payload = {
        "scope": req.body_region or "all body regions",
        "total_comparisons": len(rows),
        "cost_effective": len(ce_rows),
        "not_cost_effective": len(not_ce_rows),
        "driver_data": {
            "time_horizon": {
                "ce": _top_counts(ce_rows, "time_horizon"),
                "not_ce": _top_counts(not_ce_rows, "time_horizon"),
            },
            "condition": {
                "ce": _top_counts(ce_rows, "condition"),
                "not_ce": _top_counts(not_ce_rows, "condition"),
            },
            "intervention_type": {
                "ce": _top_counts(ce_rows, "intervention_type"),
                "not_ce": _top_counts(not_ce_rows, "intervention_type"),
            },
            "comparator_type": {
                "ce": _top_counts(ce_rows, "comparator_type"),
                "not_ce": _top_counts(not_ce_rows, "comparator_type"),
            },
            "perspective": {
                "ce": _top_counts(ce_rows, "perspective"),
                "not_ce": _top_counts(not_ce_rows, "perspective"),
            },
            "outcome_type": {
                "ce": _top_counts(ce_rows, "outcome_type"),
                "not_ce": _top_counts(not_ce_rows, "outcome_type"),
            },
            "duration_weeks": {
                "ce": _top_counts(ce_rows, "duration_weeks"),
                "not_ce": _top_counts(not_ce_rows, "duration_weeks"),
            },
            "frequency": {
                "ce": _top_counts(ce_rows, "frequency"),
                "not_ce": _top_counts(not_ce_rows, "frequency"),
            },
            "body_region": {
                "ce": _top_counts(ce_rows, "body_region"),
                "not_ce": _top_counts(not_ce_rows, "body_region"),
            },
        },
        "ce_example_notes": [r["notes"] for r in ce_rows[:8] if r.get("notes")],
        "not_ce_example_notes": [r["notes"] for r in not_ce_rows[:8] if r.get("notes")],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health economics expert. "
                "Analyse the structured data and identify which study characteristics "
                "drive whether a physiotherapy intervention is cost-effective or not. "
                "Synthesise patterns across: time horizon, intervention type, dose (frequency/duration), "
                "condition, comparator type, perspective, and outcome type. "
                "Give specific, evidence-grounded statements like: "
                "'Exercise programs with longer time horizons (≥12 months) appear more often in the dominant quadrant.' "
                "Write 4-6 paragraphs. No markdown headers. Be specific."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scope: {req.body_region or 'all body regions'}\n\n"
                f"Data (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    narrative = _llm_narrative(messages, temperature=0.1)
    return {
        "scope": req.body_region or "all",
        "total_comparisons": len(rows),
        "cost_effective": len(ce_rows),
        "not_cost_effective": len(not_ce_rows),
        "driver_summary": payload["driver_data"],
        "narrative": narrative,
    }


# ── Cross-paper comparison (ask_compare) ─────────────────────────────────────
# One unified entry point for all cross-paper questions.
# Intent detection injects the right data automatically so the user just types
# naturally in Chat without navigating any menus.

def _q(s: str) -> str:
    return s.lower()

def _has(q: str, *keywords) -> bool:
    return any(k in q for k in keywords)


def _detect_intents(question: str, region: Optional[str]) -> Dict[str, bool]:
    q = _q(question)
    return {
        # Figure questions
        "fig4": _has(q, "fig 4", "fig4", "figure 4", "physio vs non",
                     "versus usual care", "versus surgery", "versus injection"),
        "fig5": _has(q, "fig 5", "fig5", "figure 5", "physio vs physio",
                     "head-to-head", "versus another physio"),
        "both_figs": _has(q, "compare figure", "difference between fig",
                          "fig 4 and fig 5", "both figure", "figure 4 and figure 5"),
        # Extraction validation
        "validate_extraction": _has(q, "correctly extracted", "extraction correct",
                                    "table 1", "table 2", "are these correct",
                                    "is it correct", "correctly identified",
                                    "check the extraction", "verify extraction",
                                    "did you extract", "is the content"),
        # Fig 5 placement review
        "review_placement": _has(q, "agree with", "do you agree", "placement",
                                 "location of", "correctly placed", "correct quadrant",
                                 "should it be") and _has(q, "fig 5", "fig5", "figure 5",
                                                          "comparison", "quadrant"),
        # Body-region CE difference
        "region_ce_diff": region is not None and _has(q, "difference", "differ",
                           "cost-effective", "cost effective", "is there",
                           "why", "which", "what", "compare"),
        # Table 2 / intervention characteristics by region
        "table2_region": region is not None and _has(q, "frequency", "sessions",
                         "duration", "length", "session length", "per week",
                         "how often", "how long", "type of", "table 2",
                         "dose", "intensity", "supervision", "intervention"),
    }


def _table2_summary(ce_rows: List[dict], not_ce_rows: List[dict]) -> Dict[str, Any]:
    """Build Table-2-style characteristic comparison."""
    return {
        "intervention_type":          {"ce": _top_counts(ce_rows, "intervention_type"),
                                       "not_ce": _top_counts(not_ce_rows, "intervention_type")},
        "frequency_per_week":         {"ce": _top_counts(ce_rows, "frequency"),
                                       "not_ce": _top_counts(not_ce_rows, "frequency")},
        "total_sessions":             {"ce": _top_counts(ce_rows, "total_sessions"),
                                       "not_ce": _top_counts(not_ce_rows, "total_sessions")},
        "session_length":             {"ce": _top_counts(ce_rows, "session_length"),
                                       "not_ce": _top_counts(not_ce_rows, "session_length")},
        "duration_weeks":             {"ce": _top_counts(ce_rows, "duration_weeks"),
                                       "not_ce": _top_counts(not_ce_rows, "duration_weeks")},
        "supervision":                {"ce": _top_counts(ce_rows, "supervision"),
                                       "not_ce": _top_counts(not_ce_rows, "supervision")},
        "time_horizon":               {"ce": _top_counts(ce_rows, "time_horizon"),
                                       "not_ce": _top_counts(not_ce_rows, "time_horizon")},
        "comparator_type":            {"ce": _top_counts(ce_rows, "comparator_type"),
                                       "not_ce": _top_counts(not_ce_rows, "comparator_type")},
        "ce_example_details":   [r.get("intervention_detail","") for r in ce_rows[:5]],
        "not_ce_example_details":[r.get("intervention_detail","") for r in not_ce_rows[:5]],
    }


def _summarize_compare_question(question: str, body_region: Optional[str] = None) -> Dict[str, Any]:
    region = body_region or _detect_condition(question)
    intents = _detect_intents(question, region)

    # ── Load all CE comparisons ──────────────────────────────────────────────
    all_rows = query_sql("SELECT * FROM ce_comparisons")
    if not all_rows:
        all_rows = query_sql("""
            SELECT paper_id, figure_group, condition AS body_region,
                   time_horizon, perspective, outcome_type, comparator_type,
                   quadrant, notes, evidence_pages,
                   NULL AS intervention_type, NULL AS duration_weeks,
                   NULL AS frequency, NULL AS ce_conclusion,
                   NULL AS intervention_detail, NULL AS total_sessions,
                   NULL AS session_length, NULL AS supervision
            FROM ce_studies
        """)

    if not all_rows:
        return {
            "answer": "No extracted data found. Please run /build_ce_table first to extract data from the papers.",
            "rows": [],
        }

    # Filter to body region if specified
    region_rows = all_rows
    if region:
        region_rows = [r for r in all_rows if str(r.get("body_region","")).lower() == region]
        if not region_rows:
            region_rows = all_rows  # fallback

    ce_rows     = [r for r in region_rows if str(r.get("ce_conclusion","")).startswith("cost_eff")
                   or str(r.get("quadrant","")).lower() == "dominant"]
    not_ce_rows = [r for r in region_rows if str(r.get("ce_conclusion","")) == "not_cost_effective"
                   or str(r.get("quadrant","")).lower() == "dominated"]

    # ── Base payload ─────────────────────────────────────────────────────────
    payload: Dict[str, Any] = {
        "question": question,
        "scope": region or "all body regions",
        "total_comparisons": len(region_rows),
        "cost_effective_n": len(ce_rows),
        "not_cost_effective_n": len(not_ce_rows),
        "overall_ce_by_intervention":  _top_counts(ce_rows,     "intervention_type"),
        "overall_nce_by_intervention": _top_counts(not_ce_rows, "intervention_type"),
        "overall_ce_by_comparator":    _top_counts(ce_rows,     "comparator_type"),
        "overall_nce_by_comparator":   _top_counts(not_ce_rows, "comparator_type"),
        "overall_ce_by_horizon":       _top_counts(ce_rows,     "time_horizon"),
        "overall_nce_by_horizon":      _top_counts(not_ce_rows, "time_horizon"),
        "overall_ce_by_perspective":   _top_counts(ce_rows,     "perspective"),
        "ce_notes_examples":   [r.get("notes","") for r in ce_rows[:5]    if r.get("notes")],
        "nce_notes_examples":  [r.get("notes","") for r in not_ce_rows[:5] if r.get("notes")],
    }

    # ── Inject data per detected intent ──────────────────────────────────────

    # Fig 4
    if intents["fig4"] or intents["both_figs"]:
        f4 = _fig_summary("Fig4")
        payload["fig4"] = {
            "what_it_is": "Physiotherapy vs NON-physiotherapy comparator (usual care, surgery, injection, GP, wait-list). Each data point = one intervention-comparator pair from a study.",
            "total_comparisons": f4["total_comparisons"],
            "quadrant_counts": f4["quadrant_counts"],
            "top_body_regions": f4["by_body_region"][:6],
            "top_interventions": f4["by_intervention_type"][:6],
            "top_comparators": f4["by_comparator_type"][:6],
            "dominant_examples": f4["by_quadrant"].get("dominant", [])[:4],
            "dominated_examples": f4["by_quadrant"].get("dominated", [])[:4],
            "NE_examples": f4["by_quadrant"].get("NE", [])[:4],
        }

    # Fig 5
    if intents["fig5"] or intents["both_figs"] or intents["review_placement"]:
        f5 = _fig_summary("Fig5")
        payload["fig5"] = {
            "what_it_is": "Physiotherapy vs ANOTHER physiotherapy modality (e.g. exercise vs manual therapy, head-to-head). Each data point = one comparison from a study.",
            "total_comparisons": f5["total_comparisons"],
            "quadrant_counts": f5["quadrant_counts"],
            "top_body_regions": f5["by_body_region"][:6],
            "top_interventions": f5["by_intervention_type"][:6],
            "top_comparators": f5["by_comparator_type"][:6],
            "all_comparisons_with_quadrant": [
                {
                    "paper_id": r["paper_id"],
                    "body_region": r.get("body_region"),
                    "intervention": r.get("intervention_type"),
                    "vs": r.get("comparator_type"),
                    "quadrant": r.get("quadrant"),
                    "delta_cost": r.get("delta_cost_direction"),
                    "delta_effect": r.get("delta_effect_direction"),
                    "icer": r.get("icer"),
                    "notes": r.get("notes","")[:150],
                }
                for comp_list in f5["by_quadrant"].values()
                for r in comp_list
            ],
        }

    # Table 1 & 2 extraction quality
    if intents["validate_extraction"]:
        sample = all_rows[:10]
        payload["extraction_quality_check"] = {
            "what_this_is": "Sample of extracted Table 1 & Table 2 fields from the papers",
            "total_papers_extracted": len({r["paper_id"] for r in all_rows}),
            "total_comparisons_extracted": len(all_rows),
            "fields_available": ["body_region","condition","country","setting",
                                  "sample_size","study_design","time_horizon",
                                  "perspective","outcome_type","outcome_measure",
                                  "intervention_type","frequency","total_sessions",
                                  "session_length","duration_weeks","supervision",
                                  "comparator_type","quadrant","icer","ce_conclusion"],
            "unknown_rate_by_field": {
                field: round(sum(1 for r in all_rows if str(r.get(field,"")) in ("unknown","","None")) / max(len(all_rows),1), 2)
                for field in ["body_region","time_horizon","perspective","outcome_type",
                              "intervention_type","frequency","total_sessions",
                              "session_length","duration_weeks","comparator_type","quadrant"]
            },
            "sample_rows": [
                {k: r.get(k) for k in ["paper_id","body_region","condition","time_horizon",
                                         "perspective","outcome_type","intervention_type",
                                         "frequency","total_sessions","session_length",
                                         "duration_weeks","comparator_type","quadrant",
                                         "ce_conclusion","extraction_confidence"]}
                for r in sample
            ],
        }

    # Body-region CE difference
    if intents["region_ce_diff"] and region:
        payload["region_ce_analysis"] = {
            "body_region": region,
            "total": len(region_rows),
            "cost_effective_n": len(ce_rows),
            "not_cost_effective_n": len(not_ce_rows),
            "ce_interventions":  _top_counts(ce_rows,     "intervention_type"),
            "nce_interventions": _top_counts(not_ce_rows, "intervention_type"),
            "ce_comparators":    _top_counts(ce_rows,     "comparator_type"),
            "nce_comparators":   _top_counts(not_ce_rows, "comparator_type"),
            "ce_horizons":       _top_counts(ce_rows,     "time_horizon"),
            "nce_horizons":      _top_counts(not_ce_rows, "time_horizon"),
            "ce_perspectives":   _top_counts(ce_rows,     "perspective"),
        }

    # Table 2 intervention characteristics by region
    if intents["table2_region"] and region:
        payload["table2_intervention_characteristics"] = {
            "body_region": region,
            "interpretation": "Compare cost_effective vs not_cost_effective groups on each characteristic",
            **_table2_summary(ce_rows, not_ce_rows),
        }

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = """You are a health economics expert who has analysed a systematic review of physiotherapy cost-effectiveness studies.
You have structured data extracted from all included papers in the JSON below.
Answer the user's question directly, specifically, and naturally — as if explaining to a colleague.

Guidelines:
- For Fig 4/5 questions: explain what the figure represents, report quadrant counts, identify patterns, explain why placements differ.
- For extraction quality questions: report the unknown/missing rates per field, comment on which fields are reliably vs. poorly captured.
- For Fig 5 placement review: go through the comparisons, comment on whether each quadrant assignment makes sense given cost/effect directions.
- For body-region CE questions: state clearly which interventions are cost-effective vs not, and why (comparator, horizon, perspective, dose).
- For Table 2 / intervention characteristics: identify which characteristics (frequency, duration, sessions, session length, type) differ between cost-effective and not-cost-effective groups.
- Be specific with numbers. Do NOT say 'based on the data'. Do NOT invent facts."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )},
    ]

    # ── Vision: add visual description of Figure 4/5 if PDF is available ────────
    if intents["fig4"] or intents["fig5"] or intents["both_figs"]:
        fig_num = extract_figure_ref(question) or ("4" if intents["fig4"] else "5")
        try:
            sr_pdfs = sorted(PDF_DIR.glob("*.pdf"),
                             key=lambda p: (
                                 "systematic" not in p.stem.lower()
                                 and "review" not in p.stem.lower()
                             ))
            if sr_pdfs:
                pdf_path = str(sr_pdfs[0])
                pages = get_pages_for_question(pdf_path, f"Figure {fig_num}", [], max_pages=3)
                images = [img for p in pages
                          if (img := render_page_as_base64(pdf_path, p)) is not None]
                if images:
                    llm_v = OllamaClient(env_path=ENV_PATH)
                    vision_desc = llm_v.vision_chat(
                        question=(f"Describe what Figure {fig_num} shows in this "
                                  f"cost-effectiveness paper page. Focus on axes, "
                                  f"data point distribution, and quadrants."),
                        images_b64=images,
                        think=False,
                        timeout=90,
                    )
                    payload["figure_visual_description"] = vision_desc
        except Exception as e:
            print(f"[vision/compare] {e}")

    answer = _llm_narrative(messages)
    return {"answer": answer, "rows": region_rows, "structured_summary": payload}


@app.post("/ask_compare")
def ask_compare(req: CompareRequest):
    return _summarize_compare_question(req.question, body_region=req.body_region)


# ── Dataset-level summary ─────────────────────────────────────────────────────

@app.get("/dataset_summary")
def dataset_summary():
    return get_comparisons_summary()


# ══════════════════════════════════════════════════════════════════════════════
# NEXT-STEP ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Validate Table 1 & 2 extraction for one paper ─────────────────────────

@app.get("/validate_extraction/{paper_id}")
def validate_extraction(paper_id: str):
    """
    Next-step Q: "Is the content of Tables 1 and 2 correctly extracted?"

    Re-reads the source evidence from the vector store and asks the LLM to
    compare it against what was stored in ce_comparisons. Returns a structured
    verdict (agree / partially agree / disagree) per field group.
    """
    # Pull extracted rows from DB
    rows = query_sql(
        "SELECT * FROM ce_comparisons WHERE paper_id = ?", (paper_id,)
    )
    if not rows:
        return {
            "paper_id": paper_id,
            "error": "No extracted data found for this paper. Run /build_ce_table first.",
        }

    # Re-query vector store for raw source evidence
    from .ce_build import _gather_context
    context = _gather_context(store, paper_id)
    if not context.strip():
        return {
            "paper_id": paper_id,
            "error": "No source chunks found in vector store for this paper.",
        }

    # Build field snapshot to validate (Table 1 + Table 2 fields only)
    first = rows[0]
    table1_fields = {k: first.get(k) for k in [
        "body_region", "condition", "country", "setting",
        "sample_size", "study_design", "time_horizon",
        "perspective", "outcome_type", "outcome_measure",
    ]}
    all_comparisons_table2 = [
        {
            "comparison_id": r["comparison_id"],
            "intervention_type": r.get("intervention_type"),
            "intervention_detail": r.get("intervention_detail"),
            "frequency": r.get("frequency"),
            "total_sessions": r.get("total_sessions"),
            "session_length": r.get("session_length"),
            "duration_weeks": r.get("duration_weeks"),
            "supervision": r.get("supervision"),
            "comparator_type": r.get("comparator_type"),
            "comparator_detail": r.get("comparator_detail"),
            "figure_group": r.get("figure_group"),
            "quadrant": r.get("quadrant"),
            "ce_conclusion": r.get("ce_conclusion"),
            "icer": r.get("icer"),
            "notes": r.get("notes"),
        }
        for r in rows
    ]

    validation_payload = {
        "paper_id": paper_id,
        "extracted_table1": table1_fields,
        "extracted_comparisons_table2": all_comparisons_table2,
        "n_comparisons": len(rows),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health economics expert performing a quality check on automatically "
                "extracted data from a cost-effectiveness paper. "
                "Compare the extracted fields against the source evidence excerpts. "
                "For each field group, state: CORRECT / PARTIALLY CORRECT / INCORRECT / CANNOT VERIFY. "
                "Be specific: quote the source when correcting a value. "
                "Structure your answer as:\n"
                "TABLE 1 FIELDS (study characteristics):\n"
                "- [field]: [verdict] — [reason or corrected value]\n\n"
                "TABLE 2 FIELDS (intervention details per comparison):\n"
                "- [field]: [verdict] — [reason or corrected value]\n\n"
                "OVERALL VERDICT: [high confidence / medium confidence / low confidence in extraction]\n"
                "SUGGESTED CORRECTIONS: [list any fields that should be changed]"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Paper: {paper_id}\n\n"
                f"Extracted data:\n{json.dumps(validation_payload, ensure_ascii=False, indent=2)}\n\n"
                f"Source evidence from the paper:\n{context}"
            ),
        },
    ]

    verdict = _llm_narrative(messages, temperature=0.0)

    return {
        "paper_id": paper_id,
        "n_comparisons": len(rows),
        "extracted_table1": table1_fields,
        "extracted_comparisons": all_comparisons_table2,
        "llm_verdict": verdict,
        "source_evidence_chars": len(context),
    }


# ── Validate individual paper against systematic review ───────────────────────

@app.get("/validate_vs_review/{paper_id}")
def validate_vs_review(paper_id: str):
    """
    Cross-validate: compare what the systematic review says about this paper
    against what the original paper actually reports and what the DB extracted.
    """
    import re as _re

    # Extract author + year from paper_id (format: "Author-YEAR-Title")
    parts = paper_id.split("-", 2)
    author = parts[0].strip() if parts else ""
    year_m = _re.search(r'\b(19|20)\d{2}\b', paper_id)
    year = year_m.group(0) if year_m else ""

    if not author or not year:
        return {"error": f"Cannot extract author/year from paper_id: {paper_id}"}

    # Find the systematic review paper ID
    sr_rows = query_sql(
        "SELECT DISTINCT paper_id FROM ce_comparisons "
        "WHERE lower(paper_id) LIKE '%systematic review%' "
        "   OR lower(paper_id) LIKE '%review of trial%'", ()
    )
    if not sr_rows:
        return {"error": "Systematic review paper not found in database."}
    sr_id = sr_rows[0]["paper_id"]

    # Retrieve systematic review chunks mentioning this author + year
    sr_hits = store.query(
        f"{author} {year} cost-effectiveness ICER intervention comparator",
        k=10, where={"paper_id": sr_id},
    )
    sr_context = "\n".join(
        f"[SR p.{h['meta']['page']}] {h['text']}" for h in sr_hits
    )

    # Retrieve original paper chunks
    paper_hits = store.query(
        "cost-effectiveness ICER intervention comparator study design sample size",
        k=10, where={"paper_id": paper_id},
    )
    paper_context = "\n".join(
        f"[p.{h['meta']['page']}] {h['text']}" for h in paper_hits
    )

    if not paper_context.strip():
        return {"error": f"No text found for paper: {paper_id}. Is it ingested?"}

    # DB extraction for this paper
    db_rows = query_sql("SELECT * FROM ce_comparisons WHERE paper_id = ?", (paper_id,))
    db_summary = {}
    if db_rows:
        r = db_rows[0]
        db_summary = {k: r.get(k) for k in [
            "body_region", "condition", "intervention_type", "comparator_type",
            "icer", "quadrant", "ce_conclusion", "time_horizon", "perspective",
            "outcome_measure", "frequency", "total_sessions", "duration_weeks",
        ]}

    llm = OllamaClient(env_path=ENV_PATH)
    messages = [
        {"role": "system", "content": (
            "You are validating a systematic review's accuracy for one specific study. "
            "You have three sources:\n"
            "1. What the systematic review (SR) says about this paper\n"
            "2. What the original paper actually reports\n"
            "3. What was automatically extracted into our database\n\n"
            "For each key field, state: CORRECT / INCORRECT / CANNOT VERIFY.\n"
            "Fields to check: body region, intervention, comparator, ICER, quadrant, "
            "time horizon, perspective, sample size, outcome measure.\n"
            "Quote the source when flagging an error. Be concise."
        )},
        {"role": "user", "content": (
            f"Paper: {paper_id}\n\n"
            f"=== SYSTEMATIC REVIEW text about {author} ({year}) ===\n{sr_context}\n\n"
            f"=== ORIGINAL PAPER text ===\n{paper_context}\n\n"
            f"=== DATABASE extraction ===\n{json.dumps(db_summary, indent=2)}\n\n"
            "Does the systematic review accurately represent this paper? "
            "Does the database extraction match both? Report field by field."
        )},
    ]

    verdict = _llm_narrative(messages, temperature=0.0)
    return {
        "paper_id":   paper_id,
        "author":     author,
        "year":       year,
        "sr_paper":   sr_id,
        "validation": verdict,
    }


# ── Batch validation of all papers against the systematic review ──────────────

_batch_validate_progress: dict = {"running": False, "done": 0, "total": 0, "errors": 0}


@app.post("/batch_validate")
def batch_validate():
    """
    Validate ALL individual papers against the systematic review.
    Saves results to data/validation_report.json.
    Returns immediately if already running.
    """
    global _batch_validate_progress
    if _batch_validate_progress["running"]:
        return {"status": "already_running", **_batch_validate_progress}

    import re as _re

    # Find SR paper
    sr_rows = query_sql(
        "SELECT DISTINCT paper_id FROM ce_comparisons "
        "WHERE lower(paper_id) LIKE '%systematic review%' "
        "   OR lower(paper_id) LIKE '%review of trial%'", ()
    )
    if not sr_rows:
        return {"error": "Systematic review paper not found in database."}
    sr_id = sr_rows[0]["paper_id"]

    # All other papers
    all_rows = query_sql("SELECT DISTINCT paper_id FROM ce_comparisons", ())
    paper_ids = [r["paper_id"] for r in all_rows if r["paper_id"] != sr_id]

    _batch_validate_progress = {
        "running": True, "done": 0, "total": len(paper_ids), "errors": 0
    }

    llm = OllamaClient(env_path=ENV_PATH)
    results = []

    for pid in paper_ids:
        print(f"[BATCH_VALIDATE] {pid}")
        try:
            parts = pid.split("-", 2)
            author = parts[0].strip() if parts else ""
            year_m = _re.search(r'\b(19|20)\d{2}\b', pid)
            year = year_m.group(0) if year_m else ""

            sr_hits = store.query(
                f"{author} {year} cost-effectiveness ICER intervention",
                k=6, where={"paper_id": sr_id},
            )
            sr_ctx = "\n".join(f"[SR p.{h['meta']['page']}] {h['text']}" for h in sr_hits)

            paper_hits = store.query(
                "cost-effectiveness ICER intervention comparator study design",
                k=6, where={"paper_id": pid},
            )
            paper_ctx = "\n".join(f"[p.{h['meta']['page']}] {h['text']}" for h in paper_hits)

            db_rows = query_sql("SELECT * FROM ce_comparisons WHERE paper_id = ?", (pid,))
            db = {}
            if db_rows:
                r = db_rows[0]
                db = {k: r.get(k) for k in [
                    "body_region", "intervention_type", "comparator_type",
                    "icer", "quadrant", "ce_conclusion", "time_horizon", "perspective",
                ]}

            messages = [
                {"role": "system", "content": (
                    "You are validating a systematic review entry for one study. "
                    "Reply in JSON only:\n"
                    '{"overall": "correct|partial|incorrect|cannot_verify", '
                    '"fields": {"body_region":"correct|incorrect|cannot_verify", '
                    '"intervention":"correct|incorrect|cannot_verify", '
                    '"comparator":"correct|incorrect|cannot_verify", '
                    '"icer":"correct|incorrect|cannot_verify", '
                    '"quadrant":"correct|incorrect|cannot_verify"}, '
                    '"issues": ["list of specific errors found"], '
                    '"note": "one sentence summary"}'
                )},
                {"role": "user", "content": (
                    f"Paper: {pid} (Author: {author}, Year: {year})\n\n"
                    f"Systematic review text:\n{sr_ctx}\n\n"
                    f"Original paper text:\n{paper_ctx}\n\n"
                    f"Database extraction:\n{json.dumps(db)}"
                )},
            ]

            raw = llm.chat(messages, temperature=0.0, model=_chat_model, timeout=120)

            # Parse JSON from response
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                verdict = json.loads(raw[start:end]) if start >= 0 else {"overall": "cannot_verify", "note": raw[:200]}
            except Exception:
                verdict = {"overall": "cannot_verify", "note": raw[:200]}

            results.append({"paper_id": pid, "author": author, "year": year, **verdict})
            _batch_validate_progress["done"] += 1

        except Exception as e:
            print(f"  ✗ {e}")
            results.append({"paper_id": pid, "overall": "error", "note": str(e)})
            _batch_validate_progress["errors"] += 1
            _batch_validate_progress["done"] += 1

        import time as _time
        _time.sleep(3)

    # Aggregate summary
    counts = {}
    for r in results:
        v = r.get("overall", "unknown")
        counts[v] = counts.get(v, 0) + 1

    report = {
        "total_papers": len(paper_ids),
        "summary": counts,
        "results": results,
    }

    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    _batch_validate_progress["running"] = False
    print(f"[BATCH_VALIDATE] Done. Summary: {counts}")
    return {"status": "ok", **report}


@app.get("/batch_validate_status")
def batch_validate_status():
    """Check batch validation progress."""
    return _batch_validate_progress


@app.get("/batch_validate_report")
def batch_validate_report():
    """Return saved validation report."""
    if not VALIDATION_REPORT_PATH.exists():
        return {"error": "No report found. Run /batch_validate first."}
    return json.loads(VALIDATION_REPORT_PATH.read_text())


# ── 2. Review Fig 5 placements ────────────────────────────────────────────────

class ReviewFig5Request(BaseModel):
    paper_id: Optional[str] = None   # review one paper; None = review all Fig 5 papers


@app.post("/review_fig5_placements")
def review_fig5_placements(req: ReviewFig5Request):
    """
    Next-step Q: "Have a look at Figure 5 — do you agree with the location of each comparison?"

    For each Fig 5 comparison (physio vs physio), re-reads source evidence and
    asks the LLM whether the quadrant assignment is justified.
    Returns a per-comparison verdict: AGREE / UNCERTAIN / DISAGREE + reason.
    """
    sql = "SELECT * FROM ce_comparisons WHERE figure_group = 'Fig5'"
    params: tuple = ()
    if req.paper_id:
        sql += " AND paper_id = ?"
        params = (req.paper_id,)

    rows = query_sql(sql, params)
    if not rows:
        return {"message": "No Fig 5 comparisons found.", "reviews": []}

    from .ce_build import _gather_context

    reviews = []
    for row in rows:
        pid = row["paper_id"]
        cid = row["comparison_id"]

        context = _gather_context(store, pid)

        comp_summary = {
            "intervention_type": row.get("intervention_type"),
            "intervention_detail": row.get("intervention_detail"),
            "comparator_type": row.get("comparator_type"),
            "comparator_detail": row.get("comparator_detail"),
            "body_region": row.get("body_region"),
            "condition": row.get("condition"),
            "time_horizon": row.get("time_horizon"),
            "perspective": row.get("perspective"),
            "delta_cost_direction": row.get("delta_cost_direction"),
            "delta_effect_direction": row.get("delta_effect_direction"),
            "icer": row.get("icer"),
            "quadrant": row.get("quadrant"),
            "ce_conclusion": row.get("ce_conclusion"),
            "extracted_notes": row.get("notes"),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a health economics expert reviewing the quadrant placement "
                    "of a physiotherapy comparison on the cost-effectiveness plane. "
                    "Quadrant definitions:\n"
                    "  dominant = intervention LESS costly AND MORE effective\n"
                    "  dominated = intervention MORE costly AND LESS effective\n"
                    "  NE = MORE costly AND MORE effective (trade-off)\n"
                    "  SW = LESS costly AND LESS effective (trade-off)\n"
                    "  unclear = insufficient evidence for classification\n\n"
                    "Based on the source evidence, state:\n"
                    "VERDICT: AGREE / UNCERTAIN / DISAGREE\n"
                    "REASON: (1-3 sentences citing specific evidence)\n"
                    "CORRECT QUADRANT (if DISAGREE): dominant | dominated | NE | SW | unclear\n"
                    "Be concise and specific."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper: {pid} | Comparison: {cid}\n"
                    f"Assigned quadrant: {row.get('quadrant')}\n\n"
                    f"Extracted comparison data:\n{json.dumps(comp_summary, ensure_ascii=False, indent=2)}\n\n"
                    f"Source evidence:\n{context[:8000]}"
                ),
            },
        ]

        verdict_text = _llm_narrative(messages, temperature=0.0)

        # Parse simple verdict line
        verdict_tag = "UNCERTAIN"
        for line in verdict_text.splitlines():
            line_u = line.strip().upper()
            if line_u.startswith("VERDICT:"):
                tag = line_u.replace("VERDICT:", "").strip()
                if "AGREE" in tag and "DIS" not in tag:
                    verdict_tag = "AGREE"
                elif "DISAGREE" in tag:
                    verdict_tag = "DISAGREE"
                break

        reviews.append({
            "paper_id": pid,
            "comparison_id": cid,
            "assigned_quadrant": row.get("quadrant"),
            "body_region": row.get("body_region"),
            "intervention_type": row.get("intervention_type"),
            "comparator_type": row.get("comparator_type"),
            "verdict": verdict_tag,
            "llm_review": verdict_text,
        })

    n_agree = sum(1 for r in reviews if r["verdict"] == "AGREE")
    n_disagree = sum(1 for r in reviews if r["verdict"] == "DISAGREE")
    n_uncertain = sum(1 for r in reviews if r["verdict"] == "UNCERTAIN")

    return {
        "fig5_comparisons_reviewed": len(reviews),
        "agree": n_agree,
        "disagree": n_disagree,
        "uncertain": n_uncertain,
        "reviews": reviews,
    }


# ── 3. Table 2 characteristics by body region ─────────────────────────────────

class Table2RegionRequest(BaseModel):
    body_region: str   # e.g. "knee", "shoulder", "hip", "low_back"


@app.post("/table2_by_region")
def table2_by_region(req: Table2RegionRequest):
    """
    Next-step Q: "In the length/type/frequency/duration of the intervention
    for knee/hip/shoulder (separately) — is there a difference between
    cost-effective and non-cost-effective interventions?"

    Returns a Table-2-style breakdown of all intervention characteristics
    for the given body region, split by CE outcome, plus an LLM narrative
    identifying the key differentiators.
    """
    region = req.body_region.lower().strip()
    rows = query_sql(
        "SELECT * FROM ce_comparisons WHERE body_region = ?", (region,)
    )
    if not rows:
        available = [
            r["body_region"]
            for r in query_sql(
                "SELECT DISTINCT body_region FROM ce_comparisons ORDER BY body_region"
            )
        ]
        return {
            "body_region": region,
            "error": f"No data for '{region}'.",
            "available_regions": available,
        }

    ce_rows = [r for r in rows if r.get("ce_conclusion") == "cost_effective"]
    not_ce_rows = [r for r in rows if r.get("ce_conclusion") == "not_cost_effective"]
    unclear_rows = [r for r in rows if r.get("ce_conclusion") == "inconclusive"]

    def _table2_rows(lst: List[dict]) -> List[dict]:
        return [
            {
                "paper_id": r["paper_id"],
                "intervention_type": r.get("intervention_type"),
                "intervention_detail": r.get("intervention_detail"),
                "frequency": r.get("frequency"),
                "total_sessions": r.get("total_sessions"),
                "session_length": r.get("session_length"),
                "duration_weeks": r.get("duration_weeks"),
                "supervision": r.get("supervision"),
                "comparator_type": r.get("comparator_type"),
                "time_horizon": r.get("time_horizon"),
                "quadrant": r.get("quadrant"),
                "icer": r.get("icer"),
                "notes": r.get("notes"),
            }
            for r in lst
        ]

    table2_ce = _table2_rows(ce_rows)
    table2_not_ce = _table2_rows(not_ce_rows)

    # Statistical summaries for the LLM
    payload = {
        "body_region": region,
        "total_comparisons": len(rows),
        "cost_effective_n": len(ce_rows),
        "not_cost_effective_n": len(not_ce_rows),
        "inconclusive_n": len(unclear_rows),
        "intervention_type": {
            "cost_effective": _top_counts(ce_rows, "intervention_type"),
            "not_cost_effective": _top_counts(not_ce_rows, "intervention_type"),
        },
        "frequency_sessions_per_week": {
            "cost_effective": _top_counts(ce_rows, "frequency"),
            "not_cost_effective": _top_counts(not_ce_rows, "frequency"),
        },
        "total_sessions": {
            "cost_effective": _top_counts(ce_rows, "total_sessions"),
            "not_cost_effective": _top_counts(not_ce_rows, "total_sessions"),
        },
        "session_length": {
            "cost_effective": _top_counts(ce_rows, "session_length"),
            "not_cost_effective": _top_counts(not_ce_rows, "session_length"),
        },
        "duration_weeks": {
            "cost_effective": _top_counts(ce_rows, "duration_weeks"),
            "not_cost_effective": _top_counts(not_ce_rows, "duration_weeks"),
        },
        "supervision": {
            "cost_effective": _top_counts(ce_rows, "supervision"),
            "not_cost_effective": _top_counts(not_ce_rows, "supervision"),
        },
        "time_horizon": {
            "cost_effective": _top_counts(ce_rows, "time_horizon"),
            "not_cost_effective": _top_counts(not_ce_rows, "time_horizon"),
        },
        "comparator_type": {
            "cost_effective": _top_counts(ce_rows, "comparator_type"),
            "not_cost_effective": _top_counts(not_ce_rows, "comparator_type"),
        },
        "ce_example_notes": [r.get("notes", "") for r in ce_rows[:5]],
        "not_ce_example_notes": [r.get("notes", "") for r in not_ce_rows[:5]],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health economics expert synthesising intervention characteristics "
                "from a systematic review of physiotherapy cost-effectiveness studies. "
                "Your task: for the given body region, identify which Table-2 characteristics "
                "(intervention type, frequency, total sessions, session length, duration, supervision, "
                "time horizon, comparator) differ between cost-effective and non-cost-effective interventions. "
                "Use the data precisely. "
                "Format your answer as numbered points, each covering one characteristic. "
                "End with a 2-sentence overall conclusion."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Body region: {region}\n\n"
                f"Table 2 summary data (JSON):\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    narrative = _llm_narrative(messages, temperature=0.05)

    return {
        "body_region": region,
        "total_comparisons": len(rows),
        "cost_effective_n": len(ce_rows),
        "not_cost_effective_n": len(not_ce_rows),
        "inconclusive_n": len(unclear_rows),
        "table2_cost_effective": table2_ce,
        "table2_not_cost_effective": table2_not_ce,
        "characteristic_comparison": payload,
        "llm_narrative": narrative,
    }
