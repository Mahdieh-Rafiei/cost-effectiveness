from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import re
import json

from .pdf_extract import extract_pdf_pages
from .chunking import make_chunks
from .vectorstore import VectorStore
from .rag import answer_question
from .ce_build import build_ce_table
from .ce_db import query_sql
from .ollama_client import OllamaClient

ENV_PATH = "/Users/mahdie/Documents/PROBE/Data_mapping/.env"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"

app = FastAPI(title="Cost-Effectiveness Paper QA")
store = VectorStore(persist_dir=str(CHROMA_DIR), env_path=ENV_PATH, collection_name="papers")


# -----------------------------
# Utilities
# -----------------------------
def detect_condition(question: str):
    q = question.lower()

    # map user terms to broad groups used in extracted table
    if "knee" in q or "hip" in q or "ankle" in q or "foot" in q:
        return "lower limb"
    if "shoulder" in q or "elbow" in q or "wrist" in q or "hand" in q:
        return "upper limb"
    if "spine" in q or "back" in q or "neck" in q:
        return "spine"
    if "upper limb" in q:
        return "upper limb"
    if "lower limb" in q:
        return "lower limb"
    return None


def normalize_condition_for_matching(raw: str) -> str:
    """Normalize condition text for fuzzy matching."""
    x = (raw or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = x.replace("-", " ")

    # map common variants to broad categories
    if any(k in x for k in ["knee", "hip", "ankle", "foot", "lower limb", "lower extrem", "patell", "acl"]):
        return "lower limb"
    if any(k in x for k in ["shoulder", "elbow", "wrist", "hand", "upper limb", "upper extrem"]):
        return "upper limb"
    if any(k in x for k in ["spine", "back", "neck", "lumbar", "cervical"]):
        return "spine"
    return x if x else "unknown"


def top_counts(rows, key, top_n=5):
    counts = {}
    for r in rows:
        v = str(r.get(key) or "unknown").strip().lower()
        v = re.sub(r"\s+", " ", v)
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]


# -----------------------------
# API Models
# -----------------------------
class IngestRequest(BaseModel):
    pdf_path: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    paper_id: Optional[str] = None

class CompareRequest(BaseModel):
    question: str


# -----------------------------
# Core endpoints
# -----------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    pages = extract_pdf_pages(req.pdf_path)
    chunks = make_chunks(pages, max_chars=2500, overlap=300)
    store.add_chunks(chunks, batch_size=16)
    return {
        "status": "ok",
        "paper_id": Path(req.pdf_path).stem,
        "pages": len(pages),
        "chunks": len(chunks)
    }

@app.post("/ask")
def ask(req: AskRequest):
    out = answer_question(
        req.question,
        store=store,
        env_path=ENV_PATH,
        k=req.top_k,
        paper_id=req.paper_id,  # pass filter through
    )
    return out

@app.get("/status")
def status():
    n = store.col.count()
    return {"chunks_indexed": n}

@app.post("/build_ce_table")
def build_ce():
    out = build_ce_table(store=store, env_path=ENV_PATH, pdf_dir=str(PDF_DIR))
    return out

@app.get("/list_papers")
def list_papers():
    rows = query_sql("SELECT DISTINCT paper_id FROM ce_studies ORDER BY paper_id")
    return {"papers": [r["paper_id"] for r in rows]}


# -----------------------------
# Cross-paper compare logic
# -----------------------------
def summarize_compare_question(question: str, env_path: str):
    condition = detect_condition(question)

    # Pull all rows first (safer), then filter in Python with normalization
    rows = query_sql("""
        SELECT paper_id, figure_group, condition, time_horizon, perspective, outcome_type,
               comparator_type, quadrant, notes, evidence_pages
        FROM ce_studies
    """)

    if not rows:
        return {
            "answer": "I could not find any extracted cross-paper records. Please run /build_ce_table first.",
            "rows": []
        }

    # Add normalized condition to each row for matching
    for r in rows:
        r["_condition_norm"] = normalize_condition_for_matching(str(r.get("condition", "")))

    filtered_rows = rows
    if condition:
        filtered_rows = [r for r in rows if r["_condition_norm"] == condition]

        # fallback: if no exact normalized match, use all rows (and explain limitation)
        if not filtered_rows:
            filtered_rows = rows
            condition_fallback_note = (
                f"I could not find rows explicitly labeled as '{condition}', "
                f"so I used all extracted studies instead."
            )
        else:
            condition_fallback_note = None
    else:
        condition_fallback_note = None

    # Split groups
    ce_rows = []
    less_rows = []
    for r in filtered_rows:
        q = str(r.get("quadrant", "")).strip().lower()
        if q in {"dominant", "ne"}:
            ce_rows.append(r)
        else:
            less_rows.append(r)

    # Build structured summary for LLM
    summary_payload = {
        "question": question,
        "condition_filter_requested": condition or "all",
        "condition_fallback_note": condition_fallback_note,
        "n_total_filtered": len(filtered_rows),
        "n_cost_effective": len(ce_rows),
        "n_less_cost_effective": len(less_rows),
        "cost_effective_top_conditions": top_counts(ce_rows, "_condition_norm"),
        "less_cost_effective_top_conditions": top_counts(less_rows, "_condition_norm"),
        "cost_effective_top_intervention_types": top_counts(ce_rows, "comparator_type"),
        "less_cost_effective_top_intervention_types": top_counts(less_rows, "comparator_type"),
        "cost_effective_top_perspectives": top_counts(ce_rows, "perspective"),
        "less_cost_effective_top_perspectives": top_counts(less_rows, "perspective"),
        "cost_effective_top_horizons": top_counts(ce_rows, "time_horizon"),
        "less_cost_effective_top_horizons": top_counts(less_rows, "time_horizon"),
        "cost_effective_top_outcomes": top_counts(ce_rows, "outcome_type"),
        "less_cost_effective_top_outcomes": top_counts(less_rows, "outcome_type"),
        "example_cost_effective_notes": [r["notes"] for r in ce_rows[:8] if r.get("notes")],
        "example_less_cost_effective_notes": [r["notes"] for r in less_rows[:8] if r.get("notes")],
        "example_cost_effective_papers": [r["paper_id"] for r in ce_rows[:8]],
        "example_less_cost_effective_papers": [r["paper_id"] for r in less_rows[:8]],
    }

    # If user asks simple count questions, we can still use LLM for narrative but include exact counts
    llm = OllamaClient(env_path=env_path)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-grounded research assistant. "
                "Answer directly and naturally. "
                "Do NOT start with phrases like 'According to the excerpts' or 'Based on the provided excerpts'. "
                "Use ONLY the structured summary provided. "
                "Write a concise narrative answer (not JSON). "
                "If the data quality is limited, explicitly say so. "
                "If a condition filter had no exact match, mention the fallback note if present. "
                "Do not invent data."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Structured summary (JSON):\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    answer = llm.chat(messages, temperature=0.1)

    # Remove internal helper key from debug rows
    debug_rows = []
    for r in filtered_rows:
        rr = dict(r)
        rr.pop("_condition_norm", None)
        debug_rows.append(rr)

    return {
        "answer": answer,
        "rows": debug_rows  # UI can choose to hide/show
    }


@app.post("/ask_compare")
def ask_compare(req: CompareRequest):
    return summarize_compare_question(req.question, env_path=ENV_PATH)
