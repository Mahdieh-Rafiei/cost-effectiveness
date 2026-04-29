"""
RAG (Retrieval-Augmented Generation) for single-paper and cross-paper Q&A.

Improvements over the original:
- Intent detection: question type shapes retrieval queries and system prompt
- Multi-query retrieval: up to 3 specialised queries merged and deduplicated
- Relevance filtering: chunks beyond a distance threshold are dropped
- Chunk deduplication: near-identical snippets removed before context build
- Source deduplication: unique paper+page pairs only
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from .vectorstore import VectorStore
from .ollama_client import OllamaClient
from .vision import is_figure_question, get_pages_for_question, render_page_as_base64


# ── Intent detection ─────────────────────────────────────────────────────────

_INTENT_CE = {
    "cost-effective", "cost effective", "icer", "dominant", "dominated",
    "quadrant", "cost per qaly", "incremental cost", "incremental effect",
    "net monetary benefit", "willingness to pay", "wtp",
}
_INTENT_INTERVENTION = {
    "intervention", "exercise", "manual therapy", "treatment", "program",
    "frequency", "sessions", "duration", "session length", "dose",
    "supervised", "home-based", "group",
}
_INTENT_OUTCOMES = {
    "qaly", "eq-5d", "sf-36", "sf-6d", "outcome", "effectiveness",
    "quality of life", "pain", "disability", "function", "womac", "vas",
}
_INTENT_DESIGN = {
    "study design", "rct", "randomized", "participants", "sample size",
    "country", "setting", "follow-up", "time horizon", "perspective",
    "societal", "healthcare",
}


def _detect_intent(question: str) -> str:
    q = question.lower()
    scores = {
        "ce":           sum(1 for k in _INTENT_CE           if k in q),
        "intervention": sum(1 for k in _INTENT_INTERVENTION if k in q),
        "outcomes":     sum(1 for k in _INTENT_OUTCOMES     if k in q),
        "design":       sum(1 for k in _INTENT_DESIGN       if k in q),
    }
    best = max(scores, key=lambda x: scores[x])
    return best if scores[best] > 0 else "general"


def _expand_queries(question: str, intent: str) -> List[str]:
    """Return 1-3 retrieval queries: the original plus intent-specific expansions."""
    queries = [question]

    if intent == "ce":
        queries.append(
            "ICER incremental cost-effectiveness ratio dominant dominated "
            "cost per QALY net monetary benefit willingness-to-pay"
        )
    elif intent == "intervention":
        queries.append(
            "intervention sessions per week total sessions session length "
            "duration weeks frequency supervised home-based physiotherapy"
        )
    elif intent == "outcomes":
        queries.append(
            "QALY EQ-5D SF-6D quality-adjusted life year primary outcome "
            "effectiveness pain disability function VAS WOMAC"
        )
    elif intent == "design":
        queries.append(
            "randomized controlled trial RCT participants sample size "
            "country setting time horizon perspective societal healthcare"
        )

    return queries


# ── System prompts ────────────────────────────────────────────────────────────

_BASE_SYSTEM = """You are an evidence-grounded research assistant specialising in physiotherapy cost-effectiveness.
Answer ONLY using the provided evidence excerpts.
Always cite sources as (PaperID p.X).
If something is not in the excerpts, say "Not reported in the retrieved excerpts."
"""

_EXTRA_BY_INTENT = {
    "ce": (
        "Focus on: ICER values, quadrant placement (dominant/dominated/NE/SW), "
        "cost and effect directions, willingness-to-pay thresholds, CE conclusions."
    ),
    "intervention": (
        "Focus on: intervention type (exercise/manual therapy/education/mixed), "
        "frequency (sessions/week), total sessions, session length, duration (weeks), "
        "supervision level (supervised/home-based/group)."
    ),
    "outcomes": (
        "Focus on: primary outcome measure (QALY, EQ-5D, WOMAC, VAS, NRS), "
        "magnitude of effect, statistical significance, clinical relevance."
    ),
    "design": (
        "Focus on: study design (RCT/observational), sample size, country, setting, "
        "time horizon, economic perspective (societal/healthcare/patient)."
    ),
    "general": (
        "Extract concrete facts where possible: number of participants, "
        "intervention content, which arm was more cost-effective and why."
    ),
}


def _build_system_prompt(intent: str) -> str:
    extra = _EXTRA_BY_INTENT.get(intent, _EXTRA_BY_INTENT["general"])
    return f"{_BASE_SYSTEM}\n{extra}"


# ── Context building with deduplication and relevance filtering ───────────────

# ChromaDB L2 distance for nomic-embed-text: 0–0.5 = very similar, 0.5–1.5 = relevant,
# 1.5–2.0 = loosely related. Use a generous threshold and rely on top-k for quality.
_DISTANCE_THRESHOLD = 1.8


def _build_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 14000,
    ignore_threshold: bool = False,
) -> str:
    seen_text: set = set()
    parts: List[str] = []
    used = 0

    for h in hits:
        # Relevance filter
        dist = h.get("distance", 0.0)
        if not ignore_threshold and dist > _DISTANCE_THRESHOLD:
            continue

        # Near-duplicate suppression (first 120 chars as fingerprint)
        fp = h["text"][:120].strip()
        if fp in seen_text:
            continue
        seen_text.add(fp)

        paper = h["meta"]["paper_id"]
        page  = h["meta"]["page"]
        block = f"[{paper} p.{page}] {h['text']}"

        if used + len(block) > max_chars:
            break

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


def _dedup_sources(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out = []
    for h in hits:
        key = (h["meta"]["paper_id"], h["meta"]["page"])
        if key not in seen:
            seen.add(key)
            out.append({"paper_id": h["meta"]["paper_id"], "page": h["meta"]["page"]})
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    store: VectorStore,
    env_path: Optional[str] = None,
    k: int = 8,
    paper_id: Optional[str] = None,
    chat_model: Optional[str] = None,
    pdf_dir: Optional[str] = None,
) -> dict:
    intent = _detect_intent(question)
    queries = _expand_queries(question, intent)
    where   = {"paper_id": paper_id} if paper_id else None

    # Multi-query retrieval — merge and deduplicate by text fingerprint
    all_hits: List[Dict[str, Any]] = []
    seen_fp: set = set()
    for query in queries:
        try:
            hits = store.query(query, k=k, where=where)
        except Exception:
            hits = []
        for h in hits:
            fp = h["text"][:120].strip()
            if fp not in seen_fp:
                seen_fp.add(fp)
                all_hits.append(h)

    # Sort by distance (best first)
    all_hits.sort(key=lambda h: h.get("distance", 0.0))

    if not all_hits:
        return {
            "answer": "I could not find relevant excerpts for this question.",
            "sources": [],
            "intent": intent,
        }

    context = _build_context(all_hits)

    if not context.strip():
        # Last resort: take top-3 hits regardless of distance
        all_hits.sort(key=lambda h: h.get("distance", 0.0))
        context = _build_context(all_hits[:3], max_chars=14000, ignore_threshold=True)

    if not context.strip():
        return {
            "answer": "I could not find relevant text for this question in the indexed papers. "
                      "Make sure the paper is ingested (run ingest_all.py) and try rephrasing.",
            "sources": [],
            "intent": intent,
        }

    llm = OllamaClient(env_path=env_path)

    # Vision augmentation: render PDF pages when asking about figures in a specific paper
    if is_figure_question(question) and paper_id and pdf_dir:
        pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
        print(f"[vision] checking PDF: {pdf_path} exists={pdf_path.exists()}")
        if not pdf_path.exists():
            matches = sorted(Path(pdf_dir).glob(f"{paper_id}*.pdf"))
            pdf_path = matches[0] if matches else None
            print(f"[vision] glob fallback: {pdf_path}")

        if pdf_path and pdf_path.exists():
            retrieved_pages = [h["meta"]["page"] for h in all_hits[:6]]
            pages_to_render = get_pages_for_question(
                str(pdf_path), question, retrieved_pages
            )
            images = [
                img for p in pages_to_render
                if (img := render_page_as_base64(str(pdf_path), p)) is not None
            ]
            print(f"[vision] pages={pages_to_render} images_rendered={len(images)}")
            if images:
                try:
                    # Supplement with database data for this paper if count is asked
                    db_context = ""
                    _count_kw = {"how many", "count", "number of", "dominant",
                                 "dominated", "quadrant", "cost-effective", "icer"}
                    if any(k in question.lower() for k in _count_kw) and paper_id:
                        from .ce_db import query_sql
                        rows = query_sql(
                            "SELECT * FROM ce_comparisons WHERE paper_id = ?",
                            (paper_id,)
                        )
                        if rows:
                            quad_counts = {}
                            for r in rows:
                                q = r.get("quadrant", "unclear")
                                quad_counts[q] = quad_counts.get(q, 0) + 1
                            db_context = (
                                f"\n\nExtracted data for this paper: "
                                f"{len(rows)} comparison(s). "
                                f"Quadrant counts: {quad_counts}. "
                                f"CE conclusion: "
                                f"{[r.get('ce_conclusion') for r in rows]}."
                            )
                    answer = llm.vision_chat(
                        question=question,
                        images_b64=images,
                        context=context[:2000] + db_context,
                        think=False,
                        timeout=120,
                    )
                    return {
                        "answer":  answer,
                        "sources": _dedup_sources(all_hits),
                        "intent":  intent,
                        "mode":    "vision",
                    }
                except Exception as e:
                    print(f"[vision] failed, falling back to RAG: {e}")

    messages = [
        {"role": "system", "content": _build_system_prompt(intent)},
        {"role": "user",   "content": f"Question:\n{question}\n\nEvidence excerpts:\n{context}"},
    ]
    answer = llm.chat(messages, temperature=0.1, model=chat_model)

    return {
        "answer":  answer,
        "sources": _dedup_sources(all_hits),
        "intent":  intent,
    }
