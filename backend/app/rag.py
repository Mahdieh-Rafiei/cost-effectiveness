"""
RAG (Retrieval-Augmented Generation) for single-paper and cross-paper Q&A.

- Intent detection: question type shapes retrieval queries and system prompt
- Multi-query retrieval: up to 3 specialised queries merged and deduplicated
- Relevance filtering: chunks beyond a distance threshold are dropped
- Chunk deduplication: near-identical snippets removed before context build
- Adaptive k + model: automatically tuned to question intent
- Vision augmentation: always-on for single-paper mode (tables, figures, text)
- Conversation history: last N exchanges included for follow-up questions
- Streaming: answer_question_stream() yields tokens for real-time display
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

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


# ── Adaptive retrieval settings ───────────────────────────────────────────────

_INTENT_K = {
    "ce":           6,
    "intervention": 8,
    "outcomes":     8,
    "design":       8,
    "general":      12,
}

_INTENT_MODEL = {
    "ce":           "llama3.1:8b",
    "intervention": "llama3.1:8b",
    "outcomes":     "llama3.1:8b",
    "design":       "llama3.1:8b",
    "general":      "qwen3.5:9b",
}


def _is_review_paper(paper_id: Optional[str]) -> bool:
    if not paper_id:
        return False
    pid = paper_id.lower()
    return "systematic review" in pid or "review of trial" in pid


def _adaptive_settings(question: str, intent: str, paper_id: Optional[str]) -> tuple:
    if _is_review_paper(paper_id):
        return 20, _INTENT_MODEL.get("general")
    k = _INTENT_K.get(intent, 8)
    model = _INTENT_MODEL.get(intent, None)
    if not paper_id:
        k = max(k, 12)
    return k, model


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


# ── Context building ──────────────────────────────────────────────────────────

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
        dist = h.get("distance", 0.0)
        if not ignore_threshold and dist > _DISTANCE_THRESHOLD:
            continue
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


# ── Shared RAG preparation ────────────────────────────────────────────────────

def _build_rag_inputs(
    question: str,
    store: VectorStore,
    env_path: Optional[str],
    paper_id: Optional[str],
    pdf_dir: Optional[str],
    history: Optional[List[dict]],
) -> dict:
    """
    Do all retrieval, context building, and vision rendering.
    Returns a dict with:
      - error: str             early exit (no hits / no context)
      - vision_answer: str     complete answer from vision model (skip LLM call)
      - messages: list         LLM messages for text path (includes history)
      - all_hits: list
      - intent: str
      - chat_model: str | None
      - llm: OllamaClient
    """
    intent = _detect_intent(question)
    adaptive_k, adaptive_model = _adaptive_settings(question, intent, paper_id)
    queries = _expand_queries(question, intent)
    where   = {"paper_id": paper_id} if paper_id else None

    # Multi-query retrieval
    all_hits: List[Dict[str, Any]] = []
    seen_fp: set = set()
    for query in queries:
        try:
            hits = store.query(query, k=adaptive_k, where=where)
        except Exception:
            hits = []
        for h in hits:
            fp = h["text"][:120].strip()
            if fp not in seen_fp:
                seen_fp.add(fp)
                all_hits.append(h)

    all_hits.sort(key=lambda h: h.get("distance", 0.0))

    if not all_hits:
        return {"error": "I could not find relevant excerpts for this question.",
                "all_hits": [], "intent": intent, "confidence": "low", "n_relevant": 0}

    # Compute confidence from chunk relevance
    relevant = [h for h in all_hits if h.get("distance", 2.0) < _DISTANCE_THRESHOLD]
    n_relevant = len(relevant)
    avg_dist = (
        sum(h.get("distance", 0.0) for h in relevant[:5]) / min(5, len(relevant))
        if relevant else 2.0
    )
    if n_relevant >= 5 and avg_dist < 0.8:
        confidence = "high"
    elif n_relevant >= 2 and avg_dist < 1.3:
        confidence = "medium"
    else:
        confidence = "low"

    context = _build_context(all_hits)
    if not context.strip():
        context = _build_context(all_hits[:3], max_chars=14000, ignore_threshold=True)
    if not context.strip():
        return {
            "error": (
                "I could not find relevant text for this question in the indexed papers. "
                "Make sure the paper is ingested and try rephrasing."
            ),
            "all_hits": all_hits, "intent": intent, "confidence": "low", "n_relevant": 0,
        }

    llm = OllamaClient(env_path=env_path)

    # Vision path — always active in single-paper mode
    if paper_id and pdf_dir:
        pdf_path = Path(pdf_dir) / f"{paper_id}.pdf"
        if not pdf_path.exists():
            matches = sorted(Path(pdf_dir).glob(f"{paper_id}*.pdf"))
            pdf_path = matches[0] if matches else None

        if pdf_path and pdf_path.exists():
            is_review = _is_review_paper(paper_id)
            has_table = bool(__import__('re').search(r'\btable\b', question, __import__('re').IGNORECASE))
            max_pages = 12 if (is_review and has_table) else (10 if is_review else 6)
            retrieved_pages = [h["meta"]["page"] for h in all_hits[:10]]
            pages_to_render = get_pages_for_question(
                str(pdf_path), question, retrieved_pages, max_pages=max_pages
            )
            images = [
                img for p in pages_to_render
                if (img := render_page_as_base64(str(pdf_path), p)) is not None
            ]
            if images:
                try:
                    from .ce_db import query_sql

                    _count_kw = {"how many", "count", "number of", "dominant",
                                 "dominated", "quadrant", "cost-effective", "icer"}
                    need_db = is_review or any(k in question.lower() for k in _count_kw)
                    db_context = ""
                    if need_db:
                        paper_rows = query_sql(
                            "SELECT * FROM ce_comparisons WHERE paper_id = ?",
                            (paper_id,)
                        )
                        paper_quad: Dict[str, int] = {}
                        for r in paper_rows:
                            q = r.get("quadrant", "unclear")
                            paper_quad[q] = paper_quad.get(q, 0) + 1

                        all_rows = query_sql(
                            "SELECT paper_id, quadrant, ce_conclusion, body_region, "
                            "comparator_type FROM ce_comparisons "
                            "WHERE quadrant != 'unclear'", ()
                        )
                        all_quad: Dict[str, int] = {}
                        for r in all_rows:
                            q = r.get("quadrant", "unclear")
                            all_quad[q] = all_quad.get(q, 0) + 1

                        db_context = (
                            f"\n\n--- EXTRACTED DATABASE SUMMARY ---"
                            f"\nThis systematic review covers 78 individual papers."
                            f"\nClassified comparisons across all papers: {all_quad}. "
                            f"Total classified: {sum(all_quad.values())} "
                            f"(remaining are unclear/unclassified)."
                            f"\nUse these counts as the authoritative answer for any "
                            f"question about how many comparisons are dominant, dominated, "
                            f"NE, or SW across the studies in this review."
                        )

                    vision_answer = llm.vision_chat(
                        question=question,
                        images_b64=images,
                        context=context[:3000] + db_context,
                        think=False,
                        timeout=180,
                    )
                    return {
                        "vision_answer": vision_answer,
                        "all_hits": all_hits,
                        "intent": intent,
                        "chat_model": adaptive_model,
                        "confidence": confidence,
                        "n_relevant": n_relevant,
                        "llm": llm,
                    }
                except Exception as e:
                    print(f"[vision] failed, falling back to RAG: {e}")

    # Text path — build messages with conversation history
    messages: List[dict] = [{"role": "system", "content": _build_system_prompt(intent)}]
    # (confidence already computed above)
    if history:
        for h in history[-6:]:   # last 3 exchanges
            if h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})
    # For the systematic review, always inject per-figure-group DB breakdown
    db_supplement = ""
    if _is_review_paper(paper_id):
        try:
            from .ce_db import query_sql as _qsql
            fig_rows = _qsql(
                "SELECT figure_group, quadrant FROM ce_comparisons "
                "WHERE quadrant != 'unclear'", ()
            )
            fig_quad: Dict[str, Dict[str, int]] = {}
            for r in fig_rows:
                fg = r.get("figure_group", "unknown")
                q  = r.get("quadrant", "unclear")
                fig_quad.setdefault(fg, {})
                fig_quad[fg][q] = fig_quad[fg].get(q, 0) + 1

            db_supplement = (
                "\n\n--- DATABASE: QUADRANT BREAKDOWN BY FIGURE GROUP ---"
                "\nFig 4 = physiotherapy vs NON-physiotherapy (usual care, surgery, injection, GP)."
                "\nFig 5 = physiotherapy vs another PHYSIOTHERAPY modality."
                f"\nFig 4 quadrant counts: {fig_quad.get('Fig4', {})}."
                f"\nFig 5 quadrant counts: {fig_quad.get('Fig5', {})}."
                "\nUse these counts to explain differences between Figure 4 and Figure 5."
            )
        except Exception:
            pass

    messages.append({
        "role": "user",
        "content": f"Question:\n{question}\n\nEvidence excerpts:\n{context}{db_supplement}",
    })

    return {
        "vision_answer": None,
        "messages": messages,
        "all_hits": all_hits,
        "intent": intent,
        "chat_model": adaptive_model,
        "confidence": confidence,
        "n_relevant": n_relevant,
        "llm": llm,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    store: VectorStore,
    env_path: Optional[str] = None,
    k: int = 8,
    paper_id: Optional[str] = None,
    chat_model: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> dict:
    result = _build_rag_inputs(question, store, env_path, paper_id, pdf_dir, history)

    if "error" in result:
        return {"answer": result["error"], "sources": [], "intent": result["intent"],
                "confidence": "low", "n_relevant": 0}

    conf = {"confidence": result["confidence"], "n_relevant": result["n_relevant"]}

    if result.get("vision_answer"):
        return {
            "answer":  result["vision_answer"],
            "sources": _dedup_sources(result["all_hits"]),
            "intent":  result["intent"],
            "mode":    "vision",
            **conf,
        }

    llm = result["llm"]
    model = chat_model or result["chat_model"]
    answer = llm.chat(result["messages"], temperature=0.1, model=model)
    return {
        "answer":  answer,
        "sources": _dedup_sources(result["all_hits"]),
        "intent":  result["intent"],
        **conf,
    }


def answer_question_stream(
    question: str,
    store: VectorStore,
    env_path: Optional[str] = None,
    paper_id: Optional[str] = None,
    chat_model: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> Generator[str, None, None]:
    import json as _json
    result = _build_rag_inputs(question, store, env_path, paper_id, pdf_dir, history)

    # First chunk: confidence metadata (parsed by frontend)
    yield _json.dumps({
        "confidence": result.get("confidence", "low"),
        "n_relevant": result.get("n_relevant", 0),
    }) + "\n---\n"

    if "error" in result:
        yield result["error"]
        return

    if result.get("vision_answer"):
        yield result["vision_answer"]
        return

    llm = result["llm"]
    model = chat_model or result["chat_model"]
    yield from llm.chat_stream(result["messages"], temperature=0.1, model=model)
