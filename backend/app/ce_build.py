"""
Build / rebuild the ce_comparisons table from all PDFs in pdf_dir.

Strategy per paper:
  1. Run 4 targeted semantic queries (CE results, intervention, outcomes, study design).
  2. Deduplicate and merge the top chunks into a context window.
  3. Call the LLM to extract all comparison rows (multi-comparison aware).
  4. Persist each row into ce_comparisons via upsert_comparison().
"""

from pathlib import Path
from typing import List, Dict, Any

from .vectorstore import VectorStore
from .ce_extract import extract_comparisons, make_fallback_comparisons
from .ce_db import init_db, upsert_comparison, upsert_row

# ── Targeted semantic queries ────────────────────────────────────────────────
# Four specialised queries maximise recall of different evidence types.

CE_QUERY = (
    "incremental cost-effectiveness ratio ICER cost per QALY "
    "dominant dominated cost-effective willingness-to-pay threshold "
    "incremental cost incremental effect net monetary benefit"
)
INTERVENTION_QUERY = (
    "physiotherapy intervention exercise program manual therapy education "
    "sessions per week total sessions session length duration weeks "
    "supervised home-based group frequency intensity dose"
)
OUTCOME_QUERY = (
    "quality-adjusted life year QALY EQ-5D SF-6D SF-36 utility score "
    "health outcome primary outcome pain disability function VAS NRS WOMAC "
    "effectiveness clinical outcome improvement"
)
DESIGN_QUERY = (
    "randomized controlled trial RCT study design participants sample size "
    "country setting perspective societal healthcare follow-up time horizon "
    "comparator usual care surgery injection medical doctor"
)

QUERIES = [CE_QUERY, INTERVENTION_QUERY, OUTCOME_QUERY, DESIGN_QUERY]
K_PER_QUERY = 6          # chunks per query
MAX_CONTEXT_CHARS = 16000  # total context ceiling


def _gather_context(store: VectorStore, paper_id: str) -> str:
    """
    Run 4 targeted queries and build a deduplicated context string.
    Chunks from the CE query are prioritised (inserted first).
    """
    seen_ids: set = set()
    ordered_hits: List[Dict[str, Any]] = []

    for query in QUERIES:
        try:
            hits = store.query(
                question=query,
                k=K_PER_QUERY,
                where={"paper_id": paper_id},
            )
        except Exception:
            hits = []

        for h in hits:
            # Use text as dedup key (chunk_id not always in metadata)
            key = h["text"][:120]
            if key not in seen_ids:
                seen_ids.add(key)
                ordered_hits.append(h)

    # Build context string up to MAX_CONTEXT_CHARS
    parts: List[str] = []
    used = 0
    for h in ordered_hits:
        paper = h["meta"].get("paper_id", paper_id)
        page = h["meta"].get("page", "?")
        block = f"[{paper} p.{page}] {h['text']}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)


def build_comparisons(store: VectorStore, env_path: str, pdf_dir: str) -> Dict[str, Any]:
    """
    Main pipeline: for each PDF, extract all CE comparisons and persist to DB.
    Also populates the legacy ce_studies table for backward compatibility.
    """
    init_db()

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    total = len(pdfs)
    done = 0
    errors = 0

    for pdf in pdfs:
        paper_id = pdf.stem
        print(f"[BUILD] {paper_id}")

        try:
            context = _gather_context(store, paper_id)

            if not context.strip():
                print(f"  ⚠ No chunks found – using fallback row")
                comparisons = make_fallback_comparisons(paper_id, "No evidence chunks found in vector store")
            else:
                comparisons = extract_comparisons(
                    paper_id=paper_id,
                    context=context,
                    env_path=env_path,
                )

            for comp in comparisons:
                upsert_comparison(comp)

            # Backward-compat: mirror first comparison into legacy table
            first = comparisons[0]
            upsert_row({
                "paper_id": paper_id,
                "figure_group": first.get("figure_group", "unknown"),
                "condition": first.get("body_region", "unknown"),
                "time_horizon": first.get("time_horizon", "unknown"),
                "perspective": first.get("perspective", "unknown"),
                "outcome_type": first.get("outcome_type", "unknown"),
                "comparator_type": first.get("comparator_type", "unknown"),
                "quadrant": first.get("quadrant", "unclear"),
                "notes": first.get("notes", ""),
                "evidence_pages": first.get("evidence_pages", ""),
            })

            n = len(comparisons)
            print(f"  ✓ {n} comparison(s) extracted")
            done += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            errors += 1
            fallbacks = make_fallback_comparisons(paper_id, f"Pipeline error: {e}")
            for comp in fallbacks:
                try:
                    upsert_comparison(comp)
                except Exception:
                    pass

    return {
        "status": "ok",
        "papers_processed": done,
        "papers_total": total,
        "errors": errors,
    }


# ── Legacy build_ce_table kept for backward compat ───────────────────────────

def build_ce_table(store: VectorStore, env_path: str, pdf_dir: str) -> Dict[str, Any]:
    """Alias for build_comparisons (used by old /build_ce_table endpoint)."""
    return build_comparisons(store=store, env_path=env_path, pdf_dir=pdf_dir)
