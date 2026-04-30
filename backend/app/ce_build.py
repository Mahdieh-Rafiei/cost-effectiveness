"""
Build / rebuild the ce_comparisons table from all PDFs in pdf_dir.

Strategy per paper:
  1. Run 4 targeted semantic queries (CE results, intervention, outcomes, study design).
  2. Deduplicate and merge the top chunks into a context window.
  3. Call the LLM to extract all comparison rows (multi-comparison aware).
  4. Persist each row into ce_comparisons via upsert_comparison().
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any

from .vectorstore import VectorStore
from .ce_extract import extract_comparisons, make_fallback_comparisons, extract_quadrant_focused
from .ce_db import init_db, upsert_comparison, upsert_row, query_sql, get_conn

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
            time.sleep(5)  # avoid overwhelming the shared Ollama server

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


# ── Second-pass rebuild for unclear papers ────────────────────────────────────

def rebuild_unclear_papers(store: VectorStore, env_path: str) -> Dict[str, Any]:
    """
    Focused second-pass extraction for papers still marked unclear.
    Uses a shorter, targeted prompt that zeroes in on cost/effect direction.
    """
    unclear_rows = query_sql(
        "SELECT DISTINCT paper_id FROM ce_comparisons WHERE quadrant = 'unclear'", ()
    )
    paper_ids = [r["paper_id"] for r in unclear_rows]

    if not paper_ids:
        return {"status": "ok", "updated": 0, "still_unclear": 0,
                "message": "No unclear papers found"}

    updated = 0
    still_unclear = 0

    for paper_id in paper_ids:
        print(f"[REBUILD] {paper_id}")
        try:
            context = _gather_context(store, paper_id)
            if not context.strip():
                print(f"  ⚠ No context found")
                still_unclear += 1
                continue

            focused = extract_quadrant_focused(paper_id, context, env_path)
            quadrant = focused.get("quadrant", "unclear")

            if quadrant != "unclear":
                conn = get_conn()
                conn.execute("""
                    UPDATE ce_comparisons
                    SET quadrant=?, ce_conclusion=?, delta_cost_direction=?,
                        delta_effect_direction=?, icer=?, notes=?,
                        extraction_confidence=?
                    WHERE paper_id=? AND quadrant='unclear'
                """, (
                    quadrant,
                    focused.get("ce_conclusion", "inconclusive"),
                    focused.get("delta_cost_direction", "unknown"),
                    focused.get("delta_effect_direction", "unknown"),
                    focused.get("icer", "unknown"),
                    focused.get("notes", ""),
                    focused.get("extraction_confidence", "low"),
                    paper_id,
                ))
                conn.commit()
                conn.close()
                print(f"  ✓ Resolved to: {quadrant}")
                updated += 1
            else:
                print(f"  ⚠ Still unclear")
                still_unclear += 1

            time.sleep(3)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            still_unclear += 1

    return {
        "status": "ok",
        "papers_processed": len(paper_ids),
        "updated": updated,
        "still_unclear": still_unclear,
    }


# ── Legacy build_ce_table kept for backward compat ───────────────────────────

def build_ce_table(store: VectorStore, env_path: str, pdf_dir: str) -> Dict[str, Any]:
    """Alias for build_comparisons (used by old /build_ce_table endpoint)."""
    return build_comparisons(store=store, env_path=env_path, pdf_dir=pdf_dir)


# ── Enhanced context: paper + systematic review chunks ────────────────────────

def _gather_enhanced_context(store: VectorStore, paper_id: str) -> str:
    """
    Build extraction context using BOTH the paper's own chunks AND
    the systematic review's text about this paper (identified by author+year).
    This dramatically improves extraction for papers where the individual text
    chunks lacked explicit CE details.
    """
    base_context = _gather_context(store, paper_id)

    # Extract author and year from paper_id (format: "Author-YEAR-...")
    parts = paper_id.split("-", 2)
    author = parts[0].strip() if parts else ""
    year_m = re.search(r'\b(19|20)\d{2}\b', paper_id)
    year = year_m.group(0) if year_m else ""

    if not (author and year):
        return base_context

    # Find the systematic review paper
    sr_rows = query_sql(
        "SELECT DISTINCT paper_id FROM ce_comparisons "
        "WHERE lower(paper_id) LIKE '%systematic review%' "
        "   OR lower(paper_id) LIKE '%review of trial%'", ()
    )
    if not sr_rows:
        return base_context

    sr_id = sr_rows[0]["paper_id"]
    try:
        sr_hits = store.query(
            f"{author} {year} physiotherapy cost-effectiveness ICER "
            f"intervention comparator body region perspective time horizon",
            k=8, where={"paper_id": sr_id},
        )
        if sr_hits:
            sr_text = "\n\n".join(
                f"[Systematic Review p.{h['meta']['page']}] {h['text']}"
                for h in sr_hits
            )
            return (
                f"=== SYSTEMATIC REVIEW SUMMARY FOR THIS PAPER ===\n"
                f"{sr_text}\n\n"
                f"=== ORIGINAL PAPER TEXT ===\n"
                f"{base_context}"
            )
    except Exception:
        pass

    return base_context


# ── Re-extract papers flagged as incorrect/partial in validation ───────────────

_rebuild_failed_progress: Dict[str, Any] = {
    "running": False, "done": 0, "total": 0, "errors": 0
}


def rebuild_failed_papers(
    store: VectorStore,
    env_path: str,
    report_path: str,
) -> Dict[str, Any]:
    """
    Re-extract CE data for papers flagged as incorrect/partial/cannot_verify
    in the validation report. Uses enhanced context (paper + SR) to fix the
    'unknown' field problem.
    """
    global _rebuild_failed_progress
    import json as _json

    rp = Path(report_path)
    if not rp.exists():
        return {"error": "No validation report found. Run /batch_validate first."}

    report = _json.loads(rp.read_text())
    results = report.get("results", [])

    failing = [
        r for r in results
        if r.get("overall") in ("incorrect", "partial", "cannot_verify")
    ]
    paper_ids = [r["paper_id"] for r in failing]

    if not paper_ids:
        return {"status": "ok", "message": "No failing papers to re-extract.", "updated": 0}

    _rebuild_failed_progress = {
        "running": True, "done": 0, "total": len(paper_ids), "errors": 0
    }

    updated = 0
    errors_count = 0

    for paper_id in paper_ids:
        print(f"[REBUILD_FAILED] {paper_id}")
        try:
            context = _gather_enhanced_context(store, paper_id)
            if not context.strip():
                print(f"  ⚠ No context found")
                errors_count += 1
                _rebuild_failed_progress["done"] += 1
                _rebuild_failed_progress["errors"] += 1
                continue

            comparisons = extract_comparisons(
                paper_id=paper_id,
                context=context,
                env_path=env_path,
            )

            # Delete old rows for this paper then re-insert
            conn = get_conn()
            conn.execute("DELETE FROM ce_comparisons WHERE paper_id = ?", (paper_id,))
            conn.commit()
            conn.close()

            for comp in comparisons:
                upsert_comparison(comp)

            n = len(comparisons)
            print(f"  ✓ {n} comparison(s) re-extracted")
            updated += 1
            time.sleep(3)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            errors_count += 1
            _rebuild_failed_progress["errors"] += 1

        _rebuild_failed_progress["done"] += 1

    _rebuild_failed_progress["running"] = False

    return {
        "status": "ok",
        "total_failing": len(paper_ids),
        "re_extracted": updated,
        "errors": errors_count,
    }


def get_rebuild_failed_progress() -> Dict[str, Any]:
    return dict(_rebuild_failed_progress)


# ── Targeted Table 2 dose re-extraction ──────────────────────────────────────

# Queries specifically targeting intervention dose fields
_DOSE_QUERIES = [
    "sessions per week frequency visits appointments treatment schedule intensity",
    "session duration minutes length time per visit appointment",
    "total number of sessions visits encounters treatment programme",
    "weeks months duration intervention period treatment length programme",
    "supervised home-based group individual physiotherapy delivery setting",
    "exercise programme content manual therapy education intervention description",
    "comparator control usual care surgery injection medical treatment",
    "ICER incremental cost effectiveness ratio cost per QALY net monetary benefit",
]

_DOSE_PROMPT = """Paper ID: {paper_id}

Extract ONLY the following intervention delivery details from the evidence.
Look carefully in methods sections for specific numbers and frequencies.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "frequency": "<sessions per week e.g. 2x/week, 3 times/week — or unknown>",
  "total_sessions": "<total number e.g. 12, 18 — or unknown>",
  "session_length": "<minutes e.g. 60 min, 30-45 min — or unknown>",
  "duration_weeks": "<weeks e.g. 8, 12 — or unknown>",
  "supervision": "<supervised|home-based|group|mixed|unknown>",
  "intervention_type": "<exercise|manual_therapy|education|exercise+manual_therapy|exercise+education|mixed_physiotherapy|other|unknown>",
  "comparator_type": "<usual_care|medical_care|surgery|injection|wait_list|other_physiotherapy|education|other|unknown>",
  "icer": "<ICER value e.g. £23456/QALY, EUR 4984/QALY — or unknown>"
}}

Evidence:
{context}
"""

_rebuild_table2_progress: Dict[str, Any] = {
    "running": False, "done": 0, "total": 0, "updated": 0, "errors": 0
}


def _gather_dose_context(store: VectorStore, paper_id: str) -> str:
    """
    Gather context specifically targeting intervention dose fields.
    Uses 8 dose-focused queries against the paper + SR chunks about this paper.
    """
    from .ollama_client import OllamaClient as _OC  # avoid circular at module level

    seen_keys: set = set()
    hits_all: List[Dict[str, Any]] = []

    for query in _DOSE_QUERIES:
        try:
            hits = store.query(query, k=5, where={"paper_id": paper_id})
        except Exception:
            hits = []
        for h in hits:
            key = h["text"][:120]
            if key not in seen_keys:
                seen_keys.add(key)
                hits_all.append(h)

    # Build paper context
    parts: List[str] = []
    used = 0
    for h in hits_all:
        block = f"[p.{h['meta'].get('page','?')}] {h['text']}"
        if used + len(block) > 10000:
            break
        parts.append(block)
        used += len(block)
    paper_context = "\n\n".join(parts)

    # Add SR context for this paper
    pid_parts = paper_id.split("-", 2)
    author = pid_parts[0].strip() if pid_parts else ""
    year_m = re.search(r'\b(19|20)\d{2}\b', paper_id)
    year = year_m.group(0) if year_m else ""

    if author and year:
        sr_rows = query_sql(
            "SELECT DISTINCT paper_id FROM ce_comparisons "
            "WHERE lower(paper_id) LIKE '%systematic review%' "
            "   OR lower(paper_id) LIKE '%review of trial%'", ()
        )
        if sr_rows:
            sr_id = sr_rows[0]["paper_id"]
            try:
                sr_hits = store.query(
                    f"{author} {year} intervention sessions frequency duration supervised",
                    k=5, where={"paper_id": sr_id},
                )
                if sr_hits:
                    sr_text = "\n\n".join(
                        f"[SR p.{h['meta']['page']}] {h['text']}" for h in sr_hits
                    )
                    return (
                        f"=== SYSTEMATIC REVIEW summary for this paper ===\n"
                        f"{sr_text}\n\n"
                        f"=== PAPER TEXT ===\n{paper_context}"
                    )
            except Exception:
                pass

    return paper_context


def rebuild_table2_dose(store: VectorStore, env_path: str) -> Dict[str, Any]:
    """
    Targeted re-extraction of Table 2 dose fields:
    frequency, total_sessions, session_length, duration_weeks,
    supervision, intervention_type, comparator_type, icer.

    Only updates fields that are currently 'unknown' — never overwrites
    existing extracted values.
    """
    global _rebuild_table2_progress
    import json as _json

    from .ollama_client import OllamaClient

    TARGET_FIELDS = [
        "frequency", "total_sessions", "session_length", "duration_weeks",
        "supervision", "intervention_type", "comparator_type", "icer",
    ]
    UNKNOWN_VALS = {"unknown", "unclear", "", "none", "n/a"}

    # Find papers missing at least one dose field
    rows = query_sql("""
        SELECT DISTINCT paper_id FROM ce_comparisons
        WHERE (frequency   IN ('unknown','unclear','') OR frequency   IS NULL)
           OR (total_sessions IN ('unknown','unclear','') OR total_sessions IS NULL)
           OR (session_length  IN ('unknown','unclear','') OR session_length IS NULL)
           OR (duration_weeks  IN ('unknown','unclear','') OR duration_weeks IS NULL)
    """, ())

    paper_ids = [
        r["paper_id"] for r in rows
        if "systematic review" not in r["paper_id"].lower()
        and "review of trial" not in r["paper_id"].lower()
    ]

    _rebuild_table2_progress = {
        "running": True, "done": 0, "total": len(paper_ids),
        "updated": 0, "errors": 0,
    }

    llm = OllamaClient(env_path=env_path)
    updated = 0
    errors_count = 0

    for paper_id in paper_ids:
        print(f"[REBUILD_T2] {paper_id}")
        try:
            context = _gather_dose_context(store, paper_id)
            if not context.strip():
                print("  ⚠ No context")
                errors_count += 1
                _rebuild_table2_progress["errors"] += 1
                _rebuild_table2_progress["done"] += 1
                continue

            messages = [
                {"role": "system", "content":
                    "You are a health economics expert extracting intervention details. "
                    "Return ONLY valid JSON, no markdown, no explanation."},
                {"role": "user", "content":
                    _DOSE_PROMPT.format(paper_id=paper_id, context=context)},
            ]
            raw = llm.chat(messages, temperature=0.0, think=False, timeout=90)

            try:
                start = raw.find("{"); end = raw.rfind("}") + 1
                extracted = _json.loads(raw[start:end]) if start >= 0 else {}
            except Exception:
                extracted = {}

            if not extracted:
                print("  ⚠ No JSON extracted")
                errors_count += 1
                _rebuild_table2_progress["errors"] += 1
                _rebuild_table2_progress["done"] += 1
                continue

            # Get current DB values
            current_rows = query_sql(
                "SELECT * FROM ce_comparisons WHERE paper_id = ?", (paper_id,)
            )
            if not current_rows:
                _rebuild_table2_progress["done"] += 1
                continue

            # Build updates: only overwrite currently-unknown fields
            updates: Dict[str, str] = {}
            for field in TARGET_FIELDS:
                new_val = str(extracted.get(field, "")).strip()
                if not new_val or new_val.lower() in UNKNOWN_VALS:
                    continue
                # Apply to ALL comparison rows for this paper
                # (check each row individually)
                updates[field] = new_val

            if updates:
                conn = get_conn()
                for crow in current_rows:
                    row_updates = {
                        f: v for f, v in updates.items()
                        if str(crow.get(f, "")).lower().strip() in UNKNOWN_VALS
                    }
                    if row_updates:
                        set_clauses = ", ".join(f"{k} = ?" for k in row_updates)
                        conn.execute(
                            f"UPDATE ce_comparisons SET {set_clauses} "
                            f"WHERE comparison_id = ?",
                            list(row_updates.values()) + [crow["comparison_id"]],
                        )
                conn.commit()
                conn.close()
                print(f"  ✓ {list(updates.keys())}")
                updated += 1
            else:
                print("  ⚠ No new values found")

            time.sleep(2)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            errors_count += 1
            _rebuild_table2_progress["errors"] += 1

        _rebuild_table2_progress["done"] += 1
        _rebuild_table2_progress["updated"] = updated

    _rebuild_table2_progress["running"] = False
    return {
        "status": "ok",
        "total_processed": len(paper_ids),
        "updated": updated,
        "errors": errors_count,
    }


def get_rebuild_table2_progress() -> Dict[str, Any]:
    return dict(_rebuild_table2_progress)
