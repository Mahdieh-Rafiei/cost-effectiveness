from pathlib import Path
from typing import List
from .vectorstore import VectorStore
from .ce_extract import extract_ce_fields
from .ce_db import init_db, upsert_row

KEYWORD_QUERY = (
    "cost-effectiveness perspective time horizon follow-up "
    "incremental cost incremental effect ICER QALY "
    "dominant dominated cost-effective comparator usual care"
)

def build_context(hits: List[dict], max_chars: int = 14000) -> str:
    parts = []
    used = 0
    pages_used = []
    for h in hits:
        paper = h["meta"]["paper_id"]
        page = h["meta"]["page"]
        txt = h["text"]
        block = f"[{paper} p.{page}] {txt}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
        pages_used.append(page)
    return "\n\n".join(parts)

def build_ce_table(store: VectorStore, env_path: str, pdf_dir: str) -> dict:
    init_db()

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    total = len(pdfs)
    done = 0

    for pdf in pdfs:
        paper_id = pdf.stem
        try:
            hits = store.query(
                question=KEYWORD_QUERY,
                k=10,
                where={"paper_id": paper_id}
            )
            context = build_context(hits) if hits else ""

            if not context.strip():
                data = {
                    "paper_id": paper_id,
                    "figure_group": "unknown",
                    "condition": "unknown",
                    "time_horizon": "unknown",
                    "perspective": "unknown",
                    "outcome_type": "unknown",
                    "comparator_type": "unknown",
                    "quadrant": "unclear",
                    "notes": "No evidence chunks found",
                    "evidence_pages": "unknown",
                }
            else:
                data = extract_ce_fields(paper_id=paper_id, context=context, env_path=env_path)

            upsert_row(data)
            done += 1
            print(f"[OK] {paper_id}")

        except Exception as e:
            print(f"[ERR] {paper_id}: {e}")
            upsert_row({
                "paper_id": paper_id,
                "figure_group": "unknown",
                "condition": "unknown",
                "time_horizon": "unknown",
                "perspective": "unknown",
                "outcome_type": "unknown",
                "comparator_type": "unknown",
                "quadrant": "unclear",
                "notes": "Pipeline error: " + str(e),
                "evidence_pages": "unknown",
            })


    return {"status": "ok", "papers_processed": done, "papers_total": total}
