from typing import Optional, List, Dict, Any

from .vectorstore import VectorStore
from .ollama_client import OllamaClient

SYSTEM_PROMPT = """You are an evidence-grounded research assistant.
You must answer ONLY using the provided excerpts.
If the evidence is insufficient, say you don't have enough evidence.
Always cite sources in the format: (PaperID p.X).

For study-specific questions, extract concrete facts if available:
- number of participants in final analysis
- intervention content
- intervention duration
- intervention intensity (frequency/session length/intensity level)
- intervention type (exercise/manual therapy/education/etc.)
- which intervention was more cost-effective and why

If a requested item is not explicitly in the excerpts, say: "Not reported in the retrieved excerpts."
"""

def build_context(hits: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    parts = []
    used = 0
    for h in hits:
        paper = h["meta"]["paper_id"]
        page = h["meta"]["page"]
        snippet = h["text"]
        block = f"[{paper} p.{page}] {snippet}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)

def answer_question(
    question: str,
    store: VectorStore,
    env_path: Optional[str] = None,
    k: int = 8,
    paper_id: Optional[str] = None,   # ✅ added
) -> dict:
    # ✅ apply paper filter first
    where = {"paper_id": paper_id} if paper_id else None
    hits = store.query(question, k=k, where=where)

    context = build_context(hits)

    # If no hits, return safely
    if not hits:
        return {
            "answer": "I could not find relevant excerpts for this question.",
            "sources": []
        }

    llm = OllamaClient(env_path=env_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nEvidence excerpts:\n{context}"
        }
    ]
    answer = llm.chat(messages, temperature=0.2)

    return {
        "answer": answer,
        "sources": [
            {"paper_id": h["meta"]["paper_id"], "page": h["meta"]["page"]}
            for h in hits
        ]
    }
