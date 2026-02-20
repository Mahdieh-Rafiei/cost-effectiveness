from dataclasses import dataclass
from typing import Iterable
import re

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    pdf_path: str
    page: int
    text: str

def split_into_chunks(text: str, max_chars: int = 2500, overlap: int = 300) -> list[str]:
    # simple + robust character chunker
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def make_chunks(pages: Iterable, max_chars: int = 2500, overlap: int = 300) -> list[Chunk]:
    out: list[Chunk] = []
    for pg in pages:
        parts = split_into_chunks(pg.text, max_chars=max_chars, overlap=overlap)
        for j, part in enumerate(parts):
            chunk_id = f"{pg.paper_id}_p{pg.page}_c{j}"
            out.append(Chunk(
                chunk_id=chunk_id,
                paper_id=pg.paper_id,
                pdf_path=pg.pdf_path,
                page=pg.page,
                text=part
            ))
    return out
