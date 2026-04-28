"""
Extract page-level text from PDF files (PyMuPDF) and DOCX files (python-docx).
Returns a list of PageText objects for downstream chunking.
"""

from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF


@dataclass
class PageText:
    paper_id: str
    pdf_path: str
    page: int
    text: str


def _extract_pdf(p: Path) -> list[PageText]:
    doc = fitz.open(str(p))
    pages: list[PageText] = []
    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text") or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append(PageText(
                paper_id=p.stem,
                pdf_path=str(p),
                page=i + 1,
                text=text,
            ))
    doc.close()
    return pages


def _extract_docx(p: Path) -> list[PageText]:
    try:
        from docx import Document  # type: ignore
    except ImportError:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")

    doc = Document(str(p))
    # Group paragraphs into pseudo-pages of ~3000 chars
    all_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    chunk_size = 3000
    pages: list[PageText] = []
    for i, start in enumerate(range(0, max(len(all_text), 1), chunk_size), start=1):
        chunk = " ".join(all_text[start:start + chunk_size].split())
        if chunk.strip():
            pages.append(PageText(
                paper_id=p.stem,
                pdf_path=str(p),
                page=i,
                text=chunk,
            ))
    return pages


def extract_pdf_pages(pdf_path: str) -> list[PageText]:
    """Extract text from a PDF or DOCX file."""
    p = Path(pdf_path)
    if p.suffix.lower() == ".docx":
        return _extract_docx(p)
    return _extract_pdf(p)
