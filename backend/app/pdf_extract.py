from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF

@dataclass
class PageText:
    paper_id: str
    pdf_path: str
    page: int
    text: str

def extract_pdf_pages(pdf_path: str) -> list[PageText]:
    p = Path(pdf_path)
    paper_id = p.stem

    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = " ".join(text.split())  # normalize whitespace
        if text.strip():
            pages.append(PageText(
                paper_id=paper_id,
                pdf_path=str(p),
                page=i + 1,  # 1-indexed for humans
                text=text
            ))
    doc.close()
    return pages
