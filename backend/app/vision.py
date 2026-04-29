"""
Vision helpers: render PDF pages as base64 PNG for multimodal LLM queries.
"""

import base64
import re
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF — already a dependency


def render_page_as_base64(pdf_path: str, page_num: int, dpi: int = 250) -> Optional[str]:
    """
    Render a single PDF page (1-indexed) as base64 PNG.
    Returns None if the page doesn't exist or rendering fails.
    """
    try:
        doc = fitz.open(pdf_path)
        idx = page_num - 1  # convert to 0-indexed
        if idx < 0 or idx >= doc.page_count:
            doc.close()
            return None
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[idx].get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode()
    except Exception:
        return None


def find_figure_pages(pdf_path: str, figure_num: str) -> List[int]:
    """
    Search all pages for a figure number (e.g. '4', '4a').
    Matches 'Fig. 4', 'Figure 4', 'fig 4' etc.
    Returns a list of 1-indexed page numbers.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        pattern = re.compile(
            rf'\bfig(?:ure)?\.?\s*{re.escape(figure_num)}\b',
            re.IGNORECASE,
        )
        for i in range(doc.page_count):
            text = doc[i].get_text("text")
            if pattern.search(text):
                pages.append(i + 1)
        doc.close()
        return pages
    except Exception:
        return []


def extract_figure_ref(question: str) -> Optional[str]:
    """Extract the FIRST figure number from the question."""
    m = re.search(r'\bfig(?:ure)?\.?\s*(\d+[a-z]?)\b', question, re.IGNORECASE)
    return m.group(1) if m else None


def extract_all_figure_refs(question: str) -> List[str]:
    """Extract ALL figure numbers from the question (for comparative questions)."""
    return re.findall(r'\bfig(?:ure)?\.?\s*(\d+[a-z]?)\b', question, re.IGNORECASE)


def is_figure_question(question: str) -> bool:
    keywords = {"figure", "fig", "plot", "chart", "graph", "table", "image",
                "illustration", "show", "depict", "display"}
    q = question.lower()
    return any(k in q for k in keywords)


def get_pages_for_question(
    pdf_path: str,
    question: str,
    retrieved_pages: List[int],
    max_pages: int = 6,
) -> List[int]:
    """
    Return the best set of page numbers to render for a figure question.
    Combines figure-specific search + top retrieved pages, capped at max_pages.
    """
    pages: List[int] = []

    # Extract all referenced figure numbers (handles "Fig 4 vs Fig 5" comparisons)
    fig_nums = extract_all_figure_refs(question)
    for fig_num in fig_nums:
        for p in find_figure_pages(pdf_path, fig_num):
            if p not in pages:
                pages.append(p)

    # Add surrounding page for each figure page (caption often on next page)
    extra = []
    for p in pages:
        if p + 1 not in pages:
            extra.append(p + 1)
    pages.extend(extra)

    # Fill remaining slots with top retrieved pages
    for p in retrieved_pages:
        if p not in pages:
            pages.append(p)
        if len(pages) >= max_pages:
            break

    return pages[:max_pages]
