"""
Ingest all PDFs and DOCX files in data/pdfs/ into the Chroma vector store.

Usage (from backend/ directory):
    python ingest_all.py
"""

from pathlib import Path
from app.pdf_extract import extract_pdf_pages
from app.chunking import make_chunks
from app.vectorstore import VectorStore

ENV_PATH = "/Users/mahdie/Documents/1.PhysioAi/cost_effectiveness/.env"
PDF_DIR = Path("./data/pdfs")
CHROMA_DIR = Path("./data/chroma")


def main():
    store = VectorStore(persist_dir=str(CHROMA_DIR), env_path=ENV_PATH, collection_name="papers")

    supported = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.docx"))
    supported = sorted(supported)

    if not supported:
        print(f"No PDF/DOCX files found in {PDF_DIR.resolve()}")
        return

    print(f"Found {len(supported)} files. Starting ingestion…")

    for i, file_path in enumerate(supported, 1):
        print(f"[{i}/{len(supported)}] {file_path.name}")
        try:
            pages = extract_pdf_pages(str(file_path))
            chunks = make_chunks(pages, max_chars=2500, overlap=300)
            store.add_chunks(chunks, batch_size=16)
            print(f"  pages={len(pages)}  chunks={len(chunks)}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("Done. Vector DB persisted in:", CHROMA_DIR.resolve())


if __name__ == "__main__":
    main()
