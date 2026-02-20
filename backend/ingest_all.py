from pathlib import Path
from app.pdf_extract import extract_pdf_pages
from app.chunking import make_chunks
from app.vectorstore import VectorStore

ENV_PATH = "/Users/mahdie/Documents/1.PhysioAi/cost_effectiveness/.env"
PDF_DIR = Path("./data/pdfs")
CHROMA_DIR = Path("./data/chroma")

def main():
    store = VectorStore(persist_dir=str(CHROMA_DIR), env_path=ENV_PATH, collection_name="papers")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR.resolve()}")
        return

    print(f"Found {len(pdfs)} PDFs. Starting ingestion...")

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Ingesting: {pdf_path.name}")
        pages = extract_pdf_pages(str(pdf_path))
        chunks = make_chunks(pages, max_chars=2500, overlap=300)
        store.add_chunks(chunks, batch_size=16)
        print(f"  pages={len(pages)} chunks={len(chunks)}")

    print("Done. Vector DB persisted in:", CHROMA_DIR.resolve())

if __name__ == "__main__":
    main()
