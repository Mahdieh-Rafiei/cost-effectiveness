from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings

from .ollama_client import OllamaClient
from .chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str, env_path: Optional[str] = None, collection_name: str = "papers"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(name=collection_name)
        self.ollama = OllamaClient(env_path=env_path)

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 32):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.ollama.embed(texts)

            self.col.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "paper_id": c.paper_id,
                        "pdf_path": c.pdf_path,
                        "page": c.page
                    }
                    for c in batch
                ]
            )

    def query(self, question: str, k: int = 8, where: Optional[Dict[str, Any]] = None):
        q_emb = self.ollama.embed([question])[0]

        # Retry with decreasing k if Chroma/HNSW fails
        trial_ks = [k, min(k, 20), min(k, 10), min(k, 5), 3, 1]
        seen = set()
        trial_ks = [x for x in trial_ks if x > 0 and not (x in seen or seen.add(x))]

        last_err = None
        for kk in trial_ks:
            try:
                res = self.col.query(
                    query_embeddings=[q_emb],
                    n_results=kk,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )

                docs = res["documents"][0] if res.get("documents") else []
                metas = res["metadatas"][0] if res.get("metadatas") else []
                dists = res["distances"][0] if res.get("distances") else []

                hits = []
                for doc, meta, dist in zip(docs, metas, dists):
                    hits.append({"text": doc, "meta": meta, "distance": dist})
                return hits

            except Exception as e:
                last_err = e
                continue

        raise last_err
