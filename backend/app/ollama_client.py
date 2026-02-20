import os
import requests
from dotenv import load_dotenv
from typing import Optional


def load_env(env_path: Optional[str] = None):
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

class OllamaClient:
    def __init__(self, env_path: Optional[str] = None):

        load_env(env_path)

        self.api_url = os.getenv("OLLAMA_API_URL", "https://dev.chat.cosy.bio/ollama").rstrip("/")
        self.api_key = os.getenv("COSYBIO_API_KEY", "")
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        if not self.api_key:
            raise RuntimeError("COSYBIO_API_KEY is missing.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    from typing import List
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Uses Ollama embeddings endpoint. Many servers support:
          POST /api/embeddings  with {model, prompt}
        or batch endpoints. We'll do per-text to be robust.
        """
        out = []
        for t in texts:
            payload = {"model": self.embed_model, "prompt": t}
            r = requests.post(f"{self.api_url}/api/embeddings", headers=self.headers, json=payload, timeout=120)
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False,
        }
        r = requests.post(f"{self.api_url}/api/chat", headers=self.headers, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
