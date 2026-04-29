import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional, List


def load_env(env_path: Optional[str] = None):
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


class OllamaClient:
    def __init__(self, env_path: Optional[str] = None):
        load_env(env_path)

        self.api_url = os.getenv("OLLAMA_API_URL", "https://dev.chat.cosy.bio/ollama").rstrip("/")
        self.api_key = os.getenv("COSYBIO_API_KEY", "").strip()

        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b").strip()
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest").strip()

        if not self.api_key:
            raise RuntimeError("COSYBIO_API_KEY is missing in environment (.env).")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []

        for t in texts:
            payload = {"model": self.embed_model, "prompt": t}  # ✅ your server requires 'prompt'
            r = requests.post(
                f"{self.api_url}/api/embeddings",
                headers=self.headers,
                json=payload,
                timeout=120,
            )

            if r.status_code >= 400:
                raise RuntimeError(f"Embeddings error {r.status_code}: {r.text}")

            j = r.json()

            # Common Ollama response format
            if "embedding" in j:
                out.append(j["embedding"])
                continue

            # Some servers wrap in data[]
            if "data" in j and isinstance(j["data"], list) and j["data"] and "embedding" in j["data"][0]:
                out.append(j["data"][0]["embedding"])
                continue

            raise RuntimeError(f"Unexpected embeddings response JSON: {j}")

        return out

    def chat(self, messages: list, temperature: float = 0.2,
             model: Optional[str] = None, think: bool = False,
             timeout: int = 180, max_retries: int = 4) -> str:
        payload = {
            "model": model or self.chat_model,
            "messages": messages,
            "options": {"temperature": temperature},
            "think": think,
            "stream": False,
        }

        for attempt in range(max_retries):
            wait = 20 * (attempt + 1)
            try:
                r = requests.post(
                    f"{self.api_url}/api/chat",
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                )
                if r.status_code == 503:
                    print(f"    [retry {attempt+1}/{max_retries}] server busy, waiting {wait}s…")
                    time.sleep(wait)
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"Chat error {r.status_code}: {r.text}")
                if not r.text.strip():
                    print(f"    [retry {attempt+1}/{max_retries}] empty response, waiting {wait}s…")
                    time.sleep(wait)
                    continue
                j = r.json()
                return j["message"]["content"]
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"    [retry {attempt+1}/{max_retries}] timeout, waiting {wait}s…")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Chat failed after {max_retries} retries")

    def vision_chat(self, question: str, images_b64: List[str],
                    context: str = "", think: bool = False,
                    timeout: int = 120) -> str:
        vision_model = os.getenv("OLLAMA_VISION_MODEL", "gemma4:latest")
        system = (
            "You are a medical research assistant analysing pages from a physiotherapy "
            "cost-effectiveness paper. The images show PDF pages that may contain: "
            "cost-effectiveness planes (scatter plots with 4 quadrants labelled dominant, "
            "dominated, NE, SW), tables, bar charts, or forest plots. "
            "Describe exactly what you see in the figures/tables on these pages. "
            "If a CE plane is shown, state which quadrant each intervention falls in and "
            "count dominant vs dominated comparisons. "
            "Never say a figure is 'not visible' — describe the page content you can see."
        )
        user_content = question
        if context:
            user_content = f"Context from paper text:\n{context}\n\nQuestion: {question}"
        payload = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content, "images": images_b64},
            ],
            "think": think,
            "stream": False,
        }
        r = requests.post(
            f"{self.api_url}/api/chat",
            headers=self.headers,
            json=payload,
            timeout=timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Vision chat error {r.status_code}: {r.text}")
        if not r.text.strip():
            raise RuntimeError("Empty response from vision model")
        return r.json()["message"]["content"]

    # Known embedding-only model name fragments to exclude from chat model list
    _EMBED_KEYWORDS = (
        "embed", "minilm", "bge", "e5-", "arctic-embed",
        "snowflake", "gte-", "instructor",
    )

    def list_models(self) -> List[str]:
        """Return chat-capable model names (embedding models excluded)."""
        try:
            r = requests.get(
                f"{self.api_url}/api/tags",
                headers=self.headers,
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            models = data.get("models", [])
            chat_models = [
                m["name"] for m in models
                if "name" in m
                and not any(kw in m["name"].lower() for kw in self._EMBED_KEYWORDS)
            ]
            return sorted(chat_models)
        except Exception:
            return []