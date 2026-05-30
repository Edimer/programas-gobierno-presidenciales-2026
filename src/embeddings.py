from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import ROOT


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


@dataclass
class EmbeddingBackend:
    name: str
    method: str
    model: object

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.method == "openai":
            safe_texts = [text.replace("\n", " ").strip() for text in texts]
            vectors = []
            for start in range(0, len(safe_texts), 96):
                response = self.model.embeddings.create(model=self.name, input=safe_texts[start : start + 96])
                vectors.extend(item.embedding for item in response.data)
            vectors = np.asarray(vectors, dtype=np.float32)
            return l2_normalize(vectors)
        if self.method == "sentence-transformers":
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return np.asarray(vectors, dtype=np.float32)
        vectors = self.model.transform(texts).toarray()
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


def model_cache_key(model_name: str) -> str:
    return hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:10]


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_backend(model_name: str, corpus: list[str] | None = None) -> EmbeddingBackend:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_OPENAI")
    if model_name.startswith("text-embedding-") or os.getenv("OPENAI_EMBEDDING_MODEL"):
        try:
            from openai import OpenAI

            resolved_model = os.getenv("OPENAI_EMBEDDING_MODEL", model_name)
            return EmbeddingBackend(resolved_model, "openai", OpenAI(api_key=api_key))
        except Exception:
            if model_name.startswith("text-embedding-"):
                raise

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return EmbeddingBackend(model_name, "sentence-transformers", model)
    except Exception:
        if corpus is None:
            raise
        from sklearn.feature_extraction.text import TfidfVectorizer

        model = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            max_features=12_000,
        )
        model.fit(corpus)
        return EmbeddingBackend("TF-IDF fallback", "tfidf", model)
