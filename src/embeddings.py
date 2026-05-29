from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


@dataclass
class EmbeddingBackend:
    name: str
    method: str
    model: object

    def encode(self, texts: list[str]) -> np.ndarray:
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


def build_backend(model_name: str, corpus: list[str] | None = None) -> EmbeddingBackend:
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
