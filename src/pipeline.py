from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import build_backend, model_cache_key
from .io import chunk_text, extract_pdf_text, load_candidates
from .settings import ARTIFACT_DIR, CHUNK_OVERLAP, CHUNK_WORDS, DEFAULT_MODEL, ROOT


def build_corpus(max_pages: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta_rows = []
    for candidate in load_candidates():
        text, meta = extract_pdf_text(candidate["pdf"], max_pages=max_pages)
        chunks = chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
        meta_rows.append({**candidate, **meta, "chunks": len(chunks)})
        for index, chunk in enumerate(chunks):
            rows.append(
                {
                    "candidate_id": candidate["id"],
                    "candidate": candidate["name"],
                    "ticket": candidate["ticket"],
                    "party": candidate["party"],
                    "position": candidate["position"],
                    "color": candidate["color"],
                    "chunk_id": f"{candidate['id']}-{index:04d}",
                    "chunk_index": index,
                    "text": chunk,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def candidate_centroids(chunks: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    centroids = []
    rows = []
    for candidate_id, group in chunks.groupby("candidate_id", sort=False):
        idx = group.index.to_numpy()
        centroid = embeddings[idx].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        centroids.append(centroid)
        first = group.iloc[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate": first["candidate"],
                "party": first["party"],
                "position": first["position"],
                "color": first["color"],
                "chunks": len(group),
            }
        )
    return pd.DataFrame(rows), np.vstack(centroids)


def enrich_projection(chunks: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    try:
        import hdbscan
        import umap

        reducer = umap.UMAP(n_neighbors=18, min_dist=0.08, metric="cosine", random_state=42)
        xy = reducer.fit_transform(embeddings)
        min_cluster_size = max(8, min(30, len(chunks) // 18))
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(xy)
    except Exception:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        xy = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        labels = KMeans(n_clusters=min(8, max(2, len(chunks) // 40)), random_state=42, n_init="auto").fit_predict(embeddings)
    projected = chunks.copy()
    projected["x"] = xy[:, 0]
    projected["y"] = xy[:, 1]
    projected["cluster"] = labels.astype(str)
    return projected


def top_terms(chunks: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    stop_words = {
        "a", "al", "ante", "asi", "como", "con", "contra", "cual", "cuando", "de", "del", "desde",
        "donde", "e", "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese", "esta", "este",
        "estos", "fue", "ha", "hacia", "hay", "la", "las", "le", "lo", "los", "mas", "mediante",
        "mi", "no", "o", "para", "pero", "por", "que", "se", "sera", "ser", "si", "sin", "sobre",
        "son", "su", "sus", "tambien", "un", "una", "unas", "uno", "unos", "y",
        "colombia", "gobierno", "pais", "nacional", "politica", "politicas", "programa", "propuesta",
    }
    rows = []
    for candidate_id, group in chunks.groupby("candidate_id", sort=False):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=list(stop_words),
            ngram_range=(1, 2),
            max_features=2500,
        )
        matrix = vectorizer.fit_transform(group["text"])
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        for term, score in sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_n]:
            rows.append({"candidate_id": candidate_id, "candidate": group.iloc[0]["candidate"], "term": term, "score": float(score)})
    return pd.DataFrame(rows)


def build_artifacts(model_name: str = DEFAULT_MODEL, max_pages: int | None = None) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    chunks, meta = build_corpus(max_pages=max_pages)
    backend = build_backend(model_name, corpus=chunks["text"].tolist())
    embeddings = backend.encode([f"passage: {text}" for text in chunks["text"].tolist()])
    candidates, centroids = candidate_centroids(chunks, embeddings)
    projected = enrich_projection(chunks, embeddings)
    terms = top_terms(chunks)

    suffix = model_cache_key(backend.name)
    chunks_path = ARTIFACT_DIR / f"chunks_{suffix}.parquet"
    projected_path = ARTIFACT_DIR / f"projection_{suffix}.parquet"
    terms_path = ARTIFACT_DIR / f"terms_{suffix}.parquet"
    embeddings_path = ARTIFACT_DIR / f"embeddings_{suffix}.npy"
    centroids_path = ARTIFACT_DIR / f"centroids_{suffix}.npy"
    candidates_path = ARTIFACT_DIR / f"candidate_meta_{suffix}.parquet"
    diagnostics_path = ARTIFACT_DIR / f"diagnostics_{suffix}.json"

    chunks.to_parquet(chunks_path, index=False)
    projected.to_parquet(projected_path, index=False)
    terms.to_parquet(terms_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    np.save(embeddings_path, embeddings)
    np.save(centroids_path, centroids)
    diagnostics = {
        "backend": backend.name,
        "method": backend.method,
        "chunks": int(len(chunks)),
        "candidates": int(len(candidates)),
        "source_pdfs": meta.to_dict(orient="records"),
        "files": {
            "chunks": str(chunks_path.relative_to(ROOT)),
            "projection": str(projected_path.relative_to(ROOT)),
            "terms": str(terms_path.relative_to(ROOT)),
            "embeddings": str(embeddings_path.relative_to(ROOT)),
            "centroids": str(centroids_path.relative_to(ROOT)),
            "candidate_meta": str(candidates_path.relative_to(ROOT)),
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics


def latest_diagnostics() -> dict | None:
    files = sorted(ARTIFACT_DIR.glob("diagnostics_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))
