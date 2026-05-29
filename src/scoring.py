from __future__ import annotations

import numpy as np
import pandas as pd

from .embeddings import build_backend
from .settings import DEFAULT_MODEL, TOP_K_CHUNKS


def cosine_scores(query_vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return query_vectors @ matrix.T


def candidate_similarity(
    answers: list[dict],
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_meta: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = [item for item in answers if item["answer"].strip()]
    if not valid:
        empty = candidate_meta[["candidate_id", "candidate", "color"]].copy()
        empty["score"] = 0.0
        empty["confidence"] = 0.0
        return empty, pd.DataFrame()

    backend = build_backend(model_name or DEFAULT_MODEL, corpus=chunks["text"].tolist())
    query_texts = [f"query: {item['topic']}. {item['answer']}" for item in valid]
    query_vectors = backend.encode(query_texts)
    sims = cosine_scores(query_vectors, embeddings)

    detail_rows = []
    score_rows = []
    for candidate_id, group in chunks.groupby("candidate_id", sort=False):
        idx = group.index.to_numpy()
        candidate_sims = sims[:, idx]
        per_question = []
        for question_i, item in enumerate(valid):
            ranked = np.argsort(candidate_sims[question_i])[::-1][:TOP_K_CHUNKS]
            top_scores = candidate_sims[question_i][ranked]
            score = float(np.mean(top_scores[: min(5, len(top_scores))]))
            per_question.append(score * item["weight"])
            best_local = ranked[0]
            best_chunk = group.iloc[int(best_local)]
            detail_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": best_chunk["candidate"],
                    "topic": item["topic"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "similarity": score,
                    "evidence": best_chunk["text"][:520],
                }
            )
        raw = float(np.sum(per_question) / max(sum(item["weight"] for item in valid), 1e-9))
        score_rows.append({"candidate_id": candidate_id, "raw_score": raw})

    scores = pd.DataFrame(score_rows)
    if scores["raw_score"].max() > scores["raw_score"].min():
        scores["score"] = 100 * (scores["raw_score"] - scores["raw_score"].min()) / (scores["raw_score"].max() - scores["raw_score"].min())
    else:
        scores["score"] = 0.0
    scores["confidence"] = scores["score"] / max(float(scores["score"].sum()), 1e-9)
    scores = candidate_meta.merge(scores, on="candidate_id").sort_values("score", ascending=False)
    return scores, pd.DataFrame(detail_rows)
