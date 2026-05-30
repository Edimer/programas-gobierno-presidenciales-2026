from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import build_backend, load_dotenv, model_cache_key
from .io import chunk_text, extract_pdf_text, load_candidates
from .settings import ARTIFACT_DIR, CHUNK_OVERLAP, CHUNK_WORDS, DEFAULT_MODEL, ROOT, TOPIC_KEYWORDS


def _tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-záéíóúñ]+", value.lower()))


def infer_topic(text: str) -> str:
    lowered = text.lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword.lower() in lowered)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score else "Otros temas programáticos"


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
                    "topic": infer_topic(chunk),
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

    blocked_tokens = {
        "a", "al", "ante", "así", "asi", "como", "con", "contra", "cuál", "cual", "cuando", "cada", "de", "del",
        "desde", "donde", "e", "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese",
        "eso", "esta", "este", "estos", "fue", "ha", "han", "hacia", "hay", "hoy", "la", "las",
        "le", "lo", "los", "más", "mas", "mediante", "mi", "no", "o", "para", "pero", "por", "que", "se",
        "será", "sera", "ser", "si", "sin", "sobre", "son", "su", "sus", "también", "tambien", "un", "una", "unas",
        "uno", "unos", "y", "año", "años", "parte", "forma", "hacer", "debe", "deben", "serán",
        "seran", "tiene", "tienen", "ser", "estar", "desde", "hasta", "dentro", "frente", "bajo",
        "gran", "grandes", "mayor", "mejor", "nueva", "nuevo", "nuevas", "nuevos", "primer",
        "primera", "segundo", "segunda", "sido", "hemos", "vamos", "puede", "pueden", "deberá",
        "debera", "permitir", "garantizar", "fortalecer", "promover", "impulsar", "crear",
        "nos", "nosotros", "nosotras", "nuestro", "nuestra", "nuestros", "nuestras", "aquí", "aqui", "allí", "alli", "ahora",
        "este", "esta", "estos", "estas", "aquello", "aquella", "aquellas", "aquellos",
        "ivan", "iván", "cepeda", "abelardo", "espriella", "paloma", "valencia", "claudia",
        "lopez", "lópez", "sergio", "fajardo", "aida", "aída", "quilcue", "quilcué",
        "colombia", "colombiana", "colombiano", "colombianos", "gobierno", "país", "pais", "nacional",
        "política", "politica", "políticas", "politicas", "programa", "propuesta", "propuestas", "plan",
        "programas", "gobiernos", "personas", "ciudadanos", "ciudadanas", "sociedad", "sector",
        "sectores", "sistema", "sistemas", "modelo", "modelos", "desarrollo", "publico", "público",
        "publica", "pública", "publicas", "públicas", "través", "traves", "millones", "ciento",
        "cientos", "pesos", "relevante", "relevantes", "cifra", "cifras", "años", "anos",
        "2026", "2030", "2025", "2024", "castro", "uribe", "santos", "duque", "trump",
    }
    candidate_tokens = set()
    for candidate in load_candidates():
        candidate_tokens.update(_tokenize(candidate["name"]))
        candidate_tokens.update(_tokenize(candidate.get("ticket", "")))
        candidate_tokens.update(_tokenize(candidate.get("party", "")))
        candidate_tokens.update(_tokenize(candidate.get("position", "")))
    blocked_tokens.update(candidate_tokens)
    blocked_tokens.update({"gustavo", "petro", "pacto", "histórico", "historico", "derecha", "izquierda", "centro"})
    policy_tokens = {
        "seguridad", "policía", "policia", "justicia", "crimen", "delito", "extorsión", "extorsion",
        "narcotráfico", "narcotrafico", "paz", "conflicto", "víctimas", "victimas", "cárcel", "carcel",
        "salud", "hospital", "paciente", "atención", "atencion", "mental", "eps", "medicamentos",
        "educación", "educacion", "docente", "colegio", "universidad", "ciencia", "tecnología", "tecnologia",
        "empleo", "trabajo", "laboral", "productividad", "empresa", "inversión", "inversion",
        "industria", "turismo", "formalización", "formalizacion", "impuestos", "tributaria", "fiscal",
        "rural", "agrario", "campo", "tierra", "campesino", "alimentos", "regiones", "territorio",
        "energía", "energia", "petróleo", "petroleo", "gas", "minería", "mineria", "ambiente",
        "agua", "clima", "biodiversidad", "renovable", "transición", "transicion",
        "corrupción", "corrupcion", "transparencia", "contratación", "contratacion", "datos",
        "digital", "instituciones", "democracia", "descentralización", "descentralizacion",
        "vivienda", "mujeres", "jóvenes", "jovenes", "niñez", "ninez", "cuidado", "pobreza",
        "infraestructura", "movilidad", "transporte", "vías", "vias", "internet", "conectividad",
        "reforma", "regulación", "regulacion", "licencias", "financiación", "financiacion",
        "competitividad", "emprendimiento", "exportaciones", "agricultura", "ganadería", "ganaderia",
    }
    generic_phrases = {
        "bienestar social", "calidad vida", "derechos humanos", "desarrollo integral",
        "servicios públicos", "servicios publicos", "política pública", "politica publica",
    }

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        stop_words=list(blocked_tokens),
        ngram_range=(2, 4),
        token_pattern=r"(?u)\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]{4,}\b",
        min_df=2,
        max_df=0.75,
        max_features=10_000,
    )
    matrix = vectorizer.fit_transform(chunks["text"])
    terms = np.array(vectorizer.get_feature_names_out())

    def informative_phrase(term: str) -> bool:
        normalized = term.lower().strip()
        tokens = normalized.split()
        if len(tokens) < 2:
            return False
        if any(token in blocked_tokens for token in tokens):
            return False
        if any(char.isdigit() for char in normalized):
            return False
        if len(set(tokens)) == 1:
            return False
        if normalized in generic_phrases:
            return False
        if tokens[0] in {"mayor", "mejor", "gran", "nueva", "nuevo", "primera", "primer"}:
            return False
        if not any(token in policy_tokens for token in tokens):
            return False
        return True

    rows = []
    for candidate_id, group in chunks.groupby("candidate_id", sort=False):
        idx = group.index.to_numpy()
        other_idx = chunks.index[~chunks.index.isin(idx)].to_numpy()
        candidate_scores = np.asarray(matrix[idx].mean(axis=0)).ravel()
        other_scores = np.asarray(matrix[other_idx].mean(axis=0)).ravel()
        scores = candidate_scores - (0.65 * other_scores)
        candidate_rows = []
        for term, score in sorted(zip(terms, scores), key=lambda x: x[1], reverse=True):
            if score > 0 and informative_phrase(term):
                candidate_rows.append({"candidate_id": candidate_id, "candidate": group.iloc[0]["candidate"], "term": term, "score": float(score)})
            if len(candidate_rows) >= top_n * 4:
                break
        rows.extend(filter_concepts_with_llm(candidate_rows, top_n=top_n))
    return pd.DataFrame(rows)


def filter_concepts_with_llm(rows: list[dict], top_n: int = 12) -> list[dict]:
    if not rows or os.getenv("OPENAI_FILTER_CONCEPTS", "").lower() not in {"1", "true", "yes"}:
        return rows[:top_n]

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_OPENAI")
    if not api_key:
        return rows[:top_n]

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_CONCEPT_FILTER_MODEL", "gpt-4o-mini")
        concepts = [row["term"] for row in rows]
        prompt = (
            "Filtra esta lista de conceptos extraídos de un programa de gobierno colombiano. "
            "Devuelve solo conceptos programáticos informativos, no nombres propios, no partidos, "
            "no frases genéricas, no pronombres, no lugares/personas históricas, no ruido. "
            f"Elige máximo {top_n}. Responde únicamente JSON con la forma "
            '{"concepts":["concepto 1","concepto 2"]}.\n\n'
            + json.dumps(concepts, ensure_ascii=False)
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un editor riguroso de análisis político y política pública."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        selected = json.loads(content).get("concepts", [])
        selected_set = {item.strip().lower() for item in selected}
        filtered = [row for row in rows if row["term"].lower() in selected_set]
        return filtered[:top_n] or rows[:top_n]
    except Exception:
        return rows[:top_n]


def build_artifacts(model_name: str = DEFAULT_MODEL, max_pages: int | None = None) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    chunks, meta = build_corpus(max_pages=max_pages)
    backend = build_backend(model_name, corpus=chunks["text"].tolist())
    prefix = "" if backend.method == "openai" else "passage: "
    embeddings = backend.encode([f"{prefix}{text}" for text in chunks["text"].tolist()])
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
