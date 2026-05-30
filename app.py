from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.io import load_candidates, load_questions
from src.pipeline import build_artifacts, infer_topic, latest_diagnostics, top_terms
from src.scoring import candidate_similarity
from src.settings import DEFAULT_MODEL, ROOT
from src.visuals import projection_scatter, score_bars, similarity_heatmap, terms_chart


st.set_page_config(page_title="Programas presidenciales Colombia 2026", page_icon="🗳️", layout="wide")
ARTIFACT_VIEW_VERSION = "conceptos-v4-diagonal-sin-texto"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: 0 !important;}
    div[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #D7DEE8; border-radius: 8px; padding: 14px;}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #0F172A !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px; padding: 8px 12px;}
    .source-link {font-size: 0.92rem; line-height: 1.35;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts(view_version: str):
    diagnostics = latest_diagnostics()
    if diagnostics is None:
        diagnostics = build_artifacts(DEFAULT_MODEL)
    files = diagnostics["files"]
    paths = {key: ROOT / Path(value) for key, value in files.items()}
    chunks = pd.read_parquet(paths["chunks"])
    projected = pd.read_parquet(paths["projection"])
    terms = pd.read_parquet(paths["terms"])
    candidate_meta = pd.read_parquet(paths["candidate_meta"])
    embeddings = np.load(paths["embeddings"])
    centroids = np.load(paths["centroids"])
    canonical_candidates = pd.DataFrame(load_candidates())
    canonical_candidates = canonical_candidates.rename(columns={"id": "candidate_id", "name": "candidate"})
    canonical_columns = ["candidate_id", "candidate", "ticket", "party", "position", "color"]
    canonical = canonical_candidates[canonical_columns]

    def refresh_candidate_labels(frame: pd.DataFrame) -> pd.DataFrame:
        if "candidate_id" in frame.columns:
            keep_columns = [column for column in frame.columns if column not in canonical_columns or column == "candidate_id"]
            return frame[keep_columns].merge(canonical, on="candidate_id", how="left")
        return frame

    chunks = refresh_candidate_labels(chunks)
    projected = refresh_candidate_labels(projected)
    terms = refresh_candidate_labels(terms)
    candidate_meta = refresh_candidate_labels(candidate_meta)
    if "topic" not in chunks.columns:
        chunks["topic"] = chunks["text"].map(infer_topic)
    if "topic" not in projected.columns:
        projected = projected.merge(chunks[["chunk_id", "topic"]], on="chunk_id", how="left")
    terms = top_terms(chunks)
    return diagnostics, chunks, projected, terms, candidate_meta, embeddings, centroids


def topic_summary(projected: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    counts = (
        projected.groupby(["candidate", "topic"], as_index=False)
        .size()
        .rename(columns={"size": "fragmentos"})
    )
    counts["porcentaje"] = 100 * counts["fragmentos"] / counts.groupby("candidate")["fragmentos"].transform("sum").clip(lower=1)
    rows = []
    for candidate, group in counts.sort_values(["candidate", "porcentaje"], ascending=[True, False]).groupby("candidate", sort=False):
        top_topics = group.head(top_n)
        rows.append(
            {
                "Candidatura": candidate,
                "Temas más presentes": " · ".join(f"{row.topic} ({row.porcentaje:.1f}%)" for row in top_topics.itertuples()),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("Fuentes")
    st.markdown(
        """
        <div class="source-link">
        <p><a href="https://www.movimientopactohistorico.co/programa-gobierno" target="_blank">Programa de Gobierno de Iván Cepeda</a></p>
        <p><a href="https://defensoresdelapatria.com/wpcontent/uploads/2026/04/PROPUESTAS-DEL-TIGRE.pdf" target="_blank">Programa de Gobierno de Abelardo de la Espriella</a></p>
        <p><a href="https://palomavalencia.com/images/documentos/Plan%20Integrado%20de%20Gobierno%20Final_compressed.pdf" target="_blank">Programa de Gobierno de Paloma Valencia</a></p>
        <p><a href="https://claudia-lopez.com/campana/" target="_blank">Programa de Gobierno de Claudia López</a></p>
        <p><a href="https://drive.google.com/file/d/1b0JOU1qalqmih9YdYbuWIDWQEAAvYWn7/view" target="_blank">Programa de Gobierno de Sergio Fajardo</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

diagnostics, chunks, projected, terms, candidate_meta, embeddings, _centroids = load_artifacts(ARTIFACT_VIEW_VERSION)
source_rows = pd.DataFrame(diagnostics["source_pdfs"])
canonical_names = pd.DataFrame(load_candidates()).set_index("id")["name"].to_dict()
if "id" in source_rows:
    source_rows["name"] = source_rows["id"].map(canonical_names).fillna(source_rows["name"])
pages_read = int(source_rows["pages_read"].sum()) if "pages_read" in source_rows else 0

st.title("Programas presidenciales Colombia 2026")
st.caption("Explorador semántico de propuestas: similitud, agrupación temática y afinidad por respuestas ciudadanas.")

metric_cols = st.columns(4)
metric_cols[0].metric("Programas revisados", f"{len(candidate_meta)}")
metric_cols[1].metric("Páginas leídas", f"{pages_read:,}")
metric_cols[2].metric("Temas comparados", f"{projected['topic'].nunique()}")
metric_cols[3].metric("Preguntas ciudadanas", f"{len(load_questions())}")

tabs = st.tabs(["Mapa semántico", "Comparador", "Afinidad ciudadana", "Metodología"])

with tabs[0]:
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Territorio semántico de los programas")
        st.plotly_chart(projection_scatter(projected), width="stretch")
    with right:
        st.subheader("Similitud relativa entre candidaturas")
        st.plotly_chart(similarity_heatmap(candidate_meta, chunks, embeddings), width="stretch")
        st.caption("Cada punto del mapa es un fragmento del programa. La distancia es aproximada y sirve para explorar, no para ordenar ideológicamente.")
    st.subheader("Lectura temática por candidatura")
    st.caption("Resumen normalizado dentro de cada programa. No compara volumen de páginas ni cantidad absoluta de fragmentos.")
    st.dataframe(topic_summary(projected), hide_index=True, width="stretch")

with tabs[1]:
    candidate = st.selectbox("Candidatura", candidate_meta["candidate"].tolist())
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Conceptos distintivos")
        st.plotly_chart(terms_chart(terms, candidate), width="stretch")
    with col_b:
        st.subheader("Fragmentos representativos")
        topic_options = ["Todos los temas"] + sorted(projected["topic"].dropna().unique().tolist())
        selected_topic = st.selectbox("Tema", topic_options)
        subset = projected[projected["candidate"] == candidate]
        if selected_topic != "Todos los temas":
            subset = subset[subset["topic"] == selected_topic]
        if subset.empty:
            st.info("No hay fragmentos clasificados en este tema para esta candidatura.")
        else:
            idx = subset.index.to_numpy()
            local_vectors = embeddings[idx]
            centroid = local_vectors.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
            subset = subset.assign(representative_score=local_vectors @ centroid).sort_values("representative_score", ascending=False).head(6)
        for _, row in subset.iterrows():
            st.markdown(f"**{row['topic']}**")
            st.write(row["text"][:620] + "...")

with tabs[2]:
    st.subheader("¿Con cuál programa se parecen más sus respuestas?")
    st.caption("Elija la opción más cercana a su posición y ajuste la importancia de cada tema. El resultado no es una recomendación de voto.")
    questions = load_questions()
    answers = []
    for item in questions:
        with st.expander(item["topic"], expanded=item["id"] in {"security", "economy", "health"}):
            answer = st.radio(item["question"], item["options"], key=f"answer_{item['id']}", index=None)
            weight = st.slider("Importancia", 1, 10, 5, 1, key=f"weight_{item['id']}")
            answers.append({**item, "answer": answer, "weight": weight})
    if st.button("Calcular afinidad", type="primary", use_container_width=True):
        with st.spinner("Comparando respuestas contra los programas..."):
            scores, evidence = candidate_similarity(answers, chunks, embeddings, candidate_meta, diagnostics["backend"])
        st.plotly_chart(score_bars(scores), width="stretch")
        winner = scores.iloc[0]
        st.success(f"Mayor afinidad semántica: {winner['candidate']} ({winner['score']:.1f}/100)")
        st.dataframe(scores[["candidate", "party", "position", "score"]], hide_index=True, width="stretch")
        st.subheader("Evidencia textual")
        for candidate_name in scores["candidate"].head(3):
            with st.expander(candidate_name):
                rows = evidence[evidence["candidate"] == candidate_name].sort_values("similarity", ascending=False).head(5)
                for _, row in rows.iterrows():
                    st.markdown(f"**{row['topic']}** · similitud {row['similarity']:.2f}")
                    st.write(row["evidence"] + "...")

with tabs[3]:
    st.subheader("Metodología")
    st.markdown(
        """
        1. Se extrae texto de los programas oficiales enlazados en la barra lateral.
        2. Cada programa se divide en fragmentos solapados para comparar ideas concretas, no documentos enteros.
        3. Los fragmentos se convierten en embeddings semánticos con OpenAI (`text-embedding-3-large` por defecto).
        4. La similitud general entre candidaturas se calcula sobre centroides centrados respecto al corpus completo, para evitar que todos los programas aparezcan artificialmente iguales.
        5. La visualización usa UMAP + HDBSCAN para proyectar el mapa semántico y detectar agrupaciones de fragmentos.
        6. La afinidad ciudadana compara respuestas de selección múltiple contra los fragmentos más similares por candidatura y pondera cada tema de 1 a 10.

        Repositorio del proyecto: [Edimer/programas-gobierno-presidenciales-2026](https://github.com/Edimer/programas-gobierno-presidenciales-2026).
        """
    )
    st.info("Limitación clave: los programas tienen longitudes y formatos muy distintos. Un PDF escaneado o con poco texto extraíble puede subrepresentar a una candidatura.")
    with st.expander("Diagnóstico de fuentes"):
        friendly_sources = source_rows.rename(
            columns={
                "name": "Candidatura",
                "pages": "Páginas del PDF",
                "pages_read": "Páginas leídas",
                "empty_or_sparse_pages": "Páginas con poco texto",
                "characters": "Caracteres extraídos",
                "chunks": "Fragmentos comparables",
            }
        )
        columns = ["Candidatura", "Páginas del PDF", "Páginas leídas", "Páginas con poco texto", "Caracteres extraídos", "Fragmentos comparables"]
        st.dataframe(friendly_sources[columns], hide_index=True, width="stretch")
