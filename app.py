from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.io import load_questions
from src.pipeline import build_artifacts, latest_diagnostics
from src.scoring import candidate_similarity
from src.settings import ARTIFACT_DIR, DEFAULT_MODEL, ROOT
from src.visuals import projection_scatter, score_bars, similarity_heatmap, terms_chart


st.set_page_config(page_title="Programas presidenciales Colombia 2026", page_icon="🗳️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1180px;}
    h1, h2, h3 {letter-spacing: 0 !important;}
    div[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px; padding: 8px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_artifacts():
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
    return diagnostics, chunks, projected, terms, candidate_meta, embeddings, centroids


def reset_artifacts(model_name: str):
    build_artifacts(model_name)
    st.cache_data.clear()
    st.rerun()


with st.sidebar:
    st.caption("Modelo")
    model_name = st.text_input("Embedding model", value=DEFAULT_MODEL, help="Recomendado: intfloat/multilingual-e5-base. Tambien puede probar paraphrase-multilingual-mpnet-base-v2.")
    if st.button("Recalcular analisis", use_container_width=True):
        with st.spinner("Extrayendo PDFs y recalculando embeddings..."):
            reset_artifacts(model_name)

diagnostics, chunks, projected, terms, candidate_meta, embeddings, centroids = load_artifacts()

st.title("Programas presidenciales Colombia 2026")
st.caption("Explorador semantico de propuestas: similitud, agrupacion tematica y afinidad por respuestas ciudadanas.")

metric_cols = st.columns(4)
metric_cols[0].metric("Candidaturas", f"{len(candidate_meta)}")
metric_cols[1].metric("Fragmentos analizados", f"{len(chunks):,}")
metric_cols[2].metric("Modelo", diagnostics["backend"])
metric_cols[3].metric("Metodo", diagnostics["method"])

tabs = st.tabs(["Mapa semantico", "Comparador", "Afinidad ciudadana", "Metodo"])

with tabs[0]:
    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Territorio semantico de los programas")
        st.plotly_chart(projection_scatter(projected), use_container_width=True)
    with right:
        st.subheader("Similitud entre candidaturas")
        st.plotly_chart(similarity_heatmap(candidate_meta, centroids), use_container_width=True)
        st.caption("Cada punto del mapa es un fragmento del programa. La distancia es aproximada y sirve para explorar, no para ordenar ideologicamente.")

with tabs[1]:
    candidate = st.selectbox("Candidatura", candidate_meta["candidate"].tolist())
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Terminos distintivos")
        st.plotly_chart(terms_chart(terms, candidate), use_container_width=True)
    with col_b:
        st.subheader("Fragmentos representativos")
        subset = projected[projected["candidate"] == candidate].sample(min(8, (projected["candidate"] == candidate).sum()), random_state=7)
        for _, row in subset.iterrows():
            st.markdown(f"**Cluster {row['cluster']}**")
            st.write(row["text"][:620] + "...")

with tabs[2]:
    st.subheader("Con cual programa se parecen mas sus respuestas?")
    st.caption("Responda en sus palabras. El resultado compara sus textos contra fragmentos de los programas; no es una recomendacion de voto.")
    questions = load_questions()
    answers = []
    for item in questions:
        with st.expander(item["topic"], expanded=item["id"] in {"security", "economy", "health"}):
            answer = st.text_area(item["question"], key=f"answer_{item['id']}", height=92)
            weight = st.slider("Importancia", 0.5, 2.0, 1.0, 0.25, key=f"weight_{item['id']}")
            answers.append({**item, "answer": answer, "weight": weight})
    if st.button("Calcular afinidad", type="primary", use_container_width=True):
        with st.spinner("Comparando respuestas contra los programas..."):
            scores, evidence = candidate_similarity(answers, chunks, embeddings, candidate_meta, diagnostics["backend"])
        st.plotly_chart(score_bars(scores), use_container_width=True)
        winner = scores.iloc[0]
        st.success(f"Mayor afinidad semantica: {winner['candidate']} ({winner['score']:.1f}/100)")
        st.dataframe(scores[["candidate", "party", "position", "score"]], hide_index=True, use_container_width=True)
        st.subheader("Evidencia textual")
        for candidate_name in scores["candidate"].head(3):
            with st.expander(candidate_name):
                rows = evidence[evidence["candidate"] == candidate_name].sort_values("similarity", ascending=False).head(5)
                for _, row in rows.iterrows():
                    st.markdown(f"**{row['topic']}** · similitud {row['similarity']:.2f}")
                    st.write(row["evidence"] + "...")

with tabs[3]:
    st.subheader("Metodo")
    st.markdown(
        """
        1. Se extrae texto de cada PDF oficial disponible en `pdfs/`.
        2. Cada programa se divide en fragmentos solapados para comparar ideas concretas, no documentos enteros.
        3. Los fragmentos se convierten en embeddings semanticos multilingues.
        4. La similitud se calcula con coseno entre vectores normalizados.
        5. La visualizacion usa UMAP + HDBSCAN cuando estan disponibles; si fallan, usa PCA + KMeans.
        6. La afinidad ciudadana promedia los fragmentos mas similares por candidatura y por pregunta.
        """
    )
    st.info("Limitacion clave: los programas tienen longitudes y formatos muy distintos. Un PDF escaneado o con poco texto extraible puede subrepresentar a una candidatura.")
    with st.expander("Diagnostico de fuentes"):
        st.json(diagnostics["source_pdfs"])
