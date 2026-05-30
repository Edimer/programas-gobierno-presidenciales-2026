from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"


def relative_candidate_similarity(chunks: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    centered = centered / np.clip(norms, 1e-9, None)
    centroids = []
    for candidate_id in chunks.drop_duplicates("candidate_id")["candidate_id"]:
        idx = chunks.index[chunks["candidate_id"] == candidate_id].to_numpy()
        centroid = centered[idx].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        centroids.append(centroid)
    matrix = np.vstack(centroids) @ np.vstack(centroids).T
    return matrix


def similarity_heatmap(candidate_meta: pd.DataFrame, chunks: pd.DataFrame, embeddings: np.ndarray) -> go.Figure:
    names = candidate_meta["candidate"].tolist()
    matrix = relative_candidate_similarity(chunks, embeddings)
    z = matrix.astype(object)
    text = np.empty(matrix.shape, dtype=object)
    hover = np.empty(matrix.shape, dtype=object)
    for row_i, row_name in enumerate(names):
        for col_i, col_name in enumerate(names):
            if row_i == col_i:
                z[row_i, col_i] = None
                text[row_i, col_i] = ""
                hover[row_i, col_i] = ""
            else:
                text[row_i, col_i] = f"{matrix[row_i, col_i]:.2f}"
                hover[row_i, col_i] = f"<b>{row_name}</b> vs <b>{col_name}</b><br>Similitud relativa: {matrix[row_i, col_i]:.2f}<extra></extra>"
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-0.35,
            zmax=1,
            colorscale=[[0, "#B91C1C"], [0.26, "#F8FAFC"], [0.62, "#4F8BC9"], [1, "#243B53"]],
            text=text,
            texttemplate="%{text}",
            hovertemplate=hover,
            showscale=False,
        )
    )
    fig.update_layout(template=TEMPLATE, height=430, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    return fig


def projection_scatter(projected: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        projected,
        x="x",
        y="y",
        color="candidate",
        hover_data={"topic": True, "text": True, "x": False, "y": False, "cluster": True},
        color_discrete_map={row["candidate"]: row["color"] for _, row in projected.drop_duplicates("candidate").iterrows()},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.74, line=dict(width=0.7, color="#111827")))
    fig.update_layout(template=TEMPLATE, height=540, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def score_bars(scores: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        scores.sort_values("score"),
        x="score",
        y="candidate",
        orientation="h",
        color="candidate",
        color_discrete_map=dict(zip(scores["candidate"], scores["color"])),
        text=scores["score"].round(1),
    )
    fig.update_layout(template=TEMPLATE, height=360, showlegend=False, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="", yaxis_title="")
    fig.update_xaxes(range=[0, 105], ticksuffix="%")
    return fig


def terms_chart(terms: pd.DataFrame, candidate: str) -> go.Figure:
    data = terms[terms["candidate"] == candidate].sort_values("score")
    fig = px.bar(data, x="score", y="term", orientation="h", color_discrete_sequence=["#243B53"])
    fig.update_layout(template=TEMPLATE, height=390, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="", yaxis_title="")
    return fig
