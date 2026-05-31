from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"


SHORT_AXIS_LABELS = {
    "Mercado / Estado": "Mercado ←→ Estado",
    "Seguridad": "Punición ←→ Prevención",
    "Paz y conflicto": "Confrontación ←→ Negociación",
    "Ambiente y energía": "Extractivismo ←→ Transición",
    "Política social": "Focalización ←→ Universalismo",
    "Instituciones": "Autoridad ←→ Reforma",
    "Economía fiscal": "Austeridad ←→ Inversión pública",
}


SHORT_POLE_LABELS = {
    "Prevención y derechos": "Prevención",
    "Transición ecológica": "Transición",
    "Participación y reforma": "Reforma",
    "Orden y autoridad": "Autoridad",
    "Austeridad e impuestos bajos": "Austeridad",
    "Inversión pública y progresividad": "Inversión pública",
}


def readable_position(row: pd.Series) -> str:
    score = float(row["score"])
    absolute_score = abs(score)
    if absolute_score < 0.15:
        return "Mixto"
    intensity = "fuerte" if absolute_score >= 0.7 else "media" if absolute_score >= 0.4 else "leve"
    label = row["positive_label"] if score > 0 else row["negative_label"]
    label = SHORT_POLE_LABELS.get(label, label)
    return f"{label}<br>{intensity}"


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


def positioning_heatmap(positions: pd.DataFrame) -> go.Figure:
    matrix = positions.pivot(index="candidate", columns="axis", values="score")
    candidate_order = positions.drop_duplicates("candidate")["candidate"].tolist()
    axis_order = positions.drop_duplicates("axis")["axis"].tolist()
    matrix = matrix.reindex(index=candidate_order, columns=axis_order)
    label_lookup = {
        (row["candidate"], row["axis"]): readable_position(row)
        for _, row in positions.iterrows()
    }
    text = matrix.copy().astype(object)
    for candidate in matrix.index:
        for axis in matrix.columns:
            text.loc[candidate, axis] = label_lookup.get((candidate, axis), "")
    hover = []
    for candidate in matrix.index:
        row_hover = []
        for axis in matrix.columns:
            row = positions[(positions["candidate"] == candidate) & (positions["axis"] == axis)].iloc[0]
            row_hover.append(
                f"<b>{candidate}</b><br>{axis}<br>"
                f"{row['negative_label']} ← {row['score']:+.2f} → {row['positive_label']}<br>"
                f"Cobertura de evidencia: {row['coverage']:.1%}<extra></extra>"
            )
        hover.append(row_hover)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#B45309"], [0.5, "#F8FAFC"], [1, "#0F766E"]],
            text=text.to_numpy(),
            texttemplate="%{text}",
            hovertemplate=hover,
            showscale=False,
        )
    )
    fig.update_layout(template=TEMPLATE, height=500, margin=dict(l=10, r=10, t=20, b=20))
    fig.update_xaxes(
        tickangle=-20,
        tickmode="array",
        tickvals=axis_order,
        ticktext=[SHORT_AXIS_LABELS.get(axis, axis) for axis in axis_order],
    )
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
