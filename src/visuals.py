from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"


def similarity_heatmap(candidate_meta: pd.DataFrame, centroids: np.ndarray) -> go.Figure:
    names = candidate_meta["candidate"].tolist()
    matrix = centroids @ centroids.T
    fig = px.imshow(
        matrix,
        x=names,
        y=names,
        zmin=0,
        zmax=1,
        color_continuous_scale=["#F8FAFC", "#D9E8F5", "#4F8BC9", "#243B53"],
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(template=TEMPLATE, height=430, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    return fig


def projection_scatter(projected: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        projected,
        x="x",
        y="y",
        color="candidate",
        symbol="cluster",
        hover_data={"text": True, "x": False, "y": False, "cluster": True},
        color_discrete_map={row["candidate"]: row["color"] for _, row in projected.drop_duplicates("candidate").iterrows()},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.72, line=dict(width=0)))
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
    fig.update_layout(template=TEMPLATE, height=360, margin=dict(l=0, r=10, t=10, b=0), xaxis_title="", yaxis_title="")
    return fig
