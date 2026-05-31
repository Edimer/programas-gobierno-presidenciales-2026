from __future__ import annotations

import re
import unicodedata

import pandas as pd


AXES = [
    {
        "id": "state_market",
        "axis": "Mercado / Estado",
        "negative_label": "Mercado",
        "positive_label": "Estado",
        "negative_terms": [
            "inversion privada", "empresa privada", "sector privado", "libre empresa", "competitividad",
            "reduccion de impuestos", "bajar impuestos", "menos tramites", "desregulacion", "alianzas publico privadas",
            "seguridad juridica", "emprendimiento", "atraccion de inversion",
        ],
        "positive_terms": [
            "estado social", "inversion publica", "empresa publica", "redistribucion", "derechos sociales",
            "economia popular", "servicios publicos", "proteccion social", "justicia social", "igualdad",
            "intervencion del estado", "fortalecer lo publico",
        ],
    },
    {
        "id": "security",
        "axis": "Seguridad",
        "negative_label": "Punición",
        "positive_label": "Prevención y derechos",
        "negative_terms": [
            "mano dura", "fuerza publica", "pie de fuerza", "control territorial", "carcel", "judicializacion",
            "capturas", "policia", "militares", "sometimiento", "extincion de dominio", "ofensiva",
            "seguridad ciudadana", "seguridad nacional",
        ],
        "positive_terms": [
            "prevencion", "derechos humanos", "justicia restaurativa", "convivencia", "oportunidades",
            "inversion social", "jovenes", "paz territorial", "mediacion", "cuidado", "tejido social",
            "seguridad humana",
        ],
    },
    {
        "id": "peace_conflict",
        "axis": "Paz y conflicto",
        "negative_label": "Confrontación",
        "positive_label": "Negociación",
        "negative_terms": [
            "grupos armados", "combate", "ofensiva", "sometimiento", "judicializacion", "narcotrafico",
            "control territorial", "fuerza publica", "delincuencia organizada", "terrorismo",
        ],
        "positive_terms": [
            "acuerdo de paz", "implementacion del acuerdo", "negociacion", "dialogo", "paz total",
            "reconciliacion", "victimas", "justicia transicional", "reparacion", "sustitucion de cultivos",
            "paz territorial",
        ],
    },
    {
        "id": "environment_energy",
        "axis": "Ambiente y energía",
        "negative_label": "Extractivismo",
        "positive_label": "Transición ecológica",
        "negative_terms": [
            "petroleo", "gas", "mineria", "carbon", "hidrocarburos", "seguridad energetica",
            "licencias ambientales", "exploracion", "explotacion", "sector minero energetico",
        ],
        "positive_terms": [
            "transicion energetica", "energias renovables", "energia renovable", "biodiversidad",
            "cambio climatico", "deforestacion", "proteccion ambiental", "agua", "justicia climatica",
            "economia verde", "restauracion ecologica",
        ],
    },
    {
        "id": "social_policy",
        "axis": "Política social",
        "negative_label": "Focalización",
        "positive_label": "Universalismo",
        "negative_terms": [
            "focalizacion", "subsidios focalizados", "eficiencia del gasto", "poblacion vulnerable",
            "transferencias condicionadas", "merito", "evaluacion de resultados", "sostenibilidad",
        ],
        "positive_terms": [
            "derecho a la salud", "derecho a la educacion", "derechos sociales", "universal",
            "gratuidad", "renta basica", "salud publica", "educacion publica", "cuidado",
            "sistema nacional de cuidado", "bienestar",
        ],
    },
    {
        "id": "institutions",
        "axis": "Instituciones",
        "negative_label": "Orden y autoridad",
        "positive_label": "Participación y reforma",
        "negative_terms": [
            "autoridad", "orden", "cumplimiento de la ley", "meritocracia", "disciplina fiscal",
            "control", "inspeccion vigilancia", "seguridad juridica", "estabilidad institucional",
        ],
        "positive_terms": [
            "participacion ciudadana", "democracia participativa", "reforma politica", "descentralizacion",
            "transparencia", "gobierno abierto", "datos abiertos", "control ciudadano",
            "veedurias", "dialogo social",
        ],
    },
    {
        "id": "fiscal_economy",
        "axis": "Economía fiscal",
        "negative_label": "Austeridad e impuestos bajos",
        "positive_label": "Inversión pública y progresividad",
        "negative_terms": [
            "austeridad", "regla fiscal", "reducir impuestos", "bajar impuestos", "gasto eficiente",
            "deficit fiscal", "deuda publica", "confianza inversionista", "inversion privada",
        ],
        "positive_terms": [
            "impuestos progresivos", "progresividad", "reforma tributaria", "gasto social",
            "inversion publica", "redistribucion", "financiacion publica", "justicia tributaria",
            "lucha contra la evasion", "evasion",
        ],
    },
]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_text)


def count_term_hits(text: str, terms: list[str]) -> tuple[int, list[str]]:
    hits = []
    for term in terms:
        normalized = normalize_text(term)
        pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
        if re.search(pattern, text):
            hits.append(term)
    return len(hits), hits


def programmatic_positions(chunks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    evidence_rows = []
    for candidate_id, group in chunks.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        texts = [(row.chunk_id, normalize_text(row.text), row.text) for row in group.itertuples()]
        for axis in AXES:
            positive_hits = 0
            negative_hits = 0
            matched_chunks = 0
            chunk_evidence = []
            for chunk_id, normalized, original in texts:
                neg_count, neg_terms = count_term_hits(normalized, axis["negative_terms"])
                pos_count, pos_terms = count_term_hits(normalized, axis["positive_terms"])
                if neg_count or pos_count:
                    matched_chunks += 1
                    positive_hits += pos_count
                    negative_hits += neg_count
                    chunk_evidence.append(
                        {
                            "chunk_id": chunk_id,
                            "positive_hits": pos_count,
                            "negative_hits": neg_count,
                            "positive_terms": ", ".join(pos_terms),
                            "negative_terms": ", ".join(neg_terms),
                            "text": original,
                        }
                    )
            denominator = positive_hits + negative_hits
            score = (positive_hits - negative_hits) / denominator if denominator else 0.0
            coverage = matched_chunks / max(len(group), 1)
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate": first["candidate"],
                    "color": first["color"],
                    "axis_id": axis["id"],
                    "axis": axis["axis"],
                    "negative_label": axis["negative_label"],
                    "positive_label": axis["positive_label"],
                    "score": score,
                    "coverage": coverage,
                    "positive_hits": positive_hits,
                    "negative_hits": negative_hits,
                    "matched_chunks": matched_chunks,
                }
            )
            ranked = sorted(
                chunk_evidence,
                key=lambda row: abs(row["positive_hits"] - row["negative_hits"]),
                reverse=True,
            )[:3]
            for row in ranked:
                direction = axis["positive_label"] if row["positive_hits"] >= row["negative_hits"] else axis["negative_label"]
                evidence_rows.append(
                    {
                        "candidate": first["candidate"],
                        "axis": axis["axis"],
                        "direction": direction,
                        "positive_terms": row["positive_terms"],
                        "negative_terms": row["negative_terms"],
                        "evidence": row["text"][:620],
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(evidence_rows)
