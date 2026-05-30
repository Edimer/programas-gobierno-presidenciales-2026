from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"

DEFAULT_MODEL = "text-embedding-3-large"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 45
TOP_K_CHUNKS = 12

TOPIC_KEYWORDS = {
    "Economía y empleo": [
        "empleo", "trabajo", "laboral", "productividad", "empresa", "crecimiento", "industria",
        "competitividad", "emprendimiento", "formalización", "salario", "pymes", "turismo",
    ],
    "Social, salud y educación": [
        "salud", "educación", "colegio", "universidad", "docente", "hospital", "eps", "paciente",
        "pobreza", "cuidado", "mujeres", "jóvenes", "niñez", "vivienda", "pensiones",
    ],
    "Seguridad, justicia y paz": [
        "seguridad", "policía", "fuerza pública", "justicia", "delito", "crimen", "extorsión",
        "paz", "conflicto", "armados", "cárcel", "narcotráfico", "corrupción",
    ],
    "Minero-energético y ambiente": [
        "energía", "petróleo", "gas", "minería", "minero", "transición", "renovable",
        "ambiente", "clima", "agua", "deforestación", "biodiversidad", "licencia ambiental",
    ],
    "Agrario, rural y regiones": [
        "campo", "rural", "agro", "agrario", "campesino", "tierra", "regiones", "territorio",
        "municipios", "vías terciarias", "alimentos", "reforma rural",
    ],
    "Instituciones y Estado": [
        "estado", "instituciones", "democracia", "congreso", "constitución", "descentralización",
        "reforma", "transparencia", "gobernanza", "participación", "digital", "datos",
    ],
}
