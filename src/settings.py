from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
CHUNK_WORDS = 180
CHUNK_OVERLAP = 45
TOP_K_CHUNKS = 12
