from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

from .settings import CANDIDATES_PATH, QUESTIONS_PATH, ROOT


def load_candidates() -> list[dict]:
    return json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", text)
    return text.strip()


def extract_pdf_text(path: str | Path, max_pages: int | None = None) -> tuple[str, dict]:
    pdf_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    page_texts = []
    empty_pages = 0
    for page in pages:
        page_text = page.extract_text() or ""
        if len(page_text.strip()) < 40:
            empty_pages += 1
        page_texts.append(page_text)
    text = clean_text("\n".join(page_texts))
    meta = {
        "file": str(pdf_path),
        "pages": len(reader.pages),
        "pages_read": len(pages),
        "empty_or_sparse_pages": empty_pages,
        "characters": len(text),
    }
    return text, meta


def chunk_text(text: str, words: int = 180, overlap: int = 45) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, words - overlap)
    for start in range(0, len(tokens), step):
        chunk = " ".join(tokens[start : start + words])
        if len(chunk) > 220:
            chunks.append(chunk)
        if start + words >= len(tokens):
            break
    return chunks
