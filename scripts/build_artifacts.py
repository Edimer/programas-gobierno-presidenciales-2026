from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_artifacts
from src.settings import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye artefactos de análisis semántico desde los PDFs de campaña.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo de embeddings. Por defecto usa OpenAI text-embedding-3-large.")
    parser.add_argument("--max-pages", type=int, default=None, help="Límite opcional de páginas para pruebas rápidas.")
    args = parser.parse_args()
    diagnostics = build_artifacts(args.model, max_pages=args.max_pages)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
