from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_artifacts
from src.settings import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic analysis artifacts from campaign PDFs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for quick tests.")
    args = parser.parse_args()
    diagnostics = build_artifacts(args.model, max_pages=args.max_pages)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
