"""CLI: read every portfolio markdown file, chunk it, embed it, and persist to Chroma.

Run this once after setting up .env, and again any time you edit the files under
src/career_copilot/data/portfolio/.

    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.config import get_settings
from career_copilot.rag.chunker import chunk_docs
from career_copilot.rag.loader import load_portfolio_docs
from career_copilot.rag.store import index_chunks


def main() -> None:
    settings = get_settings()
    docs = load_portfolio_docs(settings.portfolio_dir)
    print(f"Loaded {len(docs)} source docs from {settings.portfolio_dir}")
    for d in docs:
        print(f"  - {d.doc_id}: {d.title}")

    chunks = chunk_docs(docs)
    print(f"\nSplit into {len(chunks)} chunks.")

    n = index_chunks(settings, chunks)
    print(f"\nIndexed {n} chunks into Chroma at {settings.chroma_dir}")


if __name__ == "__main__":
    main()
