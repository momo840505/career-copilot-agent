"""CLI smoke test: ask the portfolio index a question and see what comes back.

    python scripts/query_demo.py "does she have SQL experience?"

If this prints back relevant, correctly-attributed chunks, retrieval is working end to end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_copilot.config import get_settings
from career_copilot.rag.retriever import search


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Does she have SQL and cloud deployment experience?"
    settings = get_settings()
    hits = search(settings, query, k=4)

    print(f'Query: "{query}"\n')
    for h in hits:
        print(f"[{h.chunk_id}]  (distance={h.distance:.3f})  from: {h.doc_title}")
        print(f"  {h.text[:220]}{'...' if len(h.text) > 220 else ''}\n")


if __name__ == "__main__":
    main()
