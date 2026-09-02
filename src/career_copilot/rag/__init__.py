"""RAG: turn the portfolio markdown files into a searchable, cited knowledge base.

Pipeline: loader.py (read + parse frontmatter) -> chunker.py (split into retrievable
pieces) -> store.py (embed + persist to Chroma) -> retriever.py (query at agent run time).

Every chunk keeps a stable chunk_id and its source doc id, because Phase 4's critic node
needs to verify that every sentence the writer produces traces back to a real chunk —
that's the whole point of doing RAG "properly" instead of just stuffing everything into
the prompt.
"""
