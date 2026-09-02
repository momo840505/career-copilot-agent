"""Embed chunks with OpenAI and persist them in a local Chroma collection on disk.

Using chromadb's client directly (rather than a framework wrapper) so it's obvious what's
actually happening: chunks in -> OpenAI embedding API call -> vectors stored on disk,
indexed for cosine-similarity search.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from career_copilot.config import Settings
from career_copilot.rag.chunker import Chunk

COLLECTION_NAME = "career_copilot_portfolio"


def get_collection(settings: Settings):
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embed_model,
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(settings: Settings, chunks: list[Chunk]) -> int:
    """(Re)index all chunks. Safe to re-run — upsert replaces existing IDs."""
    collection = get_collection(settings)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"doc_id": c.doc_id, "doc_title": c.doc_title, "tags": ",".join(c.tags)}
            for c in chunks
        ],
    )
    return len(chunks)
