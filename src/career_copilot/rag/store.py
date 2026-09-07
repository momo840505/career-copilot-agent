from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from career_copilot.config import Settings
from career_copilot.rag.chunker import Chunk

COLLECTION_NAME = "career_copilot_portfolio"


def _client(settings: Settings):
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _embedding_function(settings: Settings):
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embed_model,
    )


def get_collection(settings: Settings):
    return _client(settings).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(settings),
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(settings: Settings, chunks: list[Chunk]) -> int:
    """Replace the collection with the current corpus."""
    client = _client(settings)
    existing = {getattr(item, "name", str(item)) for item in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(settings),
        metadata={"hnsw:space": "cosine"},
    )
    if chunks:
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "tags": ",".join(chunk.tags),
                }
                for chunk in chunks
            ],
        )
    return len(chunks)
