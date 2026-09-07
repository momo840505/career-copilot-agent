from types import SimpleNamespace

from career_copilot.rag import store
from career_copilot.rag.chunker import Chunk


class _Collection:
    def __init__(self):
        self.upsert_kwargs = None

    def upsert(self, **kwargs):
        self.upsert_kwargs = kwargs


class _Client:
    def __init__(self):
        self.deleted = []
        self.collection = _Collection()

    def list_collections(self):
        return [SimpleNamespace(name=store.COLLECTION_NAME)]

    def delete_collection(self, name):
        self.deleted.append(name)

    def create_collection(self, **kwargs):
        return self.collection


def test_index_rebuild_removes_the_old_collection(monkeypatch):
    client = _Client()
    monkeypatch.setattr(store, "_client", lambda settings: client)
    monkeypatch.setattr(store, "_embedding_function", lambda settings: object())

    chunks = [
        Chunk(
            chunk_id="doc::chunk0",
            doc_id="doc",
            doc_title="Doc",
            tags=["test"],
            text="current text",
        )
    ]

    count = store.index_chunks(object(), chunks)

    assert count == 1
    assert client.deleted == [store.COLLECTION_NAME]
    assert client.collection.upsert_kwargs["ids"] == ["doc::chunk0"]
