from types import SimpleNamespace

from career_copilot.rag import retriever


class _Collection:
    def query(self, **kwargs):
        return {
            "ids": [["near", "far"]],
            "documents": [["near evidence", "irrelevant evidence"]],
            "metadatas": [[
                {"doc_id": "a", "doc_title": "A"},
                {"doc_id": "b", "doc_title": "B"},
            ]],
            "distances": [[0.2, 0.9]],
            "embeddings": [[[1.0, 0.0], [0.0, 1.0]]],
        }


def test_search_filters_weak_matches_before_mmr(monkeypatch):
    monkeypatch.setattr(retriever, "get_collection", lambda settings: _Collection())
    settings = SimpleNamespace(rag_max_distance=0.75)

    results = retriever.search(settings, "SQL", k=2)

    assert [item.chunk_id for item in results] == ["near"]
