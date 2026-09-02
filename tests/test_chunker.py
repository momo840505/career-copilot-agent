"""Chunker tests — pure logic, no API key needed."""
from career_copilot.rag.chunker import chunk_docs
from career_copilot.rag.loader import SourceDoc


def _fake_doc(doc_id: str, n_paragraphs: int = 5) -> SourceDoc:
    content = "\n\n".join(
        f"This is paragraph {i} of the fake document, with enough padding text to matter "
        f"for chunk-size calculations rather than being trivially short."
        for i in range(n_paragraphs)
    )
    return SourceDoc(doc_id=doc_id, title=doc_id, tags=["test"], content=content)


def test_chunk_ids_are_stable_and_traceable_to_source_doc():
    docs = [_fake_doc("doc_a"), _fake_doc("doc_b")]
    chunks = chunk_docs(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 0
    for c in chunks:
        assert c.chunk_id.startswith(f"{c.doc_id}::chunk")
        assert c.doc_id in ("doc_a", "doc_b")
        assert c.text  # never an empty chunk


def test_chunk_ids_unique():
    docs = [_fake_doc("doc_a", n_paragraphs=10)]
    chunks = chunk_docs(docs, chunk_size=150, chunk_overlap=10)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique — the critic cites by chunk_id"


def test_small_doc_still_produces_at_least_one_chunk():
    doc = SourceDoc(doc_id="tiny", title="tiny", tags=[], content="Just one short sentence.")
    chunks = chunk_docs([doc])
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."
