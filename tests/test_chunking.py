from langchain_core.documents import Document

from rag.chunking.chunk_docs import chunk_docs, stable_chunk_id


def test_stable_chunk_id_is_deterministic() -> None:
    a = stable_chunk_id(source="/tmp/a.pdf", chunk_index=0)
    b = stable_chunk_id(source="/tmp/a.pdf", chunk_index=0)
    assert a == b
    assert stable_chunk_id(source="/tmp/a.pdf", chunk_index=1) != a


def test_chunk_docs_assigns_chunk_id() -> None:
    text = "Hello world. " * 200
    docs = [Document(page_content=text, metadata={"source": "x.pdf"})]
    chunks = chunk_docs(docs)
    assert chunks
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["chunk_id"] == stable_chunk_id(
        source="x.pdf",
        chunk_index=0,
    )
