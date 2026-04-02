from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import get_settings


def stable_chunk_id(*, source: str, chunk_index: int) -> str:
    """Deterministic id from source path and index within that source (post-split order)."""
    payload = f"{source}\0{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_docs(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=list(settings.chunk_separators),
    )
    chunks = text_splitter.split_documents(docs)
    per_source: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        idx = per_source.get(source, 0)
        per_source[source] = idx + 1
        chunk.metadata["chunk_index_in_source"] = idx
        chunk.metadata["chunk_id"] = stable_chunk_id(source=source, chunk_index=idx)
    return chunks


def process_docs(docs: list[Document]) -> list[dict[str, Any]]:
    """Legacy dict view of chunks (id + text + metadata) for notebooks or debugging."""
    chunks = chunk_docs(docs)
    return [
        {
            "id": c.metadata["chunk_id"],
            "chunk": c.page_content,
            "metadata": dict(c.metadata),
        }
        for c in chunks
    ]
