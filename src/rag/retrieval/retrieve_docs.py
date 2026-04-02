from __future__ import annotations

from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from rag.chunking.chunk_docs import chunk_docs
from rag.config import get_settings
from rag.ingestion.ingest_docs import ingest_docs
from rag.vector_store.store_documents import add_documents, get_vector_store


def index_pdfs(pdfs_dir: Path | None = None) -> int:
    """Load PDFs, chunk, embed, and persist to Chroma. Returns number of chunks written."""
    settings = get_settings()
    pdfs_dir = pdfs_dir or (settings.data_dir / "raw" / "pdfs")
    raw_docs = ingest_docs(pdfs_dir)
    chunks = chunk_docs(raw_docs)
    store = get_vector_store()
    add_documents(store, chunks)
    return len(chunks)


def get_retriever(k: int = 3) -> VectorStoreRetriever:
    """Return a retriever over the persisted vector store (no re-indexing)."""
    return get_vector_store().as_retriever(search_type="similarity", k=k)


def retrieve_documents(k: int = 3) -> VectorStoreRetriever:
    """Deprecated alias: use ``get_retriever`` (this no longer re-indexes)."""
    return get_retriever(k=k)
