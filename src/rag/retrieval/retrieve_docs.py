from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from rag.chunking.chunk_docs import chunk_docs
from rag.config import get_settings
from rag.ingestion.ingest_docs import ingest_docs
from rag.vector_store.store_documents import add_documents, get_vector_store

logger = logging.getLogger(__name__)


def index_pdfs(pdfs_dir: Path | None = None) -> int:
    """Load PDFs, chunk, embed, and persist to Chroma. Returns number of chunks written."""
    settings = get_settings()
    pdfs_dir = pdfs_dir or (settings.data_dir / "raw" / "pdfs")
    raw_docs = ingest_docs(pdfs_dir)
    chunks = chunk_docs(raw_docs)
    store = get_vector_store()
    add_documents(store, chunks)
    return len(chunks)


def get_retriever(
    k: int | None = None,
    filters: dict | None = None,
) -> BaseRetriever:
    """Return a retriever over the persisted vector store.

    When reranking is enabled, fetches `retrieval_k_fetch` candidates so the
    cross-encoder has enough material to work with.  When hybrid search is
    enabled, combines BM25 + dense (Chroma) results via EnsembleRetriever.

    Raises RuntimeError if the vector store is empty (not yet indexed).
    """
    settings = get_settings()
    final_k = k or settings.retrieval_k
    fetch_k = settings.retrieval_k_fetch if settings.enable_reranking else final_k

    store = get_vector_store()

    doc_count = store._collection.count()
    if doc_count == 0:
        raise RuntimeError(
            "Vector store is empty. Index your documents first via POST /index "
            "or by running the indexing pipeline."
        )

    search_kwargs: dict = {"k": fetch_k}
    if filters:
        search_kwargs["filter"] = filters

    chroma_retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    if not settings.enable_hybrid_search:
        return chroma_retriever

    # Build BM25 index from all stored documents
    raw = store.get()
    bm25_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
        if text
    ]
    if not bm25_docs:
        return chroma_retriever

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = fetch_k

    bm25_weight = settings.hybrid_bm25_weight
    return EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[bm25_weight, 1.0 - bm25_weight],
    )


def retrieve_documents(k: int = 3) -> BaseRetriever:
    """Deprecated alias: use ``get_retriever``."""
    return get_retriever(k=k)
