from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import get_settings
from rag.embeddings.loader import get_embeddings


def get_vector_store() -> Chroma:
    s = get_settings()
    persist = s.data_dir / "vector_store" / "chroma_db"
    return Chroma(
        collection_name=s.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist),
    )


def add_documents(vectorstore: Chroma, documents: list[Document]) -> None:
    if not documents:
        return
    ids = [str(d.metadata["chunk_id"]) for d in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
