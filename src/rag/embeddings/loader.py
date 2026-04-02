from __future__ import annotations

from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import get_settings

_embeddings: HuggingFaceEmbeddings | None = None
_lock = Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    with _lock:
        if _embeddings is None:
            s = get_settings()
            _embeddings = HuggingFaceEmbeddings(
                model_name=s.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return _embeddings


def load_embeddings() -> HuggingFaceEmbeddings:
    """Alias for ``get_embeddings`` (backward compatible name)."""
    return get_embeddings()
