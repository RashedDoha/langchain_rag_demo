"""Cross-encoder reranker: scores (query, chunk) pairs and returns the top-k chunks."""

from __future__ import annotations

from threading import Lock

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_cross_encoder: CrossEncoder | None = None
_cross_encoder_model: str | None = None
_lock = Lock()


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    global _cross_encoder, _cross_encoder_model
    with _lock:
        if _cross_encoder is None or _cross_encoder_model != model_name:
            _cross_encoder = CrossEncoder(model_name)
            _cross_encoder_model = model_name
    return _cross_encoder


def rerank(
    query: str,
    docs: list[Document],
    k: int,
    model_name: str,
) -> list[Document]:
    """Score each (query, doc) pair with a cross-encoder and return the top-k docs."""
    if not docs:
        return docs
    encoder = _get_cross_encoder(model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores: list[float] = encoder.predict(pairs).tolist()
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # Attach reranker score to metadata for logging/debugging
    result = []
    for score, doc in ranked[:k]:
        doc.metadata["reranker_score"] = round(float(score), 4)
        result.append(doc)
    return result
