from __future__ import annotations

from typing import Iterator

from rag.rag_chain import create_rag_chain


def invoke_query(
    question: str,
    *,
    filters: dict | None = None,
    request_id: str | None = None,
) -> Iterator[str]:
    chain = create_rag_chain(filters=filters, request_id=request_id)
    return chain.stream({"question": question, "request_id": request_id})
