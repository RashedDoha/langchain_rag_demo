from __future__ import annotations
from typing import Iterator


from rag.rag_chain import create_rag_chain

def invoke_query(question: str) -> Iterator[str]:
    chain = create_rag_chain()
    return chain.stream({"question": question})

