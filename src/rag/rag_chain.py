from __future__ import annotations

import logging

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

from rag.config import get_settings
from rag.prompts.loader import format_rag_user_prompt, load_prompt
from rag.reranking import rerank
from rag.retrieval.retrieve_docs import get_retriever

logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", "")
        label = f"{source}" + (f" [{cid}]" if cid else "")
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(parts)


def _retrieve_rerank_log(
    x: dict,
    retriever: BaseRetriever,
    settings,
    request_id: str | None,
) -> list[Document]:
    docs = retriever.invoke(x["question"])

    if settings.enable_reranking and docs:
        docs = rerank(x["question"], docs, settings.retrieval_k, settings.reranker_model)

    req_id = request_id or x.get("request_id", "unknown")
    logger.info(
        "Retrieved %d chunks for request_id=%s question=%r",
        len(docs),
        req_id,
        x["question"],
    )
    for i, doc in enumerate(docs):
        logger.info(
            "  rank=%d source=%s chunk_id=%s reranker_score=%s",
            i,
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("chunk_id", ""),
            doc.metadata.get("reranker_score", "n/a"),
        )

    return docs


def create_rag_chain(
    *,
    retriever: BaseRetriever | None = None,
    k: int | None = None,
    temperature: float = 0.0,
    filters: dict | None = None,
    request_id: str | None = None,
):
    """LCEL chain: retrieve (+rerank) → log → system + user prompts → chat model → string."""
    settings = get_settings()
    r = retriever or get_retriever(k=k, filters=filters)
    llm = init_chat_model(settings.rag_llm_model, temperature=temperature)
    system_text = load_prompt("system_rag.txt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system}"),
            ("human", "{human}"),
        ]
    )
    return (
        RunnablePassthrough.assign(
            documents=lambda x: _retrieve_rerank_log(x, r, settings, request_id),
        )
        | RunnablePassthrough.assign(
            system=lambda _: system_text,
            human=lambda x: format_rag_user_prompt(
                context=format_docs(x["documents"]),
                question=x["question"],
            ),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
