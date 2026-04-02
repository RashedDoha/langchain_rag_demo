from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import get_settings
from rag.prompts.loader import format_rag_user_prompt, load_prompt
from rag.retrieval.retrieve_docs import get_retriever


def format_docs(docs: list[Document]) -> str:
    parts: list[str] = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        cid = d.metadata.get("chunk_id", "")
        label = f"{source}" + (f" [{cid}]" if cid else "")
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(parts)


def create_rag_chain(
    *,
    retriever: VectorStoreRetriever | None = None,
    k: int = 3,
    temperature: float = 0.0,
):
    """LCEL chain: retrieve → system + user prompts → chat model → string."""
    settings = get_settings()
    r = retriever or get_retriever(k=k)
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
            documents=lambda x: r.invoke(x["question"]),
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
