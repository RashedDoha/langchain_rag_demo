"""Central settings: paths, models, and chunking (env-overridable)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    chroma_collection_name: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    chunk_separators: tuple[str, ...]
    rag_llm_model: str


def get_settings() -> Settings:
    root = project_root()
    data_dir = Path(os.getenv("RAG_DATA_DIR", str(root / "data"))).resolve()
    sep_raw = os.getenv("RAG_CHUNK_SEPARATORS")
    if sep_raw:
        chunk_separators = tuple(s.replace("\\n", "\n") for s in sep_raw.split("|"))
    else:
        chunk_separators = ("\n\n", "\n", " ", "")
    return Settings(
        data_dir=data_dir,
        chroma_collection_name=os.getenv("RAG_CHROMA_COLLECTION", "rag_collection"),
        embedding_model_name=os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
        chunk_separators=chunk_separators,
        rag_llm_model=os.getenv("RAG_LLM_MODEL", "anthropic:claude-sonnet-4-6"),
    )
