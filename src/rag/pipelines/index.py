from __future__ import annotations
from pathlib import Path
import sys

from langchain_community.vectorstores import VectorStore

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag.retrieval import index_pdfs
from rag import get_settings


def index_documents(vectorstore: VectorStore | None = None) -> int:
    settings = get_settings()
    pdfs_dir = settings.data_dir / "raw" / "pdfs"
    if not pdfs_dir.exists():
        print(f"PDFs directory {pdfs_dir} does not exist. Skipping indexing.")
        return 0
    print(f"Indexing PDFs from {pdfs_dir}...")
    n = index_pdfs()
    print(f"Indexed {n} chunks.")
    return n