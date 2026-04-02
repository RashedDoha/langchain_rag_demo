from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def ingest_docs(docs_dir: Path):
    print(f"Ingesting documents from {docs_dir}")
    docs = []
    for file in docs_dir.glob("**/*.pdf"):
        loader = PyPDFLoader(file)
        document = loader.load()
        docs.extend(document)
    return docs
    