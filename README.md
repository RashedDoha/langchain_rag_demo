A basic retrieval augmented generation pipeline (RAG) with a FastAPI streaming output endpoint.

It does the following functions

- Ingests data in the form of documents from the `data/raw` directory
- Chunks the documents with appropriate overlap for improved retrieval accuracy
- Stores the chunks in a vector store (Chromadb) to accomplish similarity search on a given query
- Retrieves information from the vector store given a query string
- Invokes an LLM to generate answer to the given user question by augmenting context from the retrieval process