from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.pipelines.serve import invoke_query

app = FastAPI(title="RAG Pipeline API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str


class HealthResponse(BaseModel):
    status: str


def llm_response_streamer(question: str):
    """Stream tokens from the RAG chain."""
    try:
        for token in invoke_query(question):
            yield token
    except Exception as e:
        yield f"Error: {str(e)}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Query the RAG pipeline and stream the response.

    Returns a streaming response with tokens from the LLM.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        llm_response_streamer(request.question),
        media_type="text/event-stream"
    )