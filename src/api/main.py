import asyncio
import logging
import logging.config
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.pipelines.index import index_documents
from rag.pipelines.serve import invoke_query
from rag.vector_store.store_documents import get_vector_store

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "structured",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Pipeline API", version="2.0.0")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None


class IndexResponse(BaseModel):
    status: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    document_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_vector_store_populated() -> None:
    """Raise 503 if the vector store has no documents."""
    try:
        store = get_vector_store()
        count = store._collection.count()
        if count == 0:
            raise HTTPException(
                status_code=503,
                detail="Vector store is empty. Index your documents first via POST /index.",
            )
    except HTTPException:
        raise
    except Exception:
        # If the check itself fails (e.g. store doesn't exist yet), let the
        # query proceed and surface a more descriptive error from the chain.
        pass


def _stream_tokens(question: str, filters: dict | None, request_id: str):
    try:
        for token in invoke_query(question, filters=filters, request_id=request_id):
            yield token
    except Exception as e:
        logger.error("Streaming error request_id=%s error=%s", request_id, str(e))
        yield f"\n\n[Error: {str(e)}]"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health and the number of indexed documents."""
    try:
        store = get_vector_store()
        count = store._collection.count()
    except Exception:
        count = -1
    return {"status": "healthy", "document_count": count}


@app.post("/query")
async def query(request: QueryRequest):
    """Stream tokens from the RAG chain for the given question.

    Optional ``filters`` are passed directly to the Chroma retriever as a
    metadata filter, e.g. ``{"source": "report_2024.pdf"}``.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    request_id = str(uuid.uuid4())
    logger.info(
        "Query received request_id=%s question=%r filters=%s",
        request_id,
        request.question,
        request.filters,
    )

    _check_vector_store_populated()

    return StreamingResponse(
        _stream_tokens(request.question, request.filters, request_id),
        media_type="text/event-stream",
        headers={"X-Request-ID": request_id},
    )


@app.post("/index", response_model=IndexResponse)
async def index(background_tasks: BackgroundTasks):
    """Trigger document indexing asynchronously.

    Returns immediately; indexing runs in the background.
    Check server logs for completion status.
    """
    request_id = str(uuid.uuid4())
    logger.info("Indexing started request_id=%s", request_id)
    background_tasks.add_task(_run_indexing, request_id)
    return {"status": "indexing_started", "request_id": request_id}


async def _run_indexing(request_id: str) -> None:
    try:
        n = await asyncio.to_thread(index_documents)
        logger.info("Indexing complete request_id=%s chunks_written=%d", request_id, n)
    except Exception as e:
        logger.error("Indexing failed request_id=%s error=%s", request_id, str(e))
