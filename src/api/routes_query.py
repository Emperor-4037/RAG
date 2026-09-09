"""
Query endpoint: integrates retrieval pipeline with LLM SSE streaming.
Rate-limited by slowapi.
"""
import time
import json
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.llm_client import stream_llm_response
from src.utils.rate_limiter import limiter
from src.config import get_settings
from src.observability.metrics import (
    SEMANTIC_CACHE_HITS_TOTAL,
    SEMANTIC_CACHE_MISSES_TOTAL,
    RETRIEVAL_DURATION_SECONDS,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

# Module-level singleton — initialized on first request after lifespan startup
_retrieval_pipeline: RetrievalPipeline | None = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


class QueryRequest(BaseModel):
    query: str
    stream: bool = True


@router.post("/")
@limiter.limit(get_settings().rate_limit_query)
async def handle_query(request: Request, body: QueryRequest):
    """
    Accept a user query, run retrieval pipeline, and stream the LLM response via SSE.
    Returns a cached response immediately on a cache hit (no LLM call).
    """
    pipeline = get_retrieval_pipeline()
    settings = get_settings()

    t0 = time.perf_counter()
    is_cache_hit, result = pipeline.retrieve(body.query)
    retrieval_latency = time.perf_counter() - t0
    RETRIEVAL_DURATION_SECONDS.observe(retrieval_latency)

    if is_cache_hit:
        SEMANTIC_CACHE_HITS_TOTAL.inc()
        logger.info(
            "query.cache_hit",
            query=body.query[:80],
            retrieval_latency=round(retrieval_latency, 3),
        )
        # Return cached response as a complete SSE stream
        cached_text: str = result  # type: ignore[assignment]

        async def _cached_stream():
            data = json.dumps({"content": cached_text, "cached": True})
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    # Cache miss — proceed to retrieval + LLM
    SEMANTIC_CACHE_MISSES_TOTAL.inc()
    context_chunks: list = result  # type: ignore[assignment]

    logger.info(
        "query.retrieval_complete",
        query=body.query[:80],
        num_chunks=len(context_chunks),
        retrieval_latency=round(retrieval_latency, 3),
    )

    if not context_chunks:
        async def _no_context_stream():
            msg = json.dumps({"content": "No relevant context found for your query.", "cached": False})
            yield f"data: {msg}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_context_stream(), media_type="text/event-stream")

    # Collect full response to store in cache after streaming
    collected_response: list[str] = []

    async def _sse_stream():
        async for token in stream_llm_response(body.query, context_chunks):
            collected_response.append(token)
            data = json.dumps({"content": token, "cached": False})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

        # Store to cache after full generation
        full_response = "".join(collected_response)
        if full_response.strip():
            pipeline.cache_response(body.query, full_response)

    return StreamingResponse(_sse_stream(), media_type="text/event-stream")
