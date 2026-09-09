"""
LLM streaming client for llama.cpp server.

Connects via Server-Sent Events (SSE) over HTTP using httpx + httpx-sse.
The GPU is exclusively reserved for this inference path — no CPU offload.
"""
import time
import structlog
from typing import AsyncGenerator, List
import httpx
from httpx_sse import aconnect_sse

from src.config import get_settings
from src.observability.metrics import LLM_REQUESTS_TOTAL, LLM_TTFT_SECONDS

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """You are a precise, expert assistant. Answer questions using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly. Do not fabricate information."""


def _build_prompt(query: str, context_chunks: List[str]) -> str:
    """Format the retrieved context and user query into an LLM prompt."""
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"<|system|>\n{_SYSTEM_PROMPT}\n"
        f"<|context|>\n{context}\n"
        f"<|user|>\n{query}\n"
        f"<|assistant|>"
    )


async def stream_llm_response(
    query: str,
    context_chunks: List[str],
) -> AsyncGenerator[str, None]:
    """
    Stream a response from the llama.cpp server via SSE.

    Yields text delta strings as they arrive from the LLM.
    Also records TTFT and total request status to Prometheus.
    """
    settings = get_settings()
    prompt = _build_prompt(query, context_chunks)

    payload = {
        "prompt": prompt,
        "n_predict": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        "stream": True,
        "stop": ["<|user|>", "<|system|>", "</s>"],
    }

    request_start = time.perf_counter()
    first_token_received = False

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with aconnect_sse(
                client, "POST", f"{settings.llm_base_url}/completion",
                json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.data == "[DONE]":
                        break

                    data = sse.json()
                    token = data.get("content", "")

                    if token:
                        if not first_token_received:
                            ttft = time.perf_counter() - request_start
                            LLM_TTFT_SECONDS.observe(ttft)
                            first_token_received = True
                            logger.info("llm.first_token", ttft_seconds=round(ttft, 3))

                        yield token

                        # Check llama.cpp stop signal
                        if data.get("stop", False):
                            break

        LLM_REQUESTS_TOTAL.labels(status="success").inc()

    except httpx.ConnectError:
        LLM_REQUESTS_TOTAL.labels(status="connection_error").inc()
        logger.error("llm.connection_failed", url=settings.llm_base_url)
        yield "\n\n[Error: Could not connect to LLM server. Ensure llama.cpp is running.]"

    except Exception as e:
        LLM_REQUESTS_TOTAL.labels(status="error").inc()
        logger.error("llm.stream_error", error=str(e))
        yield f"\n\n[Error: LLM generation failed — {e}]"
