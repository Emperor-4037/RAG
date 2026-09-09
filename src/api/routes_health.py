"""
Health check endpoints.
Provides liveness (/healthz) and readiness (/readyz) probes
that verify upstream dependencies are reachable.
"""
import httpx
import structlog
from fastapi import APIRouter
from redis import Redis
from qdrant_client import QdrantClient

from src.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/healthz")
def liveness():
    """Simple liveness probe — always returns OK if the process is alive."""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness():
    """
    Readiness probe — verifies connectivity to Redis, Qdrant, and the LLM server.
    Returns 200 if all healthy, 503 with a dependency map if any are down.
    """
    checks: dict[str, str] = {}

    # Redis
    try:
        r = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        logger.warning("readyz.redis_down", error=str(e))

    # Qdrant
    try:
        q = QdrantClient(url=settings.qdrant_url, timeout=2)
        q.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        logger.warning("readyz.qdrant_down", error=str(e))

    # LLM server
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.llm_base_url}/health")
            checks["llm"] = "ok" if resp.status_code == 200 else f"status: {resp.status_code}"
    except Exception as e:
        checks["llm"] = f"error: {e}"
        logger.warning("readyz.llm_down", error=str(e))

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
