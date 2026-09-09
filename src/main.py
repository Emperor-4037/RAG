"""
FastAPI application factory.

Lifecycle:
  startup  → configure logging, configure OTel tracing + metrics, attach rate limiter
  shutdown → no-op (Docker handles process cleanup)

Routes:
  /ingest     — async document ingestion (Celery-backed)
  /query      — SSE LLM query with semantic cache + retrieval
  /healthz    — liveness probe
  /readyz     — readiness probe (checks Redis, Qdrant, LLM)
  /metrics    — Prometheus scrape endpoint
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.config import get_settings
from src.observability.logging import configure_logging
from src.observability.tracing import configure_tracing, configure_metrics, instrument_fastapi
from src.utils.rate_limiter import limiter
from src.api.routes_ingest import router as ingest_router
from src.api.routes_query import router as query_router
from src.api.routes_health import router as health_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize all resources on startup."""
    settings = get_settings()

    # 1. Structured logging
    configure_logging(settings.log_level)

    # 2. OpenTelemetry tracing + metrics
    configure_tracing(settings.otel_service_name)
    configure_metrics()

    logger.info(
        "application.startup",
        service=settings.otel_service_name,
        llm_url=settings.llm_base_url,
        qdrant_url=settings.qdrant_url,
    )

    yield

    logger.info("application.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Project RAG",
        description="Local LLM Retrieval-Augmented Generation pipeline (RTX 4060, 8 GB VRAM)",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiter middleware (slowapi)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # --- Routers ---
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(health_router)

    # --- Prometheus /metrics scrape endpoint ---
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # --- OTel FastAPI auto-instrumentation ---
    instrument_fastapi(app)

    return app


# Module-level app instance for uvicorn
app = create_app()
