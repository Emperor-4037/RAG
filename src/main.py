from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes_ingest import router as ingest_router

app = FastAPI(
    title="Project RAG API",
    description="Local LLM Retrieval-Augmented Generation system",
    version="0.1.0",
)

# Standard CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ingest_router)

@app.get("/healthz", tags=["Health"])
def health_check():
    """Simple synchronous health check."""
    return {"status": "ok"}
