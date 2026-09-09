"""
Prometheus application metrics.

Counters and histograms covering:
- HTTP request count and latency (per route and status)
- Semantic cache hits/misses
- Ingestion throughput
- Retrieval and reranking latency
- LLM Time-To-First-Token (TTFT)
"""
from prometheus_client import Counter, Histogram, Gauge

# --- HTTP Layer ---
HTTP_REQUESTS_TOTAL = Counter(
    "rag_http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# --- Ingestion ---
INGEST_DOCUMENTS_TOTAL = Counter(
    "rag_ingest_documents_total",
    "Total documents submitted for ingestion",
    ["status"],
)

INGEST_CHUNKS_TOTAL = Counter(
    "rag_ingest_chunks_total",
    "Total document chunks embedded and indexed",
)

# --- Retrieval ---
SEMANTIC_CACHE_HITS_TOTAL = Counter(
    "rag_semantic_cache_hits_total",
    "Total semantic cache hits (responses served without retrieval)",
)

SEMANTIC_CACHE_MISSES_TOTAL = Counter(
    "rag_semantic_cache_misses_total",
    "Total semantic cache misses (proceeded to full retrieval)",
)

RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "End-to-end retrieval + reranking latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RERANKER_DURATION_SECONDS = Histogram(
    "rag_reranker_duration_seconds",
    "Cross-encoder reranking latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5],
)

# --- LLM Generation ---
LLM_REQUESTS_TOTAL = Counter(
    "rag_llm_requests_total",
    "Total requests forwarded to the LLM server",
    ["status"],
)

LLM_TTFT_SECONDS = Histogram(
    "rag_llm_time_to_first_token_seconds",
    "LLM Time-To-First-Token (TTFT) latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
