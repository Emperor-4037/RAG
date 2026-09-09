# Project RAG

Production-grade Retrieval-Augmented Generation pipeline engineered for **local LLM inference** on consumer hardware.

## Hardware Requirements

| Resource | Specification |
|---|---|
| **GPU** | NVIDIA RTX 4060 Laptop (8 GB VRAM) — exclusive to LLM inference |
| **CPU** | Multi-core modern CPU — runs embeddings, reranking, parsing |
| **RAM** | 16–32 GB — hosts ONNX models, vector indices, application state |

## Architecture

```
Client → FastAPI (Rate Limiting, SSE) → Semantic Cache (RedisVL)
                                           ↓ miss
                                      Dense Retrieval (Qdrant)
                                           ↓ candidates
                                      Cross-Encoder Reranker (ONNX CPU)
                                           ↓ top-N context
                                      LLM Generation (llama.cpp, GPU)
                                           ↓ SSE stream
                                        Client
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Emperor-4037/RAG.git
cd RAG

# 2. Copy environment config
cp .env.example .env

# 3. Start infrastructure
docker compose -f docker/docker-compose.yml up -d

# 4. Create virtual environment & install
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

# 5. Download models
python scripts/download_models.py

# 6. Start llama.cpp server (adjust model path)
# llama-server -m models/model.gguf -ngl 99 -c 4096 --port 8080

# 7. Run the API
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Infrastructure Services

| Service | Port | Purpose |
|---|---|---|
| **Qdrant** | 6333 (HTTP), 6334 (gRPC) | Vector database with mmap storage |
| **Redis Stack** | 6379 (Redis), 8001 (RedisInsight) | Celery broker + semantic cache |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Dashboards (admin/admin) |

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## License

MIT
