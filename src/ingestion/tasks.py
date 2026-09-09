import uuid
import logging
from pathlib import Path
from celery import shared_task
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from src.config import get_settings
from src.ingestion.celery_app import celery_app
from src.ingestion.parser import parse_document
from src.ingestion.chunker import TokenChunker
from src.embeddings.onnx_embedder import ONNXEmbedder

logger = logging.getLogger(__name__)

# Lazy initialization globals to reuse heavy resources across task runs
embedder = None
chunker = None
qdrant = None

def _initialize_worker_resources():
    """Initialize ML models and DB clients once per worker process."""
    global embedder, chunker, qdrant
    settings = get_settings()

    if chunker is None:
        chunker = TokenChunker()
    if embedder is None:
        embedder = ONNXEmbedder()
    if qdrant is None:
        qdrant = QdrantClient(url=settings.qdrant_url)
        # Ensure collection exists
        if not qdrant.collection_exists(settings.qdrant_collection_name):
            qdrant.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )

@celery_app.task(bind=True)
def process_document_task(self, file_path_str: str, original_filename: str):
    """
    Celery task: Parse document -> Chunk text -> Compute embeddings -> Upsert to Qdrant.
    """
    _initialize_worker_resources()
    settings = get_settings()
    file_path = Path(file_path_str)

    try:
        # 1. Parse
        self.update_state(state="PROGRESS", meta={"stage": "parsing"})
        text = parse_document(file_path)
        
        # 2. Chunk
        self.update_state(state="PROGRESS", meta={"stage": "chunking"})
        chunks = chunker.chunk(text)
        
        if not chunks:
            return {"status": "skipped", "reason": "No text extracted"}

        # 3. Embed (in batches)
        self.update_state(state="PROGRESS", meta={"stage": "embedding", "total_chunks": len(chunks)})
        batch_size = settings.embedding_batch_size
        points = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = embedder.embed(batch_chunks)
            
            for j, emb in enumerate(embeddings):
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload={
                            "text": batch_chunks[j],
                            "source": original_filename,
                            "chunk_index": i + j
                        }
                    )
                )

        # 4. Upsert to Vector DB
        self.update_state(state="PROGRESS", meta={"stage": "indexing", "total_chunks": len(points)})
        qdrant.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points
        )
        
        # Cleanup temp file
        file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "chunks_processed": len(points),
            "source": original_filename
        }

    except Exception as e:
        logger.error(f"Failed to process document {original_filename}: {str(e)}")
        raise
