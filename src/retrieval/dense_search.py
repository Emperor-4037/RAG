from typing import List
from qdrant_client import QdrantClient
from src.config import get_settings

class DenseSearch:
    """
    Dense vector retrieval from Qdrant.
    """
    def __init__(self):
        self.settings = get_settings()
        self.qdrant = QdrantClient(url=self.settings.qdrant_url)
        self.collection_name = self.settings.qdrant_collection_name

    def search(self, query_vector: list[float], top_k: int = None) -> List[str]:
        """
        Fetch top-K most similar context chunks from Qdrant.
        """
        if top_k is None:
            top_k = self.settings.retrieval_top_k
            
        if not self.qdrant.collection_exists(self.collection_name):
            return []

        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        # Extract the original chunk text from the payload
        return [hit.payload.get("text", "") for hit in results if hit.payload]
