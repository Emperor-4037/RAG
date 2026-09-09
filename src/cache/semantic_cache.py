import uuid
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from src.config import get_settings

class SemanticCache:
    """
    RedisVL-backed semantic cache to short-circuit retrieval and generation
    for exact or highly-similar repeated queries.
    """
    def __init__(self):
        self.settings = get_settings()
        self.threshold = self.settings.semantic_cache_threshold
        
        schema_dict = {
            "index": {
                "name": "semantic_cache",
                "prefix": "cache:",
                "storage_type": "hash"
            },
            "fields": [
                {"name": "response", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": self.settings.embedding_dimension,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32"
                    }
                }
            ]
        }
        
        self.client = Redis.from_url(self.settings.redis_url)
        self.schema = IndexSchema.from_dict(schema_dict)
        self.index = SearchIndex(self.schema, self.client)
        self.index.create(overwrite=False)

    def check(self, query_vector: list[float]) -> str | None:
        """
        Check if a semantically similar query exists in the cache.
        Returns the cached response if similarity >= threshold, else None.
        """
        query = VectorQuery(
            vector=query_vector,
            vector_field_name="vector",
            return_fields=["response", "vector_distance"],
            num_results=1
        )
        
        results = self.index.query(query)
        if not results:
            return None
            
        hit = results[0]
        # RedisVL returns cosine distance. Similarity = 1.0 - distance
        distance = float(hit.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        
        if similarity >= self.threshold:
            return hit["response"]
            
        return None

    def store(self, query_vector: list[float], response: str):
        """
        Store a new query-response pair in the semantic cache.
        """
        self.index.load([
            {
                "id": str(uuid.uuid4()),
                "vector": query_vector,
                "response": response
            }
        ])
