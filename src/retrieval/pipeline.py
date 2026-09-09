from typing import List, Tuple, Union
from src.config import get_settings
from src.embeddings.onnx_embedder import ONNXEmbedder
from src.retrieval.dense_search import DenseSearch
from src.retrieval.reranker import ONNXReranker
from src.cache.semantic_cache import SemanticCache

class RetrievalPipeline:
    """
    End-to-end orchestration of the retrieval phase:
    1. Check Semantic Cache
    2. Dense Vector Search (Candidates)
    3. Cross-Encoder Reranking (Top-N)
    """
    def __init__(self):
        self.settings = get_settings()
        # Initialize heavy components once
        self.embedder = ONNXEmbedder()
        self.dense_search = DenseSearch()
        self.reranker = ONNXReranker()
        self.cache = SemanticCache()

    def retrieve(self, query: str) -> Tuple[bool, Union[str, List[str]]]:
        """
        Process a query through the retrieval pipeline.
        Returns:
            (is_cache_hit: bool, result: str | List[str])
            If is_cache_hit is True, result is the cached string response.
            If is_cache_hit is False, result is a list of top-N reranked context chunks.
        """
        # 1. Embed Query
        query_vector = self.embedder.embed([query])[0].tolist()

        # 2. Semantic Cache Check
        cached_response = self.cache.check(query_vector)
        if cached_response:
            return True, cached_response

        # 3. Dense Retrieval Candidate Gathering
        candidates = self.dense_search.search(query_vector, top_k=self.settings.retrieval_top_k)
        if not candidates:
            return False, []

        # 4. Cross-Encoder Reranking
        reranked_pairs = self.reranker.rerank(query, candidates)
        top_contexts = [doc for doc, score in reranked_pairs]

        return False, top_contexts

    def cache_response(self, query: str, response: str):
        """
        Store a successful generated response in the semantic cache.
        """
        query_vector = self.embedder.embed([query])[0].tolist()
        self.cache.store(query_vector, response)
