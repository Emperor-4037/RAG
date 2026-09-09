"""
Unit tests for DenseSearch.
Mocks QdrantClient to avoid requiring a live Qdrant instance.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_dense_search():
    with patch("src.retrieval.dense_search.get_settings") as mock_settings, \
         patch("src.retrieval.dense_search.QdrantClient") as mock_qdrant_cls:

        mock_settings.return_value = MagicMock(
            qdrant_url="http://localhost:6333",
            qdrant_collection_name="documents",
            retrieval_top_k=5,
        )

        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_qdrant_cls.return_value = mock_qdrant

        from src.retrieval.dense_search import DenseSearch
        searcher = DenseSearch()
        searcher.qdrant = mock_qdrant
        return searcher, mock_qdrant


class TestDenseSearch:
    def test_search_returns_list_of_strings(self, mock_dense_search):
        searcher, mock_qdrant = mock_dense_search
        mock_hit = MagicMock()
        mock_hit.payload = {"text": "relevant chunk", "source": "doc.pdf"}
        mock_qdrant.search.return_value = [mock_hit]

        results = searcher.search([0.1] * 768)
        assert isinstance(results, list)
        assert results == ["relevant chunk"]

    def test_search_empty_collection(self, mock_dense_search):
        searcher, mock_qdrant = mock_dense_search
        mock_qdrant.collection_exists.return_value = False

        results = searcher.search([0.1] * 768)
        assert results == []

    def test_search_respects_top_k(self, mock_dense_search):
        searcher, mock_qdrant = mock_dense_search
        hits = [MagicMock(payload={"text": f"chunk {i}"}) for i in range(5)]
        mock_qdrant.search.return_value = hits

        results = searcher.search([0.1] * 768, top_k=5)
        assert len(results) == 5

    def test_search_handles_missing_payload(self, mock_dense_search):
        searcher, mock_qdrant = mock_dense_search
        mock_hit = MagicMock()
        mock_hit.payload = None
        mock_qdrant.search.return_value = [mock_hit]

        results = searcher.search([0.1] * 768)
        assert results == []
