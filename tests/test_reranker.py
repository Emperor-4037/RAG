"""
Unit tests for ONNXReranker.
Uses mocking to avoid requiring the ONNX reranker model on disk.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_reranker():
    with patch("src.retrieval.reranker.get_settings") as mock_settings, \
         patch("src.retrieval.reranker.Tokenizer") as mock_tokenizer_cls, \
         patch("src.retrieval.reranker.ort.InferenceSession") as mock_session_cls, \
         patch("pathlib.Path.exists", return_value=True):

        mock_settings.return_value = MagicMock(
            reranker_model_name="Xenova/ms-marco-MiniLM-L-6-v2",
            reranker_top_n=3,
        )

        mock_enc = MagicMock()
        mock_enc.ids = [0] * 512
        mock_enc.attention_mask = [1] * 512
        mock_enc.type_ids = [0] * 512
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        from src.retrieval.reranker import ONNXReranker
        return ONNXReranker(), mock_tokenizer, mock_session


class TestONNXReranker:
    def test_rerank_empty_candidates_returns_empty(self, mock_reranker):
        reranker, _, _ = mock_reranker
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_returns_sorted_descending(self, mock_reranker):
        reranker, mock_tokenizer, mock_session = mock_reranker
        candidates = ["low relevance doc", "high relevance doc", "medium relevance doc"]
        scores = np.array([[0.1], [0.9], [0.5]], dtype=np.float32)
        mock_session.run.return_value = [scores]
        mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[0]*512, attention_mask=[1]*512, type_ids=[0]*512)
        ] * 3

        results = reranker.rerank("query", candidates)
        assert len(results) > 0
        returned_scores = [score for _, score in results]
        # Verify descending order
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_rerank_respects_top_n(self, mock_reranker):
        reranker, mock_tokenizer, mock_session = mock_reranker
        candidates = [f"doc {i}" for i in range(10)]
        scores = np.array([[float(i)] for i in range(10)], dtype=np.float32)
        mock_session.run.return_value = [scores]
        mock_tokenizer.encode_batch.return_value = [
            MagicMock(ids=[0]*512, attention_mask=[1]*512, type_ids=[0]*512)
        ] * 10

        results = reranker.rerank("query", candidates)
        # reranker_top_n is mocked to 3
        assert len(results) == 3
