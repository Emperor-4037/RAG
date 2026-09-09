"""
Unit tests for ONNXEmbedder.
Uses mocking to avoid requiring the ONNX model file on disk.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_embedder():
    """Return a fully mocked ONNXEmbedder that simulates ONNX output."""
    with patch("src.embeddings.onnx_embedder.get_settings") as mock_settings, \
         patch("src.embeddings.onnx_embedder.Tokenizer") as mock_tokenizer_cls, \
         patch("src.embeddings.onnx_embedder.ort.InferenceSession") as mock_session_cls, \
         patch("pathlib.Path.exists", return_value=True):

        mock_settings.return_value = MagicMock(
            embedding_model_name="Xenova/bge-base-en-v1.5",
            embedding_dimension=768,
        )

        # Mock tokenizer encoding
        mock_enc = MagicMock()
        mock_enc.ids = [101, 102, 103] + [0] * 509
        mock_enc.attention_mask = [1, 1, 1] + [0] * 509
        mock_enc.type_ids = [0] * 512
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode_batch.return_value = [mock_enc]
        mock_tokenizer_cls.from_file.return_value = mock_tokenizer

        # Mock ONNX session: outputs a deterministic hidden state
        batch_size = 1
        hidden_size = 768
        mock_session = MagicMock()
        fake_hidden = np.random.rand(batch_size, 512, hidden_size).astype(np.float32)
        mock_session.run.return_value = [fake_hidden]
        mock_session_cls.return_value = mock_session

        from src.embeddings.onnx_embedder import ONNXEmbedder
        return ONNXEmbedder()


class TestONNXEmbedder:
    def test_embed_returns_numpy_array(self, mock_embedder):
        result = mock_embedder.embed(["Hello world"])
        assert isinstance(result, np.ndarray)

    def test_embed_correct_shape(self, mock_embedder):
        result = mock_embedder.embed(["Hello world"])
        assert result.shape == (1, 768)

    def test_embed_is_l2_normalized(self, mock_embedder):
        result = mock_embedder.embed(["Hello world"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_batch(self, mock_embedder):
        """Ensure batch input is accepted without error."""
        # Re-mock for batch of 3
        mock_embedder.session.run.return_value = [
            np.random.rand(3, 512, 768).astype(np.float32)
        ]
        mock_embedder.tokenizer.encode_batch.return_value = [
            MagicMock(ids=[0]*512, attention_mask=[1]*512, type_ids=[0]*512)
        ] * 3
        # Just check it doesn't raise
        result = mock_embedder.embed(["a", "b", "c"])
        assert result is not None
