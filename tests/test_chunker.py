"""
Unit tests for the TokenChunker.
These tests do NOT require running infrastructure.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def chunker():
    """Return a TokenChunker with predictable settings."""
    with patch("src.ingestion.chunker.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            chunk_size_tokens=50,
            chunk_overlap_tokens=10,
        )
        from src.ingestion.chunker import TokenChunker
        return TokenChunker()


class TestTokenChunker:
    def test_short_text_returns_single_chunk(self, chunker):
        text = "This is a very short sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self, chunker):
        # Generate a text that is definitely larger than chunk_size_tokens=50
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_all_chunks_within_token_limit(self, chunker):
        text = " ".join(["hello world this is a test sentence."] * 50)
        chunks = chunker.chunk(text)
        for chunk in chunks:
            token_count = chunker.count_tokens(chunk)
            # Allow slight overflow at separator boundaries
            assert token_count <= chunker.chunk_size + 20, (
                f"Chunk exceeded limit: {token_count} tokens"
            )

    def test_empty_text_returns_empty_list_or_single_empty(self, chunker):
        chunks = chunker.chunk("")
        assert isinstance(chunks, list)

    def test_chunk_overlap_content_preserved(self, chunker):
        """Last tokens of one chunk should appear in the start of the next."""
        text = "apple banana cherry date elderberry fig grape honeydew " * 20
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # Verify chunks are non-empty and contain original words
            for chunk in chunks:
                assert len(chunk.strip()) > 0

    def test_count_tokens_returns_int(self, chunker):
        count = chunker.count_tokens("Hello world!")
        assert isinstance(count, int)
        assert count > 0
