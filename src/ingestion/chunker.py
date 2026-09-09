from typing import List
import tiktoken
from src.config import get_settings

class TokenChunker:
    """
    Recursive character text splitter that strictly respects token boundaries using tiktoken.
    """
    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size_tokens
        self.chunk_overlap = settings.chunk_overlap_tokens
        # Use cl100k_base (standard for recent OpenAI models, good generic BPE)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk(self, text: str) -> List[str]:
        # Hierarchical separators
        return self._split_text(text, ["\n\n", "\n", " ", ""])

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[0]
        
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        
        current_chunk_splits = []
        current_length = 0

        for s in splits:
            l = self.count_tokens(s)
            
            # If adding this split exceeds chunk size, finalize the current chunk
            if current_length + l > self.chunk_size and current_chunk_splits:
                chunk_text = separator.join(current_chunk_splits)
                final_chunks.append(chunk_text)
                
                # Apply overlap
                while current_length > self.chunk_overlap and len(current_chunk_splits) > 1:
                    removed = current_chunk_splits.pop(0)
                    current_length -= self.count_tokens(removed)
                    
            current_chunk_splits.append(s)
            current_length += l

        if current_chunk_splits:
            final_chunks.append(separator.join(current_chunk_splits))

        # Recursively chunk any segments that are still too large
        if len(separators) > 1:
            result = []
            for chunk in final_chunks:
                if self.count_tokens(chunk) > self.chunk_size:
                    result.extend(self._split_text(chunk, separators[1:]))
                else:
                    result.append(chunk)
            return result

        return final_chunks
