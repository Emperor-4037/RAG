import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
from typing import List, Tuple
from src.config import get_settings

class ONNXReranker:
    """
    CPU-optimized ONNX runtime for cross-encoder reranking models.
    Scores query-document pairs to distill context before LLM generation.
    """
    def __init__(self):
        self.settings = get_settings()
        self.model_dir = Path("models") / self.settings.reranker_model_name.replace("/", "_")
        
        if not self.model_dir.exists():
            raise RuntimeError(
                f"Reranker model directory {self.model_dir} not found. "
                "Run `python scripts/download_models.py` first."
            )

        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Score and sort candidate chunks against the query using the cross-encoder.
        Returns the top-N candidates sorted by score descending.
        """
        if not candidates:
            return []
            
        # Cross-encoders expect pairs of (query, document)
        pairs = [(query, doc) for doc in candidates]
        encodings = self.tokenizer.encode_batch(pairs)
        
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        outputs = self.session.run(None, ort_inputs)
        # outputs[0] contains the logits. Flatten to 1D array.
        scores = outputs[0].flatten()
        
        scored_candidates = list(zip(candidates, scores))
        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:self.settings.reranker_top_n]
