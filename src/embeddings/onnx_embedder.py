import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
from typing import List
from src.config import get_settings

class ONNXEmbedder:
    """
    CPU-optimized ONNX runtime for dense embedding models.
    Avoids PyTorch dependencies and operates strictly within RAM footprint constraints.
    """
    def __init__(self):
        self.settings = get_settings()
        self.model_dir = Path("models") / self.settings.embedding_model_name.replace("/", "_")
        
        if not self.model_dir.exists():
            raise RuntimeError(
                f"Model directory {self.model_dir} not found. "
                "Run `python scripts/download_models.py` first."
            )

        # Initialize Tokenizer (HuggingFace format)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)

        # Initialize ONNX CPU execution session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Optimize for multi-core modern CPU
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of strings, applying CLS pooling and L2 normalization.
        """
        encodings = self.tokenizer.encode_batch(texts)
        
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        # Run inference
        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]

        # BAAI/bge models use [CLS] pooling (first token)
        cls_embeddings = last_hidden_state[:, 0, :]

        # L2 Normalize
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        embeddings = cls_embeddings / norms
        
        return embeddings
