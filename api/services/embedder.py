"""
Qwen3-Embedding-4B singleton embedder for the Mazag RAG system.

Falls back to all-MiniLM-L6-v2 if Qwen model fails to load (RAM-constrained environments).
The embedding model is loaded ONCE at startup and reused across requests.
"""

import numpy as np
from typing import List, Union
import threading
import logging
from api.config import get_settings

logger = logging.getLogger(__name__)

_embedder = None
_embedder_lock = threading.Lock()
_embedding_dim: int = 0


class Embedder:
    """Wraps a sentence-transformers model for embedding generation."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        # Run a dummy embed to determine actual dimension
        test = self.model.encode(["test"], convert_to_numpy=True)
        self.dim = test.shape[1]
        logger.info(f"✅ Embedding model loaded | dim={self.dim}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed one or more texts. Always returns shape (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
            show_progress_bar=False,
        )
        return vecs  # shape: (n, dim)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text, return shape (dim,)."""
        return self.embed([text])[0]


def get_embedder() -> Embedder:
    """
    Return the singleton embedder, loading it on first call.
    Thread-safe via a lock.
    """
    global _embedder, _embedding_dim
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:  # Double-checked locking
            return _embedder

        settings = get_settings()
        primary_model = settings.embedding_model
        fallback_model = "all-MiniLM-L6-v2"

        try:
            _embedder = Embedder(primary_model)
        except Exception as e:
            logger.warning(
                f"⚠️  Failed to load {primary_model}: {e}\n"
                f"Falling back to {fallback_model}"
            )
            _embedder = Embedder(fallback_model)

        _embedding_dim = _embedder.dim
        return _embedder


def get_embedding_dim() -> int:
    """Return the embedding dimension of the loaded model."""
    global _embedding_dim
    if _embedding_dim == 0:
        get_embedder()
    return _embedding_dim
