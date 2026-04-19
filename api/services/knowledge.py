"""
RAG Knowledge Service for Mazag.

On startup:
  - If a saved FAISS index exists → load it (fast)
  - Else → chunk all .txt files in knowledge dir, embed, save FAISS index

At query time:
  - Embed user message → search FAISS → return top-k relevant passages
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_vector_store = None   # FAISSVectorStore instance
_is_ready = False


# ── Text chunker (reused from prototype) ──────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 512, overlap_words: int = 20) -> List[str]:
    """Split text into overlapping chunks by sentence/paragraph."""
    import re
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Large paragraph → split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) > chunk_size and current:
                    chunks.append(current.strip())
                    # keep overlap words
                    words = current.split()
                    current = " ".join(words[-overlap_words:]) + " " + sent
                else:
                    current = (current + " " + sent).strip()
            if current.strip():
                chunks.append(current.strip())
                current = ""
        elif len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            words = current.split()
            current = " ".join(words[-overlap_words:]) + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]  # filter tiny chunks


# ── Simple in-memory FAISS wrapper ────────────────────────────────────────────

class FAISSStore:
    """Thin FAISS wrapper that also stores the raw texts."""

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # Inner product → cosine for L2-normed vecs
        self.texts: List[str] = []
        self.metadata: List[dict] = []

    def add_batch(self, texts: List[str], embeddings: np.ndarray, metadata: Optional[List[dict]] = None):
        if metadata is None:
            metadata = [{}] * len(texts)
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[dict]:
        k = min(k, len(self.texts))
        if k == 0:
            return []
        q = query_vec.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })
        return results

    def size(self) -> int:
        return len(self.texts)

    def save(self, path: str):
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "FAISSStore":
        import faiss
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.index = index
        store.texts = data["texts"]
        store.metadata = data["metadata"]
        return store


# ── Public API ────────────────────────────────────────────────────────────────

def _index_exists(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "index.faiss"))
        and os.path.exists(os.path.join(path, "data.pkl"))
    )


async def initialize_knowledge(vector_store_path: str, knowledge_dir: str) -> None:
    """
    Called at app startup. Loads or builds the FAISS knowledge index.
    Runs embedder.get_embedder() which triggers model download on first run.
    """
    global _vector_store, _is_ready
    from api.services.embedder import get_embedder

    if _index_exists(vector_store_path):
        logger.info("📦 Loading existing FAISS index from disk …")
        _vector_store = FAISSStore.load(vector_store_path)
        logger.info(f"✅ FAISS loaded | {_vector_store.size()} chunks")
        _is_ready = True
        return

    logger.info("🔨 Building FAISS index from knowledge documents …")
    embedder = get_embedder()

    # Gather all .txt files
    knowledge_path = Path(knowledge_dir)
    txt_files = list(knowledge_path.glob("*.txt"))
    if not txt_files:
        logger.warning(f"⚠️  No .txt files found in {knowledge_dir}")
        _vector_store = FAISSStore(dim=embedder.dim)
        _is_ready = True
        return

    all_chunks: List[str] = []
    all_meta: List[dict] = []

    for fpath in txt_files:
        text = fpath.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_meta.append({"source": fpath.name})

    logger.info(f"  Embedding {len(all_chunks)} chunks …")
    # Embed in batches of 32 to avoid OOM
    batch_size = 32
    embeddings_list = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vecs = embedder.embed(batch)
        embeddings_list.append(vecs)
    embeddings = np.vstack(embeddings_list)

    _vector_store = FAISSStore(dim=embedder.dim)
    _vector_store.add_batch(all_chunks, embeddings, all_meta)

    logger.info(f"  Saving FAISS index to {vector_store_path} …")
    _vector_store.save(vector_store_path)

    logger.info(f"✅ FAISS index built | {_vector_store.size()} chunks from {len(txt_files)} documents")
    _is_ready = True


def retrieve_context(query: str, top_k: int = 3) -> Optional[str]:
    """
    Embed the query and retrieve relevant passages.
    Returns a single concatenated context string, or None if nothing useful found.
    """
    global _vector_store, _is_ready
    if not _is_ready or _vector_store is None or _vector_store.size() == 0:
        return None

    from api.services.embedder import get_embedder
    embedder = get_embedder()
    query_vec = embedder.embed_one(query)

    results = _vector_store.search(query_vec, k=top_k)
    if not results:
        return None

    # Only use results with reasonable similarity (cosine > 0.3 for L2-normed vecs)
    relevant = [r for r in results if r["score"] > 0.3]
    if not relevant:
        return None

    context_parts = [r["text"] for r in relevant]
    return "\n\n---\n\n".join(context_parts)
