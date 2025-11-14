"""
Vector Store Module for Mazag RAG System
Stores and retrieves embeddings using FAISS for efficient similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pickle
import os
from dataclasses import dataclass, asdict


@dataclass
class StoredChunk:
    """Represents a stored chunk with its embedding"""
    chunk_id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    Supports multiple index types and similarity metrics.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'ip')
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Store metadata and texts separately
        self.chunks: List[StoredChunk] = []
        self.id_to_idx: Dict[str, int] = {}
    
    def _create_index(self):
        """Create FAISS index based on type and metric"""
        import faiss
        
        # Choose base index based on metric
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity (then use IP)
            self._normalize = True
            base_index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            self._normalize = False
            base_index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.metric == "ip":  # Inner product
            self._normalize = False
            base_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Wrap with index type
        if self.index_type == "flat":
            return base_index
        elif self.index_type == "ivf":
            # IVF with 100 clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            # HNSW with 32 connections
            return faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        if self._normalize:
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
            return embedding / (norm + 1e-8)
        return embedding
    
    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a single chunk to the vector store.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Original text
            embedding: Vector embedding
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}
        
        # Normalize if using cosine similarity
        embedding = self._normalize_embedding(embedding)
        
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(embedding.astype('float32'))
        
        # Store chunk info
        idx = len(self.chunks)
        chunk = StoredChunk(
            chunk_id=chunk_id,
            text=text,
            embedding=embedding[0],
            metadata=metadata
        )
        self.chunks.append(chunk)
        self.id_to_idx[chunk_id] = idx
    
    def add_batch(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add multiple chunks in batch (more efficient).
        
        Args:
            chunk_ids: List of chunk IDs
            texts: List of texts
            embeddings: Array of embeddings (n_chunks, embedding_dim)
            metadata_list: List of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{} for _ in chunk_ids]
        
        # Normalize embeddings
        embeddings = self._normalize_embedding(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunk info
        start_idx = len(self.chunks)
        for i, (chunk_id, text, embedding, metadata) in enumerate(
            zip(chunk_ids, texts, embeddings, metadata_list)
        ):
            chunk = StoredChunk(
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                metadata=metadata
            )
            self.chunks.append(chunk)
            self.id_to_idx[chunk_id] = start_idx + i
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dicts with chunk info and optionally scores
        """
        # Normalize query
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, len(self.chunks))  # Don't search for more than we have
        if k == 0:
            return []
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        # Retrieve results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            chunk = self.chunks[idx]
            result = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            
            if return_scores:
                # Convert distance to similarity score
                if self.metric == "cosine" or self.metric == "ip":
                    result["score"] = float(dist)
                elif self.metric == "l2":
                    # Convert L2 distance to similarity (0-1 range)
                    result["score"] = 1.0 / (1.0 + float(dist))
            
            results.append(result)
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID"""
        if chunk_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[chunk_id]
        chunk = self.chunks[idx]
        
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata
        }
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk (marks as deleted, doesn't remove from index).
        Returns True if deleted, False if not found.
        """
        if chunk_id not in self.id_to_idx:
            return False
        
        idx = self.id_to_idx[chunk_id]
        # Mark as deleted in metadata
        self.chunks[idx].metadata["_deleted"] = True
        return True
    
    def save(self, directory: str):
        """
        Save vector store to disk.
        
        Args:
            directory: Directory to save files
        """
        import faiss
        
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save chunks and metadata
        chunks_path = os.path.join(directory, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "id_to_idx": self.id_to_idx,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "metric": self.metric
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load vector store from disk.
        
        Args:
            directory: Directory containing saved files
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        import faiss
        
        # Load FAISS index
        index_path = os.path.join(directory, "index.faiss")
        index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        chunks_path = os.path.join(directory, "chunks.pkl")
        with open(chunks_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct store
        store = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"],
            metric=data["metric"]
        )
        store.index = index
        store.chunks = data["chunks"]
        store.id_to_idx = data["id_to_idx"]
        
        return store
    
    def size(self) -> int:
        """Return number of chunks in store"""
        return len(self.chunks)
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks (excluding deleted)"""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in self.chunks
            if not chunk.metadata.get("_deleted", False)
        ]


# Simple in-memory store for smaller datasets
class SimpleVectorStore:
    """
    Simple in-memory vector store using numpy.
    Good for small datasets or quick prototyping.
    """
    
    def __init__(self, metric: str = "cosine"):
        """
        Args:
            metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.metric = metric
        self.chunks: List[StoredChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
    
    def add(self, chunk_id: str, text: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a chunk"""
        if metadata is None:
            metadata = {}
        
        idx = len(self.chunks)
        chunk = StoredChunk(chunk_id, text, embedding, metadata)
        self.chunks.append(chunk)
        self.id_to_idx[chunk_id] = idx
        
        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        if self.metric == "cosine":
            # Cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(doc_norms, query_norm.T).flatten()
        elif self.metric == "dot":
            # Dot product
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        elif self.metric == "euclidean":
            # Negative euclidean distance (higher is more similar)
            distances = np.linalg.norm(self.embeddings - query_embedding, axis=1)
            similarities = -distances
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Get top k
        k = min(k, len(self.chunks))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(similarities[idx])
            })
        
        return results
    
    def size(self) -> int:
        return len(self.chunks)

