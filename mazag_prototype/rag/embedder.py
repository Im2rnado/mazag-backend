"""
Embedding Module for Mazag RAG System
Converts text chunks into dense vector embeddings for semantic search.
Supports multiple embedding models.
"""

from typing import List, Union, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed text(s) into vector(s)"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings"""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers for local embedding generation.
    Recommended models:
    - 'all-MiniLM-L6-v2': Fast, 384-dim, good for English
    - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual, 384-dim
    - 'all-mpnet-base-v2': High quality, 768-dim
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Hugging Face model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class GeminiEmbedder(BaseEmbedder):
    """
    Uses Google Gemini API for embeddings.
    Supports: text-embedding-004 (768-dim)
    """
    
    def __init__(
        self, 
        api_key: str,
        model: str = "text-embedding-004"
    ):
        """
        Args:
            api_key: Google AI API key
            model: Embedding model name
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self._embedding_dim = 768  # text-embedding-004 dimension
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Gemini API"""
        import google.generativeai as genai
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        
        return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class OpenAICompatibleEmbedder(BaseEmbedder):
    """
    Uses OpenAI-compatible API for embeddings.
    Can work with OpenAI, or Gemini via OpenAI compatibility.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        embedding_dim: int = 1536
    ):
        """
        Args:
            api_key: API key
            model: Model name
            base_url: Optional base URL for API
            embedding_dim: Dimension of embeddings
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        
        self.model_name = model
        self._embedding_dim = embedding_dim
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI-compatible API"""
        if isinstance(texts, str):
            texts = [texts]
        
        # OpenAI API accepts batches
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class TFIDFEmbedder(BaseEmbedder):
    """
    Simple TF-IDF based embeddings as a baseline.
    Good for quick prototyping and comparison.
    """
    
    def __init__(self, max_features: int = 512):
        """
        Args:
            max_features: Maximum number of features (vocabulary size)
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )
        
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._is_fitted = False
        self._embedding_dim = max_features
    
    def fit(self, texts: List[str]):
        """Fit the vectorizer on a corpus"""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        self._embedding_dim = len(self.vectorizer.get_feature_names_out())
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate TF-IDF embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self._is_fitted:
            # Auto-fit on first use
            self.fit(texts)
        
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self._embedding_dim


# Factory function for easy embedder creation
def create_embedder(
    embedder_type: str = "sentence-transformer",
    **kwargs
) -> BaseEmbedder:
    """
    Factory function to create embedders.
    
    Args:
        embedder_type: Type of embedder to create
            - 'sentence-transformer': Local sentence transformers
            - 'gemini': Google Gemini embeddings
            - 'openai': OpenAI or compatible API
            - 'tfidf': Simple TF-IDF baseline
        **kwargs: Arguments passed to embedder constructor
        
    Returns:
        Initialized embedder instance
    """
    embedders = {
        "sentence-transformer": SentenceTransformerEmbedder,
        "gemini": GeminiEmbedder,
        "openai": OpenAICompatibleEmbedder,
        "tfidf": TFIDFEmbedder
    }
    
    if embedder_type not in embedders:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            f"Choose from: {list(embedders.keys())}"
        )
    
    return embedders[embedder_type](**kwargs)


# Utility functions
def embed_texts(
    texts: Union[str, List[str]],
    embedder_type: str = "sentence-transformer",
    **kwargs
) -> np.ndarray:
    """
    Quick utility to embed texts.
    
    Args:
        texts: Text(s) to embed
        embedder_type: Type of embedder
        **kwargs: Embedder arguments
        
    Returns:
        Embeddings array
    """
    embedder = create_embedder(embedder_type, **kwargs)
    return embedder.embed(texts)

