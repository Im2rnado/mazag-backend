"""
Mazag Backend Configuration
Reads from .env file using pydantic-settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenRouter / LLM
    openrouter_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b:free"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 500

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "mazag"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG / FAISS
    vector_store_path: str = "./api/data/vector_store"
    knowledge_dir: str = "./api/data/knowledge"
    rag_top_k: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
