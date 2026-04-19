"""
MongoDB connection using Motor (async driver).
Provides a shared database client for the FastAPI app.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from api.config import get_settings
from typing import Optional

_client: Optional[AsyncIOMotorClient] = None


async def connect_db() -> None:
    """Create and store the MongoDB client."""
    global _client
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_url)
    # Verify connection
    await _client.admin.command("ping")
    print(f"✅ MongoDB connected: {settings.mongodb_url} / {settings.mongodb_db_name}")


async def close_db() -> None:
    """Close the MongoDB client."""
    global _client
    if _client:
        _client.close()
        print("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Return the Mazag database instance."""
    settings = get_settings()
    if _client is None:
        raise RuntimeError("Database not connected. Call connect_db() first.")
    return _client[settings.mongodb_db_name]
