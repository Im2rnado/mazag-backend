"""
Mazag FastAPI Application Entry Point

Startup sequence:
  1. Connect to MongoDB
  2. Load/build embedding model (Qwen3-Embedding-4B or fallback)
  3. Load/build FAISS knowledge index
  4. Register all routers
  5. Serve requests
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.database import connect_db, close_db
from api.services.embedder import get_embedder
from api.services.knowledge import initialize_knowledge
from api.routers import auth, chat, therapists, exercises, onboarding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("═" * 60)
    logger.info("  🧠 Mazag AI Backend — Starting up")
    logger.info("═" * 60)

    # 1. MongoDB
    await connect_db()

    # 2. Embedding model (blocks until model loaded / downloaded)
    logger.info(f"Loading embedding model: {settings.embedding_model} …")
    embedder = get_embedder()
    logger.info(f"Embedding model ready: {embedder.model_name} (dim={embedder.dim})")

    # 3. Knowledge base / FAISS
    await initialize_knowledge(
        vector_store_path=settings.vector_store_path,
        knowledge_dir=settings.knowledge_dir,
    )

    logger.info("═" * 60)
    logger.info("  ✅ Mazag backend ready — http://localhost:8000")
    logger.info("  📖 API docs        — http://localhost:8000/docs")
    logger.info("═" * 60)

    yield  # ← Application runs between here and below

    # Shutdown
    await close_db()
    logger.info("Mazag backend shut down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mazag Mental Health AI API",
    description="Backend API for Mazag — an Egyptian-aware mental health AI companion.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow everything for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(onboarding.router)
app.include_router(chat.router)
app.include_router(therapists.router)
app.include_router(exercises.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "Mazag API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    from api.services.knowledge import _is_ready as rag_ready
    from api.services.embedder import _embedder
    return {
        "status": "ok",
        "embedding_model": _embedder.model_name if _embedder else "not loaded",
        "rag_ready": rag_ready,
    }
