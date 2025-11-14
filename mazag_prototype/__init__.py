"""
Mazag - AI-Powered Mental Health Companion
A comprehensive backend engine for mental health chatbot with RAG, guardrails, and recommendations.
"""

__version__ = "0.1.0"
__author__ = "Mazag Team"

from .main import MazagEngine, MazagConfig, create_mazag_engine, chat_with_mazag_full

__all__ = [
    "MazagEngine",
    "MazagConfig",
    "create_mazag_engine",
    "chat_with_mazag_full"
]

