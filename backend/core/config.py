"""
Typed configuration — all settings come from environment variables with sensible defaults.
Load with: from backend.core.config import settings
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    # LLM provider: "openai" | "anthropic" | "google"
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # Provider API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Model names
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

    # Vector store
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Memory
    enable_session_persistence: bool = os.getenv("ENABLE_SESSION_PERSISTENCE", "true").lower() == "true"
    sqlite_checkpoint_path: str = os.getenv("SQLITE_CHECKPOINT_PATH", "./checkpoints.db")

    # Logging
    log_file: str = os.getenv("LOG_FILE", "./logs/app.log")

    # Retrieval
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    retrieval_rerank_top_n: int = int(os.getenv("RETRIEVAL_RERANK_TOP_N", "5"))

    # Chunking
    narrative_chunk_size: int = int(os.getenv("NARRATIVE_CHUNK_SIZE", "500"))
    narrative_chunk_overlap: int = int(os.getenv("NARRATIVE_CHUNK_OVERLAP", "50"))


settings = Settings()
