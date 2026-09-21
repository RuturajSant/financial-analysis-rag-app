"""
Checkpointer factory — returns the right checkpointer for the environment.

Design (from references/orchestration.md):
- Local dev: InMemorySaver (lost on restart, fine for iteration).
- Deployed (EC2): SqliteSaver file-backed on the container volume.
- Always pass a consistent thread_id on every graph.invoke() call — a missing/
  inconsistent thread_id is the most common cause of "graph forgot everything" bugs.
"""
from __future__ import annotations

import os

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


def get_checkpointer():
    """
    Return the appropriate LangGraph checkpointer.

    Returns None when session persistence is disabled (stateless mode).
    Set env var SQLITE_CHECKPOINT_PATH to use SQLite (recommended for deployment).
    Leave unset or set to "memory" to use the in-memory saver.
    """
    if not settings.enable_session_persistence:
        logger.info("checkpointer_disabled")
        return None

    path = settings.sqlite_checkpoint_path
    use_sqlite = path and path.lower() != "memory"

    if use_sqlite:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(path)
            logger.info("checkpointer_sqlite", extra={"path": path})
            return checkpointer
        except ImportError:
            logger.warning(
                "sqlite_checkpointer_unavailable",
                extra={"reason": "langgraph[sqlite] not installed, falling back to memory"},
            )

    from langgraph.checkpoint.memory import MemorySaver

    logger.info("checkpointer_memory")
    return MemorySaver()
