"""
Ingestion pipeline — top-level entry point.

Usage:
    from backend.ingestion.pipeline import ingest_document
    doc_id = ingest_document("/path/to/report.pdf")
"""
from __future__ import annotations

import uuid
from pathlib import Path

from backend.core.logger import get_logger
from backend.ingestion.chunker import chunk_blocks
from backend.ingestion.parser import parse_pdf
from backend.ingestion.vector_store import upsert_chunks

logger = get_logger(__name__)


def ingest_document(pdf_path: str | Path) -> str:
    """
    Full ingest pipeline: parse -> chunk -> embed -> store.

    Returns:
        doc_id (str): stable ID for the document, used for retrieval.
    """
    pdf_path = Path(pdf_path)
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path.resolve())))

    logger.info("ingest_start", extra={"path": str(pdf_path), "doc_id": doc_id})

    blocks = parse_pdf(pdf_path)
    chunks = chunk_blocks(blocks, doc_id)
    upserted = upsert_chunks(chunks, doc_id)

    logger.info(
        "ingest_complete",
        extra={"doc_id": doc_id, "chunks_total": len(chunks), "upserted": upserted},
    )
    return doc_id
