"""
Table-aware chunker for financial documents.

Design (from references/chunking.md):
- Tables are atomic — never split a table across chunk boundaries.
- Narrative text uses RecursiveCharacterTextSplitter with paragraph-preferred boundaries.
- Every chunk gets ChunkMetadata with content_hash for change-detection.
- Section title is embedded INTO the chunk text as a prefix for semantic retrieval.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Literal

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ingestion.parser import ParsedBlock

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A ready-to-embed chunk with full metadata."""
    chunk_id: str
    doc_id: str
    text: str                               # text to embed (includes section_title prefix)
    page_number: int
    section_type: Literal["narrative", "table", "footnote"]
    section_title: str | None
    section_id: str                         # groups siblings from the same logical section
    table_id: str | None
    content_hash: str


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_section_id(doc_id: str, section_title: str | None, page_number: int) -> str:
    """Stable ID grouping chunks from the same logical section."""
    key = f"{doc_id}::{section_title or 'unknown'}::{page_number}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _prefix_text(text: str, section_title: str | None) -> str:
    """Embed section title into chunk text for semantic retrieval (from chunking.md)."""
    if section_title:
        return f"[{section_title}]\n{text}"
    return text


def chunk_blocks(blocks: list[ParsedBlock], doc_id: str) -> list[Chunk]:
    """
    Convert ParsedBlocks into Chunks ready for embedding.

    - Tables: one chunk per table (atomic — no splitting).
    - Narrative/Footnote: split with RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.narrative_chunk_size,
        chunk_overlap=settings.narrative_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []

    for block in blocks:
        section_id = _make_section_id(doc_id, block.section_title, block.page_number)

        if block.section_type == "table":
            # Tables are atomic — single chunk regardless of size
            prefixed = _prefix_text(block.text, block.section_title)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=prefixed,
                    page_number=block.page_number,
                    section_type="table",
                    section_title=block.section_title,
                    section_id=section_id,
                    table_id=block.table_id,
                    content_hash=_content_hash(prefixed),
                )
            )
        else:
            # Narrative / footnote — split by tokens
            sub_texts = splitter.split_text(block.text)
            for sub in sub_texts:
                prefixed = _prefix_text(sub, block.section_title)
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=prefixed,
                        page_number=block.page_number,
                        section_type=block.section_type,
                        section_title=block.section_title,
                        section_id=section_id,
                        table_id=None,
                        content_hash=_content_hash(prefixed),
                    )
                )

    logger.info(
        "chunking_done",
        extra={"doc_id": doc_id, "blocks": len(blocks), "chunks": len(chunks)},
    )
    return chunks
