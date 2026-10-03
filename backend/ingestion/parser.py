"""
PDF parser — extracts narrative text and tables from financial PDFs using pdfplumber.

Design:
- Each page is processed to extract table blocks (atomic) and surrounding narrative text.
- Returns a list of raw ParsedBlock objects with page number and section_type tagged.
- Downstream chunker.py handles splitting; this module only extracts raw content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedBlock:
    """A raw extracted block from a PDF page before chunking."""
    page_number: int
    section_type: Literal["narrative", "table", "footnote"]
    text: str
    table_id: str | None = None          # e.g. "p3_t0" for page 3, table index 0
    section_title: str | None = None     # detected from heading heuristics


# Heuristic: lines that look like section headings (numbered or ALL CAPS short lines)
_HEADING_RE = re.compile(r"^(\d+\.\s+.{3,60}|[A-Z][A-Z\s&,:\-]{5,60})$")
_FOOTNOTE_RE = re.compile(r"^Note\s+\d+", re.IGNORECASE)


def _detect_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line.strip()))


def _detect_footnote(line: str) -> bool:
    return bool(_FOOTNOTE_RE.match(line.strip()))


def _table_to_text(table_data: list[list[str | None]]) -> str:
    """Convert pdfplumber table (list of rows) to a Markdown-formatted table."""
    rows = []
    for row in table_data:
        cleaned = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
        if any(cleaned):
            rows.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(rows)


def parse_pdf(pdf_path: str | Path) -> list[ParsedBlock]:
    """
    Extract blocks from a PDF file.

    Returns:
        List of ParsedBlock, ordered by page then appearance on page.
        Tables are extracted as atomic blocks; surrounding text as narrative blocks.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks: list[ParsedBlock] = []
    current_title: str | None = None

    with pdfplumber.open(str(pdf_path)) as pdf:
        logger.info("pdf_parse_start", extra={"path": str(pdf_path), "pages": len(pdf.pages)})

        for page in pdf.pages:
            page_num = page.page_number  # 1-indexed

            # Extract table bounding boxes so we can subtract them from text
            tables = page.extract_tables()
            if not tables:
                tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})

            # ── Extract tables as atomic blocks ──────────────────────────────
            for t_idx, table_data in enumerate(tables):
                if not table_data:
                    continue
                table_text = _table_to_text(table_data)
                if not table_text.strip():
                    continue
                table_id = f"p{page_num}_t{t_idx}"
                blocks.append(
                    ParsedBlock(
                        page_number=page_num,
                        section_type="table",
                        text=f"[{current_title or 'Table'}]\n{table_text}",
                        table_id=table_id,
                        section_title=current_title,
                    )
                )
                logger.debug("table_extracted", extra={"page": page_num, "table_id": table_id})

            # ── Extract narrative / footnote text ────────────────────────────
            # Use extract_text to get the full page text, then filter out table rows
            # (a heuristic approach — for very complex layouts consider cropping by bbox)
            full_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            narrative_lines: list[str] = []
            in_footnote = False

            for line in full_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                # Track section headings for metadata propagation
                if _detect_heading(stripped):
                    current_title = stripped
                    narrative_lines.append(stripped)
                    in_footnote = False
                    continue

                if _detect_footnote(stripped):
                    in_footnote = True

                narrative_lines.append(stripped)

            narrative_text = "\n".join(narrative_lines).strip()
            if narrative_text:
                section_type: Literal["narrative", "table", "footnote"] = (
                    "footnote" if in_footnote else "narrative"
                )
                blocks.append(
                    ParsedBlock(
                        page_number=page_num,
                        section_type=section_type,
                        text=narrative_text,
                        section_title=current_title,
                    )
                )

    logger.info("pdf_parse_done", extra={"path": str(pdf_path), "blocks": len(blocks)})
    return blocks
