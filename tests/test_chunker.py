"""
Unit tests for backend/ingestion/chunker.py.

Covers:
- _content_hash: deterministic SHA-256
- _prefix_text: section title embedding
- _make_section_id: stable MD5-based IDs
- chunk_blocks:
    - table blocks produce exactly ONE chunk (atomic)
    - long narrative blocks split into MULTIPLE chunks
    - all chunks carry correct metadata (doc_id, section_type, content_hash)
    - chunks have non-empty content_hash
"""
from __future__ import annotations

import hashlib
import unittest

from backend.ingestion.chunker import (
    Chunk,
    _content_hash,
    _make_section_id,
    _prefix_text,
    chunk_blocks,
)
from backend.ingestion.parser import ParsedBlock


# ── Helper factories ──────────────────────────────────────────────────────────

def _table_block(text: str = "Revenue\t100\nCOGS\t60", page: int = 1) -> ParsedBlock:
    return ParsedBlock(
        page_number=page,
        section_type="table",
        text=text,
        table_id=f"p{page}_t0",
        section_title="Financials",
    )


def _narrative_block(text: str, page: int = 1, title: str | None = "Overview") -> ParsedBlock:
    return ParsedBlock(
        page_number=page,
        section_type="narrative",
        text=text,
        section_title=title,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestContentHash(unittest.TestCase):
    def test_deterministic(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        self.assertEqual(h1, h2)

    def test_different_inputs_different_hashes(self):
        self.assertNotEqual(_content_hash("abc"), _content_hash("xyz"))

    def test_is_sha256(self):
        text = "test"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(_content_hash(text), expected)


class TestPrefixText(unittest.TestCase):
    def test_with_title(self):
        result = _prefix_text("Some paragraph.", "Revenue Analysis")
        self.assertEqual(result, "[Revenue Analysis]\nSome paragraph.")

    def test_without_title(self):
        result = _prefix_text("Some paragraph.", None)
        self.assertEqual(result, "Some paragraph.")

    def test_empty_title_treated_as_none(self):
        # Empty string is falsy — behaves same as None
        result = _prefix_text("text", "")
        self.assertEqual(result, "text")


class TestMakeSectionId(unittest.TestCase):
    def test_stable_for_same_inputs(self):
        id1 = _make_section_id("doc-1", "Revenue", 3)
        id2 = _make_section_id("doc-1", "Revenue", 3)
        self.assertEqual(id1, id2)

    def test_different_docs_produce_different_ids(self):
        id1 = _make_section_id("doc-1", "Revenue", 3)
        id2 = _make_section_id("doc-2", "Revenue", 3)
        self.assertNotEqual(id1, id2)

    def test_none_title_is_stable(self):
        id1 = _make_section_id("doc-1", None, 1)
        id2 = _make_section_id("doc-1", None, 1)
        self.assertEqual(id1, id2)

    def test_length_12(self):
        sid = _make_section_id("doc-1", "Title", 1)
        self.assertEqual(len(sid), 12)


class TestChunkBlocksTableAtomicity(unittest.TestCase):
    """Tables MUST produce exactly one chunk regardless of size."""

    DOC_ID = "test-doc-table"

    def test_single_table_produces_one_chunk(self):
        block = _table_block()
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertEqual(len(chunks), 1)

    def test_large_table_still_one_chunk(self):
        # Build a large table that would exceed narrative_chunk_size
        rows = "\n".join(f"Row {i}\t{i * 1000}" for i in range(200))
        block = _table_block(text=rows)
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertEqual(len(chunks), 1, "Large table must not be split")

    def test_table_chunk_has_correct_section_type(self):
        chunks = chunk_blocks([_table_block()], self.DOC_ID)
        self.assertEqual(chunks[0].section_type, "table")

    def test_table_chunk_preserves_table_id(self):
        block = _table_block(page=3)
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertEqual(chunks[0].table_id, "p3_t0")

    def test_table_chunk_has_content_hash(self):
        chunks = chunk_blocks([_table_block()], self.DOC_ID)
        self.assertTrue(len(chunks[0].content_hash) > 0)

    def test_multiple_tables_one_chunk_each(self):
        blocks = [_table_block(page=1), _table_block(page=2)]
        chunks = chunk_blocks(blocks, self.DOC_ID)
        self.assertEqual(len(chunks), 2)


class TestChunkBlocksNarrativeSplitting(unittest.TestCase):
    """Long narrative text must be split into multiple chunks."""

    DOC_ID = "test-doc-narrative"

    def _long_narrative(self, sentences: int = 100) -> str:
        # Each sentence ~50 chars → total >> default chunk_size=500
        return "  ".join(
            f"The company achieved record revenue growth in quarter {i}." for i in range(sentences)
        )

    def test_short_narrative_is_single_chunk(self):
        block = _narrative_block("Short text.")
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertEqual(len(chunks), 1)

    def test_long_narrative_splits_into_multiple_chunks(self):
        block = _narrative_block(self._long_narrative(100))
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertGreater(len(chunks), 1, "Long narrative must be split")

    def test_narrative_chunks_have_no_table_id(self):
        block = _narrative_block("Short narrative text.")
        chunks = chunk_blocks([block], self.DOC_ID)
        for chunk in chunks:
            self.assertIsNone(chunk.table_id)

    def test_narrative_section_type_preserved(self):
        block = _narrative_block("Text.")
        chunks = chunk_blocks([block], self.DOC_ID)
        for chunk in chunks:
            self.assertEqual(chunk.section_type, "narrative")

    def test_footnote_section_type_preserved(self):
        block = ParsedBlock(
            page_number=1,
            section_type="footnote",
            text="Note 1: This is a footnote.",
            section_title=None,
        )
        chunks = chunk_blocks([block], self.DOC_ID)
        for chunk in chunks:
            self.assertEqual(chunk.section_type, "footnote")


class TestChunkBlocksMetadata(unittest.TestCase):
    """All chunks must carry correct doc_id, page, and unique chunk_ids."""

    DOC_ID = "test-doc-meta"

    def test_doc_id_propagated(self):
        blocks = [_table_block(), _narrative_block("Some text.")]
        chunks = chunk_blocks(blocks, self.DOC_ID)
        for chunk in chunks:
            self.assertEqual(chunk.doc_id, self.DOC_ID)

    def test_page_number_propagated(self):
        block = _table_block(page=7)
        chunks = chunk_blocks([block], self.DOC_ID)
        self.assertEqual(chunks[0].page_number, 7)

    def test_chunk_ids_are_unique(self):
        long_text = "  ".join(f"Sentence number {i} with content." for i in range(200))
        blocks = [_narrative_block(long_text), _table_block()]
        chunks = chunk_blocks(blocks, self.DOC_ID)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "chunk_ids must be unique")

    def test_content_hash_matches_text(self):
        block = _table_block(text="Rev\t500\nNet\t100")
        chunks = chunk_blocks([block], self.DOC_ID)
        chunk = chunks[0]
        expected_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        self.assertEqual(chunk.content_hash, expected_hash)

    def test_section_title_embedded_in_text(self):
        block = _narrative_block("Paragraph content.", title="Risk Factors")
        chunks = chunk_blocks([block], self.DOC_ID)
        for chunk in chunks:
            self.assertIn("Risk Factors", chunk.text)

    def test_mixed_blocks_preserve_order_types(self):
        blocks = [
            _table_block(page=1),
            _narrative_block("Narrative text here.", page=2),
            _table_block(page=3),
        ]
        chunks = chunk_blocks(blocks, self.DOC_ID)
        # First and last should be table chunks
        self.assertEqual(chunks[0].section_type, "table")
        self.assertEqual(chunks[-1].section_type, "table")


if __name__ == "__main__":
    unittest.main()
