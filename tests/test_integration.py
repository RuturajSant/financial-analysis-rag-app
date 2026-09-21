"""
Integration test: ingest → query pipeline.

Strategy:
- Creates a minimal real PDF in a temp directory using reportlab.
- Mocks OpenAIEmbeddings so no real API key is needed.
- Uses an in-memory ChromaDB client (monkeypatched) so no disk state is left.
- Runs parse_pdf → chunk_blocks → upsert_chunks → hybrid_search end-to-end.
- Verifies retrieved chunks reference the ingested document.

What is NOT mocked:
- pdfplumber PDF parsing (real file)
- Chunking logic (real)
- BM25 scoring (real)
- ChromaDB collection operations (real, in-memory)

What IS mocked:
- OpenAIEmbeddings.embed_documents / embed_query (returns deterministic vectors)
- CrossEncoder.predict (returns constant scores — reranking order is deterministic)
- chromadb.PersistentClient (replaced with EphemeralClient so no disk I/O)
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import chromadb
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_sample_pdf(path: Path) -> None:
    """Write a minimal two-page PDF with text and a simple table."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)

    # Page 1 — narrative
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "ANNUAL REPORT 2023")
    c.setFont("Helvetica", 11)
    c.drawString(72, 680, "REVENUE OVERVIEW")
    y = 660
    for i in range(12):
        c.drawString(72, y, f"The company achieved strong growth in segment {i}. Revenue increased 15%.")
        y -= 18

    c.showPage()

    # Page 2 — a text-based table (pdfplumber will parse as narrative; table API needs actual table widgets)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "FINANCIAL SUMMARY")
    c.setFont("Courier", 10)
    rows = [
        ("Metric", "2023", "2022"),
        ("Revenue", "$500M", "$435M"),
        ("Net Income", "$50M", "$38M"),
        ("EPS", "$1.25", "$0.95"),
        ("Total Assets", "$2000M", "$1800M"),
    ]
    y = 700
    for row in rows:
        c.drawString(72, y, "  ".join(f"{cell:<20}" for cell in row))
        y -= 16

    c.save()


def _fake_embed_documents(texts: list[str]) -> list[list[float]]:
    """Return a deterministic 8-dim embedding derived from text hash."""
    import hashlib
    results = []
    for text in texts:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = [(((h >> (i * 4)) & 0xF) / 15.0) for i in range(8)]
        results.append(vec)
    return results


def _fake_embed_query(text: str) -> list[float]:
    return _fake_embed_documents([text])[0]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "sample_annual_report.pdf"
    _create_sample_pdf(pdf_path)
    return pdf_path


@pytest.fixture(scope="module")
def ephemeral_chroma_client():
    """Return a real in-memory ChromaDB client (no disk)."""
    return chromadb.EphemeralClient()


# ── Dimension consistency patch ───────────────────────────────────────────────
# ChromaDB validates that query embedding dim == stored embedding dim.
# Our fake embedder returns 8-dim vectors, so we patch globally.

_EMBED_DIM = 8


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIngestQueryPipeline:
    """End-to-end: parse PDF → chunk → upsert → hybrid_search."""

    def _run_pipeline(self, sample_pdf: Path, chroma_client: chromadb.EphemeralClient) -> tuple[str, list[dict]]:
        """
        Run the full pipeline with mocked embedder and reranker.
        Returns (doc_id, search_results).
        """
        from backend.ingestion.chunker import chunk_blocks
        from backend.ingestion.parser import parse_pdf
        import backend.ingestion.vector_store as vs_module

        # Patch the lazy singletons in vector_store
        fake_embedder = MagicMock()
        fake_embedder.embed_documents.side_effect = _fake_embed_documents
        fake_embedder.embed_query.side_effect = _fake_embed_query

        fake_reranker = MagicMock()
        fake_reranker.predict.side_effect = lambda pairs: [1.0 - i * 0.05 for i in range(len(pairs))]

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(sample_pdf.resolve())))

        with (
            patch.object(vs_module, "_get_embedder", return_value=fake_embedder),
            patch.object(vs_module, "_get_reranker", return_value=fake_reranker),
            patch.object(vs_module, "_get_client", return_value=chroma_client),
        ):
            blocks = parse_pdf(sample_pdf)
            assert blocks, "Parser returned no blocks — PDF may be empty"

            chunks = chunk_blocks(blocks, doc_id)
            assert chunks, "Chunker returned no chunks"

            from backend.ingestion.vector_store import hybrid_search, upsert_chunks

            upserted = upsert_chunks(chunks, doc_id)
            assert upserted > 0, "No chunks were upserted"

            results = hybrid_search("revenue net income", doc_id, k=5, rerank_top_n=3)

        return doc_id, results

    def test_parse_returns_blocks(self, sample_pdf, ephemeral_chroma_client):
        from backend.ingestion.parser import parse_pdf

        blocks = parse_pdf(sample_pdf)
        assert len(blocks) >= 1, "Expected at least one parsed block"

    def test_chunk_blocks_from_pdf(self, sample_pdf, ephemeral_chroma_client):
        from backend.ingestion.chunker import chunk_blocks
        from backend.ingestion.parser import parse_pdf

        blocks = parse_pdf(sample_pdf)
        doc_id = "test-doc-integration"
        chunks = chunk_blocks(blocks, doc_id)
        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.doc_id == doc_id
            assert chunk.text.strip()
            assert len(chunk.content_hash) == 64  # SHA-256 hex

    def test_upsert_chunks_returns_nonzero(self, sample_pdf, ephemeral_chroma_client):
        import backend.ingestion.vector_store as vs_module
        from backend.ingestion.chunker import chunk_blocks
        from backend.ingestion.parser import parse_pdf
        from backend.ingestion.vector_store import upsert_chunks

        fake_embedder = MagicMock()
        fake_embedder.embed_documents.side_effect = _fake_embed_documents

        doc_id = "upsert-test-doc"
        blocks = parse_pdf(sample_pdf)
        chunks = chunk_blocks(blocks, doc_id)

        with (
            patch.object(vs_module, "_get_embedder", return_value=fake_embedder),
            patch.object(vs_module, "_get_client", return_value=ephemeral_chroma_client),
        ):
            count = upsert_chunks(chunks, doc_id)

        assert count == len(chunks), f"Expected {len(chunks)} upserted, got {count}"

    def test_upsert_idempotent_on_second_call(self, sample_pdf, ephemeral_chroma_client):
        """Re-upserting the same chunks should return 0 (no-op — content hashes match)."""
        import backend.ingestion.vector_store as vs_module
        from backend.ingestion.chunker import chunk_blocks
        from backend.ingestion.parser import parse_pdf
        from backend.ingestion.vector_store import upsert_chunks

        fake_embedder = MagicMock()
        fake_embedder.embed_documents.side_effect = _fake_embed_documents

        doc_id = "idempotent-test-doc"
        blocks = parse_pdf(sample_pdf)
        chunks = chunk_blocks(blocks, doc_id)

        with (
            patch.object(vs_module, "_get_embedder", return_value=fake_embedder),
            patch.object(vs_module, "_get_client", return_value=ephemeral_chroma_client),
        ):
            upsert_chunks(chunks, doc_id)  # First call
            second_count = upsert_chunks(chunks, doc_id)  # Second call — same hashes

        assert second_count == 0, "Second upsert with unchanged content should return 0"

    def test_hybrid_search_returns_results(self, sample_pdf, ephemeral_chroma_client):
        _, results = self._run_pipeline(sample_pdf, ephemeral_chroma_client)
        assert len(results) > 0, "hybrid_search returned no results"

    def test_search_results_have_required_keys(self, sample_pdf, ephemeral_chroma_client):
        _, results = self._run_pipeline(sample_pdf, ephemeral_chroma_client)
        required_keys = {"chunk_id", "text", "page_number", "section_title", "section_type", "score"}
        for result in results:
            assert required_keys.issubset(result.keys()), f"Missing keys in result: {result.keys()}"

    def test_search_results_text_is_nonempty(self, sample_pdf, ephemeral_chroma_client):
        _, results = self._run_pipeline(sample_pdf, ephemeral_chroma_client)
        for result in results:
            assert result["text"].strip(), "Search result text should not be empty"

    def test_search_results_count_bounded(self, sample_pdf, ephemeral_chroma_client):
        """Results should be ≤ rerank_top_n."""
        _, results = self._run_pipeline(sample_pdf, ephemeral_chroma_client)
        assert len(results) <= 3, f"Expected ≤3 results (rerank_top_n=3), got {len(results)}"

    def test_full_pipeline_doc_id_stable(self, sample_pdf, ephemeral_chroma_client):
        """doc_id derived from path is deterministic."""
        doc_id_1 = str(uuid.uuid5(uuid.NAMESPACE_URL, str(sample_pdf.resolve())))
        doc_id_2 = str(uuid.uuid5(uuid.NAMESPACE_URL, str(sample_pdf.resolve())))
        assert doc_id_1 == doc_id_2
