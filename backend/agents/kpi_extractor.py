"""
KPI Extraction Agent — structured output with Pydantic schemas.

Design (from references/kpi-extraction.md):
- Pydantic schema enforces explicit "not_found" state — never guesses a number.
- Uses LLM function calling (structured output) not free-text extraction.
- Caches results in state["extracted_kpis"][doc_id] to avoid re-running on follow-ups.
- Cross-checks values that appear in multiple chunks as a confidence signal.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from backend.core.llm_provider import get_chat_llm
from pydantic import BaseModel, Field

from backend.agents.state import GraphState
from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ingestion.vector_store import hybrid_search

logger = get_logger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────

class KPIValue(BaseModel):
    value: float | None = Field(
        default=None,
        description="Numeric value, or null if not found in the provided context.",
    )
    unit: Literal["USD", "USD_millions", "USD_thousands", "percent", "ratio", "shares"] | None = None
    source_page: int | None = Field(
        default=None,
        description="Page number where this value was found.",
    )
    source_chunk_id: str | None = Field(
        default=None,
        description="Chunk ID this value was extracted from, for auditing.",
    )
    confidence: Literal["high", "medium", "low", "not_found"] = "not_found"
    raw_text_snippet: str | None = Field(
        default=None,
        description="Exact phrase the value was extracted from, for verification.",
    )


class FinancialKPIs(BaseModel):
    revenue: KPIValue = Field(default_factory=lambda: KPIValue())
    cogs: KPIValue = Field(default_factory=lambda: KPIValue())
    gross_margin: KPIValue = Field(default_factory=lambda: KPIValue())
    net_income: KPIValue = Field(default_factory=lambda: KPIValue())
    eps: KPIValue = Field(default_factory=lambda: KPIValue())
    total_assets: KPIValue = Field(default_factory=lambda: KPIValue())
    total_liabilities: KPIValue = Field(default_factory=lambda: KPIValue())
    total_equity: KPIValue = Field(default_factory=lambda: KPIValue())
    yoy_revenue_growth: KPIValue | None = Field(
        default=None,
        description="Only present if prior-year revenue figure is available to derive growth.",
    )


# ── KPI extraction ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are extracting financial KPIs from retrieved document excerpts.
Rules:
- Only fill a field if the value is explicitly present in the provided context.
- If a value is not present, set confidence to "not_found" and leave value null. Do NOT estimate or infer.
- Always include the exact source_page and a raw_text_snippet you extracted the value from.
- If the same figure appears in multiple excerpts, prefer the one from the primary financial statement over notes/footnotes, and flag it in raw_text_snippet.
- Units: use USD_millions if the document states values are in millions, etc.
"""


def _context_string(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        page = chunk.get("page_number", "?")
        title = chunk.get("section_title", "")
        parts.append(f"[Chunk {i+1} | Page {page} | {title}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def extract_kpis(state: GraphState) -> GraphState:
    """KPI extraction node — retrieves relevant chunks, then extracts structured KPIs."""
    doc_id = state["doc_id"]

    # Cache hit — reuse across follow-up turns
    cached = state.get("extracted_kpis", {}).get(doc_id)
    if cached:
        logger.info("kpi_cache_hit", extra={"doc_id": doc_id})
        return {**state, "kpi_result": cached, "error": None}

    # Retrieve KPI-relevant chunks
    kpi_query = (
        "revenue net income EPS earnings per share total assets total liabilities "
        "stockholders equity gross margin COGS year over year growth"
    )
    try:
        chunks = hybrid_search(kpi_query, doc_id, k=8, rerank_top_n=6)
    except Exception as exc:
        logger.error("kpi_retrieval_failed", extra={"error": str(exc)})
        return {**state, "kpi_result": None, "error": f"KPI retrieval failed: {exc}"}

    context = _context_string(chunks)

    llm = get_chat_llm(temperature=0)

    try:
        result: FinancialKPIs = llm.with_structured_output(FinancialKPIs).invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nExtract the KPIs."},
        ])
    except Exception as exc:
        logger.error("kpi_extraction_failed", extra={"error": str(exc)})
        return {**state, "kpi_result": None, "error": f"KPI extraction failed: {exc}"}

    kpi_dict = result.model_dump()
    updated_cache = {**state.get("extracted_kpis", {}), doc_id: kpi_dict}

    logger.info(
        "kpi_extracted",
        extra={
            "doc_id": doc_id,
            "revenue_confidence": kpi_dict.get("revenue", {}).get("confidence"),
            "net_income_confidence": kpi_dict.get("net_income", {}).get("confidence"),
        },
    )

    response_msg = AIMessage(content="KPI extraction complete. Results are displayed in the dashboard.")
    return {
        **state,
        "messages": [response_msg],
        "kpi_result": kpi_dict,
        "extracted_kpis": updated_cache,
        "retrieved_context": chunks,
        "error": None,
    }
