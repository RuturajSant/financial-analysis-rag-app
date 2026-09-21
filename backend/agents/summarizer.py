"""
Summary Agent — map-reduce for whole-report, single-pass for section summaries.

Design (from references/orchestration.md):
- Whole-report: map-reduce (chunk-level summaries combined) to stay within context window.
- Section: single retrieval-then-summarize pass.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from backend.core.llm_provider import get_chat_llm

from backend.agents.state import GraphState
from backend.core.logger import get_logger
from backend.ingestion.vector_store import hybrid_search

logger = get_logger(__name__)

_MAP_SYSTEM = """\
You are a financial analyst. Summarize the following excerpt from an annual report in 2-4 sentences.
Focus on key financial figures, business highlights, or risks. Be factual and concise.
"""

_REDUCE_SYSTEM = """\
You are a financial analyst. Combine the following section-level summaries into a single coherent
executive summary of the annual report. Present it in 3-5 paragraphs covering:
1. Business overview and revenue performance
2. Profitability and margins
3. Balance sheet health and cash flow
4. Key risks or forward-looking notes (if present)
Do not invent information not in the provided summaries.
"""

_SECTION_SYSTEM = """\
You are a financial analyst. Summarize the following section of a financial report in 3-6 sentences.
Be specific about numbers and be factual. Do not invent information not in the provided context.
"""


def _map_reduce_summary(chunks: list[dict], llm: ChatOpenAI) -> str:
    """Summarize each chunk, then combine into a final summary."""
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        try:
            resp = llm.invoke([
                SystemMessage(content=_MAP_SYSTEM),
                HumanMessage(content=chunk["text"]),
            ])
            chunk_summaries.append(resp.content.strip())
        except Exception as exc:
            logger.warning("map_chunk_failed", extra={"chunk_index": i, "error": str(exc)})
            chunk_summaries.append("[Summary unavailable for this section]")

    combined = "\n\n".join(f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries))

    try:
        final = llm.invoke([
            SystemMessage(content=_REDUCE_SYSTEM),
            HumanMessage(content=combined),
        ])
        return final.content.strip()
    except Exception as exc:
        logger.error("reduce_failed", extra={"error": str(exc)})
        return combined  # fallback: return concatenated chunk summaries


def summarize(state: GraphState) -> GraphState:
    """Summary Agent node — whole-report (map-reduce) or section (single-pass)."""
    doc_id = state["doc_id"]
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Detect whether this is a whole-report or section request
    whole_report_signals = {"summary", "summarize", "overview", "executive summary", "whole", "entire", "full report"}
    is_whole_report = any(sig in query.lower() for sig in whole_report_signals)

    llm = get_chat_llm(temperature=0.2)

    try:
        if is_whole_report:
            # Retrieve a broad set of chunks covering the whole document
            chunks = hybrid_search(
                "executive overview revenue profit assets equity cash flow risks",
                doc_id,
                k=16,
                rerank_top_n=12,
            )
            summary = _map_reduce_summary(chunks, llm)
        else:
            # Section-level: targeted retrieval then single-pass summarization
            chunks = hybrid_search(query, doc_id, k=8, rerank_top_n=5)
            context = "\n\n---\n\n".join(
                f"[Page {c.get('page_number')} | {c.get('section_title', '')}]\n{c['text']}"
                for c in chunks
            )
            resp = llm.invoke([
                SystemMessage(content=_SECTION_SYSTEM),
                HumanMessage(content=f"Section context:\n{context}\n\nUser request: {query}"),
            ])
            summary = resp.content.strip()
            chunks = chunks

    except Exception as exc:
        logger.error("summarization_failed", extra={"error": str(exc)})
        return {**state, "summary_result": None, "error": f"Summarization failed: {exc}"}

    logger.info(
        "summary_generated",
        extra={"doc_id": doc_id, "is_whole_report": is_whole_report, "chunks_used": len(chunks)},
    )

    return {
        **state,
        "messages": [AIMessage(content=summary)],
        "summary_result": summary,
        "retrieved_context": chunks,
        "error": None,
    }
